from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import os


class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1,
                                                                                     center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[ i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and  df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        return frame_nums_with_ball_hits


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'rb') as f:
                    ball_detections = pickle.load(f)
                return ball_detections
            except FileNotFoundError:
                print(f"Warning: The file '{stub_path}' does not exist. Proceeding with frame detection.")
            except Exception as e:
                print(f"An error occurred while loading the stub: {e}")

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump(ball_detections, f)
            except Exception as e:
                print(f"An error occurred while saving the stub: {e}")

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def draw_bboxes(self, video_frames, player_detections, ball_shot_frames=None):
        output_video_frames = []
        trajectory_points = []  # Store ball center positions for trajectory
        trajectory_length = 15  # Number of previous positions to show
        frames_without_ball = 0  # Track consecutive frames without ball detection

        for frame_idx, (frame, ball_dict) in enumerate(zip(video_frames, player_detections)):
            if not ball_dict:
                # No ball detected - increment counter and potentially clear trajectory
                frames_without_ball += 1
                if frames_without_ball > 5:
                    # Clear trajectory after 5 frames without detection
                    trajectory_points.clear()
                output_video_frames.append(frame)
                continue

            # Ball detected - reset counter
            frames_without_ball = 0

            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Check if this point is reasonable (not a huge jump from last point)
                if trajectory_points:
                    last_x, last_y = trajectory_points[-1]
                    distance = ((center_x - last_x)**2 + (center_y - last_y)**2)**0.5
                    # If ball "jumped" more than 200 pixels, clear trajectory (likely false detection)
                    if distance > 200:
                        trajectory_points.clear()

                # Add to trajectory
                trajectory_points.append((center_x, center_y))
                if len(trajectory_points) > trajectory_length:
                    trajectory_points.pop(0)

                # Draw trajectory trail with fading effect (only if we have enough points)
                if len(trajectory_points) >= 3:
                    for i in range(1, len(trajectory_points)):
                        alpha = i / len(trajectory_points)  # Fade effect
                        color = (0, int(255 * alpha), int(255 * alpha))  # Yellow fading
                        thickness = max(1, int(3 * alpha))
                        cv2.line(frame, trajectory_points[i-1], trajectory_points[i], color, thickness)

                # Draw ball circle
                cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                cv2.circle(frame, (center_x, center_y), 10, (0, 200, 200), 2)

            # Draw shot indicator if this is a shot frame (subtle highlight only)
            if ball_shot_frames and frame_idx in ball_shot_frames:
                if ball_dict:
                    bbox = list(ball_dict.values())[0]
                    cx, cy = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)
                    cv2.circle(frame, (cx, cy), 30, (0, 0, 255), 3)

            output_video_frames.append(frame)

        return output_video_frames







