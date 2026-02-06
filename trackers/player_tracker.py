from ultralytics import YOLO
import cv2
import pickle
import os
import sys
import numpy as np
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox


class PlayerTracker:
    """
    Padel Player Tracker - Detects and tracks 4 players (2 teams of 2)

    Player IDs:
    - Team 1 (bottom/near side): Players 1, 2 (green markers)
    - Team 2 (top/far side): Players 3, 4 (red markers)

    Within each team, players are distinguished by X position (left/right)
    """

    # Calibrated court boundaries for 1920x1080
    # These define the actual playing area polygon
    COURT_POLYGON = np.array([
        [583, 323],    # Far-Left (top-left)
        [1343, 323],   # Far-Right (top-right)
        [1696, 942],   # Near-Right (bottom-right)
        [228, 944],    # Near-Left (bottom-left)
    ], dtype=np.float32)

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # For temporal consistency - track last known positions
        self.last_positions = {}  # {player_id: (center_x, center_y)}
        self.frames_since_seen = {}  # {player_id: count}

    def point_in_court(self, x, y, frame_width, frame_height):
        """Check if a point is inside the court polygon"""
        # Scale court polygon to frame size
        scale_x = frame_width / 1920.0
        scale_y = frame_height / 1080.0
        scaled_polygon = self.COURT_POLYGON.copy()
        scaled_polygon[:, 0] *= scale_x
        scaled_polygon[:, 1] *= scale_y

        # Add margin around court (players can be slightly outside lines)
        # Expand polygon by ~5% on each side
        center_x = scaled_polygon[:, 0].mean()
        center_y = scaled_polygon[:, 1].mean()
        expanded_polygon = scaled_polygon.copy()
        for i in range(4):
            dx = scaled_polygon[i, 0] - center_x
            dy = scaled_polygon[i, 1] - center_y
            expanded_polygon[i, 0] = center_x + dx * 1.08
            expanded_polygon[i, 1] = center_y + dy * 1.08

        # Point in polygon test
        result = cv2.pointPolygonTest(expanded_polygon.reshape(-1, 1, 2), (x, y), False)
        return result >= 0

    def choose_and_filter_players(self, court_keypoints, player_detections):
        # Player IDs are already assigned in detect_frame based on court position
        return player_detections

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if not frames:
            print("No frames provided for detection.")
            return []

        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'rb') as f:
                    player_detections = pickle.load(f)
                return player_detections
            except FileNotFoundError:
                print(f"Warning: The file '{stub_path}' does not exist. Proceeding with frame detection.")
            except Exception as e:
                print(f"An error occurred while loading the stub: {e}")

        # Reset tracking state for new detection run
        self.last_positions = {}
        self.frames_since_seen = {1: 0, 2: 0, 3: 0, 4: 0}

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump(player_detections, f)
            except Exception as e:
                print(f"An error occurred while saving the stub: {e}")

        return player_detections

    def detect_frame(self, frame, court_bounds=None):
        """Detect up to 4 players and assign IDs with temporal consistency"""
        results = self.model.predict(frame, verbose=False)[0]
        id_name_dict = results.names

        frame_height, frame_width = frame.shape[:2]
        mid_y = frame_height * 0.5

        # Collect all valid player detections with confidence
        candidates = []
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            conf = box.conf.tolist()[0] if hasattr(box, 'conf') else 0.5

            if object_cls_name == "person":
                x1, y1, x2, y2 = result
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                bbox_height = y2 - y1
                bbox_width = x2 - x1

                # Relaxed filter: Keep players in broader court area
                min_y = frame_height * 0.10
                max_y = frame_height * 0.98
                min_x = frame_width * 0.05
                max_x = frame_width * 0.95

                # Relaxed size filter - allow smaller players at far court
                y_ratio = center_y / frame_height
                min_height = frame_height * (0.03 + 0.08 * y_ratio)
                max_height = frame_height * 0.60

                aspect_ratio = bbox_height / max(bbox_width, 1)

                foot_x = center_x
                foot_y = y2

                in_court = self.point_in_court(foot_x, foot_y, frame_width, frame_height)

                if (min_y < center_y < max_y and
                    min_x < center_x < max_x and
                    min_height < bbox_height < max_height and
                    aspect_ratio > 0.8 and
                    in_court):
                    candidates.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'bbox': result,
                        'conf': conf,
                        'height': bbox_height
                    })

        # Sort by confidence
        candidates.sort(key=lambda x: x['conf'], reverse=True)
        candidates = candidates[:4]  # Keep top 4

        player_dict = {}

        # TEMPORAL CONSISTENCY: If we have previous positions, match by proximity
        if self.last_positions and len(candidates) >= 2:
            used_candidates = set()
            max_distance = frame_width * 0.15  # Max movement between frames

            # Match each known player to nearest candidate
            for player_id in [1, 2, 3, 4]:
                if player_id not in self.last_positions:
                    continue

                last_pos = self.last_positions[player_id]
                best_dist = float('inf')
                best_idx = -1

                for idx, c in enumerate(candidates):
                    if idx in used_candidates:
                        continue

                    # Check team constraint: P1,P2 should stay in bottom, P3,P4 in top
                    if player_id in [1, 2] and c['center_y'] < mid_y * 0.7:
                        continue  # Skip if Team 1 player detected too far up
                    if player_id in [3, 4] and c['center_y'] > mid_y * 1.3:
                        continue  # Skip if Team 2 player detected too far down

                    dist = measure_distance((c['center_x'], c['center_y']), last_pos)
                    if dist < best_dist and dist < max_distance:
                        best_dist = dist
                        best_idx = idx

                if best_idx >= 0:
                    player_dict[player_id] = candidates[best_idx]['bbox']
                    used_candidates.add(best_idx)
                    self.frames_since_seen[player_id] = 0

            # Assign remaining candidates using position-based logic
            remaining = [c for idx, c in enumerate(candidates) if idx not in used_candidates]
            unassigned_ids = [pid for pid in [1, 2, 3, 4] if pid not in player_dict]

            if remaining and unassigned_ids:
                # Split remaining by court half
                for c in remaining:
                    if c['center_y'] >= mid_y:  # Bottom half = Team 1
                        for pid in [1, 2]:
                            if pid in unassigned_ids:
                                player_dict[pid] = c['bbox']
                                unassigned_ids.remove(pid)
                                break
                    else:  # Top half = Team 2
                        for pid in [3, 4]:
                            if pid in unassigned_ids:
                                player_dict[pid] = c['bbox']
                                unassigned_ids.remove(pid)
                                break

        else:
            # INITIAL ASSIGNMENT: First frame or no previous data
            if len(candidates) >= 4:
                candidates.sort(key=lambda x: x['center_y'])
                team2_players = candidates[:2]
                team1_players = candidates[-2:]

                team1_players.sort(key=lambda x: x['center_x'])
                team2_players.sort(key=lambda x: x['center_x'])

                player_dict[1] = team1_players[0]['bbox']
                player_dict[2] = team1_players[1]['bbox']
                player_dict[3] = team2_players[0]['bbox']
                player_dict[4] = team2_players[1]['bbox']

            elif len(candidates) >= 2:
                candidates.sort(key=lambda x: x['center_y'])
                bottom = [c for c in candidates if c['center_y'] >= mid_y]
                top = [c for c in candidates if c['center_y'] < mid_y]

                if len(bottom) >= 2:
                    bottom.sort(key=lambda x: x['center_x'])
                    player_dict[1] = bottom[0]['bbox']
                    player_dict[2] = bottom[1]['bbox']
                elif len(bottom) == 1:
                    player_dict[1] = bottom[0]['bbox']

                if len(top) >= 2:
                    top.sort(key=lambda x: x['center_x'])
                    player_dict[3] = top[0]['bbox']
                    player_dict[4] = top[1]['bbox']
                elif len(top) == 1:
                    player_dict[3] = top[0]['bbox']

        # Update last known positions
        for player_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            self.last_positions[player_id] = ((x1 + x2) / 2, (y1 + y2) / 2)
            self.frames_since_seen[player_id] = 0

        # Increment frames since seen for missing players
        for pid in [1, 2, 3, 4]:
            if pid not in player_dict:
                self.frames_since_seen[pid] = self.frames_since_seen.get(pid, 0) + 1
                # Clear position if not seen for too long (30 frames = 1 second)
                if self.frames_since_seen.get(pid, 0) > 30:
                    self.last_positions.pop(pid, None)

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        """Draw markers above players with team colors"""
        output_video_frames = []

        # Team colors
        # Team 1 (bottom): Green shades
        # Team 2 (top): Red/Orange shades
        colors = {
            1: ((0, 200, 0), (0, 255, 0)),      # Team 1 - Player 1: Green
            2: ((0, 150, 100), (0, 200, 150)),  # Team 1 - Player 2: Teal
            3: ((0, 0, 200), (0, 0, 255)),      # Team 2 - Player 3: Red
            4: ((0, 100, 200), (0, 150, 255)),  # Team 2 - Player 4: Orange
        }

        for frame, player_dict in zip(video_frames, player_detections):
            for player_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                color, border_color = colors.get(player_id, ((128, 128, 128), (200, 200, 200)))

                # Triangle position: centered above player's head
                center_x = int((x1 + x2) / 2)
                top_y = int(y1) - 10

                # Triangle points
                triangle_size = 18
                pt1 = (center_x, top_y)
                pt2 = (center_x - triangle_size, top_y - triangle_size)
                pt3 = (center_x + triangle_size, top_y - triangle_size)

                pts = np.array([pt1, pt2, pt3], np.int32)
                cv2.fillPoly(frame, [pts], color)
                cv2.polylines(frame, [pts], True, border_color, 2)

                # Add player number label
                label = f"P{player_id}"
                cv2.putText(frame, label, (center_x - 10, top_y - triangle_size - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            output_video_frames.append(frame)

        return output_video_frames
