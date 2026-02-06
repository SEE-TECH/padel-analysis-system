import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                                )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0] * 28

        # point 0
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1]
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3]
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7]
        # #point 8
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        # #point 10
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
        drawing_key_points[25] = drawing_key_points[17]
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),

            (0, 1),
            (8, 9),
            (10, 11),
            (2, 3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (
        self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point,
                                   closest_key_point_index,
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):

        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position,
                                                                                               closest_key_point)

        # Conver pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )

        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = (self.drawing_key_points[closest_key_point_index * 2],
                                        self.drawing_key_points[closest_key_point_index * 2 + 1]
                                        )

        mini_court_player_position = (closest_mini_coourt_keypoint[0] + mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1] + mini_court_y_distance_pixels
                                      )

        return mini_court_player_position

    def compute_homography(self, original_court_key_points):
        """Compute homography matrix from video court points to mini court points"""
        # Use 4 corner points for homography (keypoints 0, 1, 2, 3 are the court corners)
        # Original keypoints: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        src_points = np.array([
            [original_court_key_points[0], original_court_key_points[1]],   # Point 0
            [original_court_key_points[2], original_court_key_points[3]],   # Point 1
            [original_court_key_points[4], original_court_key_points[5]],   # Point 2
            [original_court_key_points[6], original_court_key_points[7]],   # Point 3
        ], dtype=np.float32)

        # Corresponding mini court points
        dst_points = np.array([
            [self.drawing_key_points[0], self.drawing_key_points[1]],   # Point 0
            [self.drawing_key_points[2], self.drawing_key_points[3]],   # Point 1
            [self.drawing_key_points[4], self.drawing_key_points[5]],   # Point 2
            [self.drawing_key_points[6], self.drawing_key_points[7]],   # Point 3
        ], dtype=np.float32)

        # Compute homography
        H, _ = cv2.findHomography(src_points, dst_points)
        return H

    def transform_point_homography(self, point, H):
        """Transform a single point using homography matrix"""
        pt = np.array([[point[0], point[1], 1]], dtype=np.float32).T
        transformed = H @ pt
        transformed = transformed / transformed[2]  # Normalize
        return (float(transformed[0]), float(transformed[1]))

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        # Player heights for all 4 padel players
        default_height = (constants.PLAYER_1_HEIGHT_METERS + constants.PLAYER_2_HEIGHT_METERS +
                          constants.PLAYER_3_HEIGHT_METERS + constants.PLAYER_4_HEIGHT_METERS) / 4
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS,
            3: constants.PLAYER_3_HEIGHT_METERS,
            4: constants.PLAYER_4_HEIGHT_METERS
        }

        output_player_boxes = []
        output_ball_boxes = []

        # Compute homography for ball position (more accurate)
        H = self.compute_homography(original_court_key_points)

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)

            # Skip frames with no players detected
            if not player_bbox:
                # Use previous frame's data if available
                if output_player_boxes:
                    output_player_boxes.append(output_player_boxes[-1].copy())
                else:
                    output_player_boxes.append({})
                # Ball position
                if H is not None:
                    ball_mini_court_pos = self.transform_point_homography(ball_position, H)
                    ball_x = max(self.court_start_x, min(self.court_end_x, ball_mini_court_pos[0]))
                    ball_y = max(self.court_start_y, min(self.court_end_y, ball_mini_court_pos[1]))
                    output_ball_boxes.append({1: (ball_x, ball_y)})
                else:
                    output_ball_boxes.append({1: ball_position})
                continue

            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position,
                                                                                               get_center_of_bbox(
                                                                                                   player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Use homography for accurate perspective transformation
                # This properly maps video coordinates to mini-court coordinates
                mini_court_pos = self.transform_point_homography(foot_position, H)

                # Clamp to mini court bounds
                player_x = max(self.court_start_x, min(self.court_end_x, mini_court_pos[0]))
                player_y = max(self.court_start_y, min(self.court_end_y, mini_court_pos[1]))

                output_player_bboxes_dict[player_id] = (player_x, player_y)

            # Use homography for ball position
            ball_mini_court_pos = self.transform_point_homography(ball_position, H)

            # Clamp to mini court bounds
            ball_x = max(self.court_start_x, min(self.court_end_x, ball_mini_court_pos[0]))
            ball_y = max(self.court_start_y, min(self.court_end_y, ball_mini_court_pos[1]))

            output_ball_boxes.append({1: (ball_x, ball_y)})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(self, frames, postions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, color, -1)
        return frames

    def generate_heatmap(self, positions_list, resolution=50):
        """Generate a heatmap from a list of (x, y) positions"""
        if not positions_list:
            return None

        # Create bins for the mini court area
        x_bins = np.linspace(self.court_start_x, self.court_end_x, resolution)
        y_bins = np.linspace(self.court_start_y, self.court_end_y, resolution)

        # Extract x and y coordinates
        x_coords = [p[0] for p in positions_list]
        y_coords = [p[1] for p in positions_list]

        # Create 2D histogram
        heatmap, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
        heatmap = heatmap.T  # Transpose for correct orientation

        # Apply Gaussian blur first for smoother distribution
        heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (7, 7), 0)

        # Normalize to 0-255 AFTER blur
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

        return heatmap

    def draw_heatmap_on_frame(self, frame, heatmap, colormap=None, alpha=0.7, use_custom=True):
        """Draw a heatmap overlay on the mini court area with fading effect"""
        if heatmap is None:
            return frame

        # Resize heatmap to match mini court dimensions
        court_width = int(self.court_end_x - self.court_start_x)
        court_height = int(self.court_end_y - self.court_start_y)
        heatmap_resized = cv2.resize(heatmap, (court_width, court_height))

        # Create custom red-yellow-orange heatmap using vectorized numpy operations
        heatmap_norm = heatmap_resized.astype(np.float32)
        heatmap_colored = np.zeros((court_height, court_width, 3), dtype=np.float32)

        # Mask for different intensity ranges (lowered thresholds)
        mask_low = (heatmap_norm >= 5) & (heatmap_norm < 80)
        mask_med = (heatmap_norm >= 80) & (heatmap_norm < 170)
        mask_high = heatmap_norm >= 170

        # Low intensity: black to yellow (BGR: [0, G, R])
        t_low = np.clip((heatmap_norm - 5) / 75.0, 0, 1)
        heatmap_colored[:, :, 1] = np.where(mask_low, 220 * t_low, heatmap_colored[:, :, 1])  # G
        heatmap_colored[:, :, 2] = np.where(mask_low, 255 * t_low, heatmap_colored[:, :, 2])  # R

        # Medium intensity: yellow to orange
        t_med = np.clip((heatmap_norm - 80) / 90.0, 0, 1)
        heatmap_colored[:, :, 1] = np.where(mask_med, 220 - 100 * t_med, heatmap_colored[:, :, 1])  # G
        heatmap_colored[:, :, 2] = np.where(mask_med, 255, heatmap_colored[:, :, 2])  # R

        # High intensity: orange to red
        t_high = np.clip((heatmap_norm - 170) / 85.0, 0, 1)
        heatmap_colored[:, :, 1] = np.where(mask_high, 120 - 120 * t_high, heatmap_colored[:, :, 1])  # G
        heatmap_colored[:, :, 2] = np.where(mask_high, 255, heatmap_colored[:, :, 2])  # R

        heatmap_colored = heatmap_colored.astype(np.uint8)

        # Create alpha mask based on intensity (fading effect)
        alpha_mask = (heatmap_resized.astype(np.float32) / 255.0) * alpha
        alpha_mask = np.clip(alpha_mask, 0, alpha)

        # Hide only near-zero values
        alpha_mask[heatmap_resized < 3] = 0

        # Get the region of interest
        roi = frame[int(self.court_start_y):int(self.court_end_y),
                    int(self.court_start_x):int(self.court_end_x)]

        # Blend heatmap with frame using per-pixel alpha (vectorized)
        alpha_3d = alpha_mask[:, :, np.newaxis]
        roi[:] = (roi * (1 - alpha_3d) + heatmap_colored * alpha_3d).astype(np.uint8)

        return frame

    def create_player_heatmaps(self, player_positions):
        """Create heatmaps for all 4 padel players from their positions across all frames"""
        player_positions_dict = {1: [], 2: [], 3: [], 4: []}

        for frame_positions in player_positions:
            for pid in [1, 2, 3, 4]:
                if pid in frame_positions:
                    player_positions_dict[pid].append(frame_positions[pid])

        heatmaps = {}
        for pid in [1, 2, 3, 4]:
            heatmaps[pid] = self.generate_heatmap(player_positions_dict[pid])

        return heatmaps

    def create_shot_heatmap(self, ball_positions, shot_frames):
        """Create heatmap of shot locations"""
        shot_positions = []
        for frame_idx in shot_frames:
            if frame_idx < len(ball_positions) and 1 in ball_positions[frame_idx]:
                shot_positions.append(ball_positions[frame_idx][1])

        return self.generate_heatmap(shot_positions, resolution=30)

    def draw_heatmap_legend(self, frame, x, y, width=150, height=20, label="Activity"):
        """Draw a color legend for the heatmap"""
        # Create gradient
        gradient = np.linspace(0, 255, width).astype(np.uint8)
        gradient = np.tile(gradient, (height, 1))
        gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

        # Place gradient on frame
        frame[y:y+height, x:x+width] = gradient_colored

        # Add border
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 255), 1)

        # Add labels
        cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Low", (x, y+height+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(frame, "High", (x+width-25, y+height+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return frame

    def draw_mini_court_with_heatmap(self, frame, heatmap, title="Heatmap"):
        """Draw mini court with heatmap overlay"""
        # Draw background
        frame = self.draw_background_rectangle(frame)

        # Draw heatmap
        frame = self.draw_heatmap_on_frame(frame, heatmap, alpha=0.5)

        # Draw court lines on top
        frame = self.draw_court(frame)

        # Add title
        cv2.putText(frame, title, (self.start_x + 10, self.start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def draw_mini_court_with_live_heatmap(self, frames, player_positions, ball_positions, shot_frames,
                                          shot_player_mapping=None, alternate_interval=72):
        """
        Draw 3 stacked mini courts on the right side for padel doubles (4 players):
        1. Bird eye view (current positions)
        2. Team movement heatmap (Team 1 vs Team 2)
        3. Shot placement heatmap
        """
        output_frames = []
        if shot_player_mapping is None:
            shot_player_mapping = {}

        # Accumulated positions with timestamps for time-based decay heatmaps
        # Format: list of (position, frame_idx) tuples
        player_accumulated = {1: [], 2: [], 3: [], 4: []}
        # Shots accumulated by team
        team1_shots_accumulated = []  # Team 1: Players 1, 2
        team2_shots_accumulated = []  # Team 2: Players 3, 4

        # Heatmap decay settings
        decay_window = 360  # Frames to keep (15 seconds at 24fps)
        decay_rate = 0.995  # Exponential decay per frame (older = fainter)

        # Calculate dimensions for 3 stacked courts
        total_height = 900  # Much taller
        court_spacing = 40  # Good space between courts
        single_court_height = (total_height - 2 * court_spacing) // 3
        court_width = 220  # Wider courts
        buffer = 30

        # Identity function - no flip needed, homography maps correctly
        def flip_y(pos):
            return pos

        # Team colors for padel doubles
        # Team 1 (bottom): P1, P2 - green shades
        # Team 2 (top): P3, P4 - red shades
        player_colors = {
            1: (0, 200, 0),      # P1 - Green
            2: (0, 180, 120),    # P2 - Teal
            3: (0, 0, 200),      # P3 - Red
            4: (0, 100, 200),    # P4 - Orange
        }

        for frame_idx, frame in enumerate(frames):
            # Accumulate player positions with timestamp for decay
            if frame_idx < len(player_positions):
                for pid in [1, 2, 3, 4]:
                    if pid in player_positions[frame_idx]:
                        pos = flip_y(player_positions[frame_idx][pid])
                        player_accumulated[pid].append((pos, frame_idx))
                        # Remove old positions outside decay window
                        while player_accumulated[pid] and frame_idx - player_accumulated[pid][0][1] > decay_window:
                            player_accumulated[pid].pop(0)

            # Accumulate shot positions by team - use PLAYER position when they hit the shot
            if frame_idx in shot_frames and frame_idx < len(player_positions):
                shooter_id = shot_player_mapping.get(frame_idx, None)
                if shooter_id and shooter_id in player_positions[frame_idx]:
                    # Use player's position when they hit the shot
                    shot_pos = flip_y(player_positions[frame_idx][shooter_id])
                    if shooter_id in [1, 2]:  # Team 1
                        team1_shots_accumulated.append(shot_pos)
                    elif shooter_id in [3, 4]:  # Team 2
                        team2_shots_accumulated.append(shot_pos)

            # Calculate court positions
            right_x = frame.shape[1] - court_width - buffer
            court_1_y = buffer
            court_2_y = court_1_y + single_court_height + court_spacing
            court_3_y = court_2_y + single_court_height + court_spacing

            # Scale factors for mapping positions to smaller courts
            scale_x = court_width / self.drawing_rectangle_width
            scale_y = single_court_height / self.drawing_rectangle_height

            def draw_single_court_background(frame, start_x, start_y, width, height, title, title_color):
                """Draw court background and title only"""
                end_x = start_x + width
                end_y = start_y + height

                # Background
                overlay = frame.copy()
                cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (255, 255, 255), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                # Title background
                cv2.rectangle(frame, (start_x, start_y - 24), (end_x, start_y - 2), (40, 40, 40), -1)
                cv2.rectangle(frame, (start_x, start_y - 24), (end_x, start_y - 2), title_color, 2)
                # Title text
                text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                text_x = start_x + (width - text_size[0]) // 2
                cv2.putText(frame, title, (text_x, start_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, title_color, 1, cv2.LINE_AA)

                court_padding = 10
                court_start_x = start_x + court_padding
                court_start_y = start_y + court_padding
                court_end_x = end_x - court_padding
                court_end_y = end_y - court_padding

                return frame, court_start_x, court_start_y, court_end_x - court_start_x, court_end_y - court_start_y

            def draw_court_lines(frame, court_x, court_y, court_w, court_h):
                """Draw court lines on top of heatmap"""
                court_end_x = court_x + court_w
                court_end_y = court_y + court_h
                court_mid_y = (court_y + court_end_y) // 2

                # Outer boundary
                cv2.rectangle(frame, (court_x, court_y), (court_end_x, court_end_y), (0, 0, 0), 2)
                # Net line
                cv2.line(frame, (court_x, court_mid_y), (court_end_x, court_mid_y), (180, 0, 0), 2)
                # Service lines
                service_offset = court_h // 4
                cv2.line(frame, (court_x, court_y + service_offset), (court_end_x, court_y + service_offset), (0, 0, 0), 1)
                cv2.line(frame, (court_x, court_end_y - service_offset), (court_end_x, court_end_y - service_offset), (0, 0, 0), 1)
                # Center service line
                cv2.line(frame, ((court_x + court_end_x) // 2, court_y + service_offset),
                        ((court_x + court_end_x) // 2, court_end_y - service_offset), (0, 0, 0), 1)
                return frame

            def map_position(pos, court_x, court_y, court_w, court_h):
                """Map original mini court position to scaled court"""
                rel_x = (pos[0] - self.court_start_x) / (self.court_end_x - self.court_start_x)
                rel_y = (pos[1] - self.court_start_y) / (self.court_end_y - self.court_start_y)
                return (int(court_x + rel_x * court_w), int(court_y + rel_y * court_h))

            def generate_scaled_heatmap(positions, court_x, court_y, court_w, court_h, resolution=25):
                """Generate heatmap for scaled court"""
                if not positions:
                    return None
                mapped = [map_position(p, court_x, court_y, court_w, court_h) for p in positions]
                x_coords = [p[0] for p in mapped]
                y_coords = [p[1] for p in mapped]
                x_bins = np.linspace(court_x, court_x + court_w, resolution)
                y_bins = np.linspace(court_y, court_y + court_h, resolution)
                heatmap, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
                heatmap = heatmap.T
                heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (5, 5), 0)
                if heatmap.max() > 0:
                    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
                return heatmap, court_x, court_y, court_w, court_h

            def draw_heatmap_scaled(frame, heatmap_data, alpha=0.8, color_mode='hot'):
                """Draw heatmap on scaled court area with vivid colors"""
                if heatmap_data is None or heatmap_data[0] is None:
                    return frame
                heatmap, cx, cy, cw, ch = heatmap_data
                heatmap_resized = cv2.resize(heatmap, (cw, ch))

                # Vivid color mapping (yellow -> orange -> red)
                heatmap_norm = heatmap_resized.astype(np.float32)
                heatmap_colored = np.zeros((ch, cw, 3), dtype=np.float32)

                # Low intensity: yellow
                mask_low = (heatmap_norm >= 1) & (heatmap_norm < 100)
                t_low = np.clip((heatmap_norm - 1) / 99.0, 0, 1)
                heatmap_colored[:, :, 2] = np.where(mask_low, 255, heatmap_colored[:, :, 2])  # R
                heatmap_colored[:, :, 1] = np.where(mask_low, 255 * (1 - t_low * 0.3), heatmap_colored[:, :, 1])  # G

                # Medium intensity: orange
                mask_med = (heatmap_norm >= 100) & (heatmap_norm < 180)
                t_med = np.clip((heatmap_norm - 100) / 80.0, 0, 1)
                heatmap_colored[:, :, 2] = np.where(mask_med, 255, heatmap_colored[:, :, 2])  # R
                heatmap_colored[:, :, 1] = np.where(mask_med, 180 - 100 * t_med, heatmap_colored[:, :, 1])  # G

                # High intensity: red
                mask_high = heatmap_norm >= 180
                heatmap_colored[:, :, 2] = np.where(mask_high, 255, heatmap_colored[:, :, 2])  # R
                heatmap_colored[:, :, 1] = np.where(mask_high, 80 * (1 - (heatmap_norm - 180) / 75.0), heatmap_colored[:, :, 1])  # G

                heatmap_colored = np.clip(heatmap_colored, 0, 255).astype(np.uint8)

                # Alpha based on intensity
                alpha_mask = np.clip((heatmap_resized.astype(np.float32) / 255.0) * alpha + 0.1, 0, alpha)
                alpha_mask[heatmap_resized < 1] = 0

                roi = frame[cy:cy+ch, cx:cx+cw]
                alpha_3d = alpha_mask[:, :, np.newaxis]
                roi[:] = (roi * (1 - alpha_3d) + heatmap_colored * alpha_3d).astype(np.uint8)
                return frame

            def draw_team_heatmap(frame, team1_positions, team2_positions, cx, cy, cw, ch, current_frame, alpha=0.7):
                """Draw team heatmaps with time-based decay (Team 1 = green, Team 2 = red)"""
                # Combine positions by team - format is (pos, frame_idx)
                all_team1 = []
                all_team2 = []
                for pid in [1, 2]:
                    all_team1.extend(team1_positions.get(pid, []))
                for pid in [3, 4]:
                    all_team2.extend(team2_positions.get(pid, []))

                if not all_team1 and not all_team2:
                    return frame

                # Create empty heatmap canvases
                team1_heatmap = np.zeros((ch, cw), dtype=np.float32)
                team2_heatmap = np.zeros((ch, cw), dtype=np.float32)

                # Heat radius - larger means more spread
                heat_radius = max(12, min(cw, ch) // 10)

                # Paint heat circles for Team 1 positions with decay
                for pos_data in all_team1:
                    pos, pos_frame = pos_data
                    # Calculate decay weight based on age
                    age = current_frame - pos_frame
                    weight = decay_rate ** age  # Exponential decay
                    mapped = map_position(pos, cx, cy, cw, ch)
                    px, py = int(mapped[0] - cx), int(mapped[1] - cy)
                    cv2.circle(team1_heatmap, (px, py), heat_radius, weight, -1)

                # Paint heat circles for Team 2 positions with decay
                for pos_data in all_team2:
                    pos, pos_frame = pos_data
                    age = current_frame - pos_frame
                    weight = decay_rate ** age
                    mapped = map_position(pos, cx, cy, cw, ch)
                    px, py = int(mapped[0] - cx), int(mapped[1] - cy)
                    cv2.circle(team2_heatmap, (px, py), heat_radius, weight, -1)

                # Apply heavy Gaussian blur for smooth heat effect
                blur_kernel = (25, 25)
                team1_heatmap = cv2.GaussianBlur(team1_heatmap, blur_kernel, 0)
                team2_heatmap = cv2.GaussianBlur(team2_heatmap, blur_kernel, 0)

                # Normalize independently (each team's max = 1.0)
                if team1_heatmap.max() > 0:
                    team1_heatmap = team1_heatmap / team1_heatmap.max()
                if team2_heatmap.max() > 0:
                    team2_heatmap = team2_heatmap / team2_heatmap.max()

                # Apply intensity curve for more dramatic effect
                team1_heatmap = np.power(team1_heatmap, 0.5)
                team2_heatmap = np.power(team2_heatmap, 0.5)

                # Create vivid colored heatmaps
                # Team 1: Cyan to Green gradient
                team1_colored = np.zeros((ch, cw, 3), dtype=np.float32)
                team1_colored[:, :, 1] = team1_heatmap * 255  # Green (strong)
                team1_colored[:, :, 0] = team1_heatmap * 200  # Cyan tint
                team1_colored[:, :, 2] = team1_heatmap * 50   # Slight warmth

                # Team 2: Orange to Red gradient
                team2_colored = np.zeros((ch, cw, 3), dtype=np.float32)
                team2_colored[:, :, 2] = team2_heatmap * 255  # Red (strong)
                team2_colored[:, :, 1] = team2_heatmap * 100  # Orange tint
                team2_colored[:, :, 0] = team2_heatmap * 50   # Slight purple

                # Combine heatmaps
                combined_colored = np.clip(team1_colored + team2_colored, 0, 255).astype(np.uint8)

                # Strong alpha for vivid effect
                combined_intensity = np.maximum(team1_heatmap, team2_heatmap)
                alpha_mask = np.clip(combined_intensity * 0.85 + 0.1, 0, 0.9)
                alpha_mask[combined_intensity < 0.01] = 0

                # Apply to frame
                roi = frame[cy:cy+ch, cx:cx+cw]
                alpha_3d = alpha_mask[:, :, np.newaxis]
                roi[:] = (roi * (1 - alpha_3d) + combined_colored * alpha_3d).astype(np.uint8)

                return frame

            # ===== COURT 1: Bird Eye View (current positions) =====
            frame, c1_x, c1_y, c1_w, c1_h = draw_single_court_background(
                frame, right_x, court_1_y, court_width, single_court_height,
                "LIVE POSITIONS", (0, 255, 255))
            frame = draw_court_lines(frame, c1_x, c1_y, c1_w, c1_h)

            # Draw current player positions for all 4 players (with Y flipped)
            if frame_idx < len(player_positions):
                for player_id, position in player_positions[frame_idx].items():
                    flipped_pos = flip_y(position)
                    mapped = map_position(flipped_pos, c1_x, c1_y, c1_w, c1_h)
                    # Team colors
                    color = player_colors.get(player_id, (128, 128, 128))

                    # Professional player marker with glow effect
                    # Outer glow
                    glow_overlay = frame.copy()
                    cv2.circle(glow_overlay, mapped, 12, color, -1)
                    cv2.addWeighted(glow_overlay, 0.3, frame, 0.7, 0, frame)

                    # Main circle with gradient effect (darker inner)
                    cv2.circle(frame, mapped, 8, color, -1, cv2.LINE_AA)
                    # Darker inner core
                    dark_color = (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60))
                    cv2.circle(frame, mapped, 5, dark_color, -1, cv2.LINE_AA)
                    # White border
                    cv2.circle(frame, mapped, 8, (255, 255, 255), 2, cv2.LINE_AA)

                    # Player number in circle
                    cv2.putText(frame, str(player_id), (mapped[0] - 4, mapped[1] + 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw ball (with Y flipped) - enhanced visibility
            if frame_idx < len(ball_positions):
                for _, position in ball_positions[frame_idx].items():
                    flipped_pos = flip_y(position)
                    mapped = map_position(flipped_pos, c1_x, c1_y, c1_w, c1_h)

                    # Pulsing glow effect for ball
                    pulse = 0.6 + 0.4 * np.sin(frame_idx * 0.3)  # Pulsing alpha

                    # Outer glow (yellow-green)
                    glow_overlay = frame.copy()
                    cv2.circle(glow_overlay, mapped, 14, (50, 255, 200), -1)
                    cv2.addWeighted(glow_overlay, 0.25 * pulse, frame, 1 - 0.25 * pulse, 0, frame)

                    # Middle glow
                    cv2.circle(frame, mapped, 9, (100, 255, 220), -1, cv2.LINE_AA)
                    # Inner bright core (tennis ball green-yellow)
                    cv2.circle(frame, mapped, 6, (0, 255, 255), -1, cv2.LINE_AA)
                    # White highlight
                    cv2.circle(frame, mapped, 3, (200, 255, 255), -1, cv2.LINE_AA)
                    # Sharp outer ring
                    cv2.circle(frame, mapped, 9, (0, 200, 200), 2, cv2.LINE_AA)

            # ===== COURT 2: Team Movement Heatmap =====
            frame, c2_x, c2_y, c2_w, c2_h = draw_single_court_background(
                frame, right_x, court_2_y, court_width, single_court_height,
                "TEAM MOVEMENT", (0, 165, 255))

            # Draw team heatmap (green=Team 1, red=Team 2) BEFORE court lines
            has_positions = any(len(player_accumulated[pid]) > 0 for pid in [1, 2, 3, 4])
            if has_positions:
                frame = draw_team_heatmap(frame, player_accumulated, player_accumulated,
                                          c2_x, c2_y, c2_w, c2_h, frame_idx, alpha=0.85)

            # Draw court lines on top of heatmap
            frame = draw_court_lines(frame, c2_x, c2_y, c2_w, c2_h)

            # ===== COURT 3: Shot Placement by Team =====
            frame, c3_x, c3_y, c3_w, c3_h = draw_single_court_background(
                frame, right_x, court_3_y, court_width, single_court_height,
                "SHOT PLACEMENT", (255, 100, 100))

            # Draw court lines first
            frame = draw_court_lines(frame, c3_x, c3_y, c3_w, c3_h)

            # Draw shot markers for Team 1 (green)
            for shot_pos in team1_shots_accumulated:
                mapped = map_position(shot_pos, c3_x, c3_y, c3_w, c3_h)
                cv2.circle(frame, mapped, 10, (0, 100, 0), -1)   # Outer glow
                cv2.circle(frame, mapped, 6, (0, 200, 0), -1)    # Inner fill
                cv2.circle(frame, mapped, 7, (255, 255, 255), 1) # White border

            # Draw shot markers for Team 2 (red)
            for shot_pos in team2_shots_accumulated:
                mapped = map_position(shot_pos, c3_x, c3_y, c3_w, c3_h)
                cv2.circle(frame, mapped, 10, (0, 0, 100), -1)   # Outer glow
                cv2.circle(frame, mapped, 6, (0, 0, 200), -1)    # Inner fill
                cv2.circle(frame, mapped, 7, (255, 255, 255), 1) # White border

            output_frames.append(frame)

        return output_frames
