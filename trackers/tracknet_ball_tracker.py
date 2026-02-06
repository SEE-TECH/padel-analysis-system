"""
TrackNet-based Ball Tracker for Padel

Uses TrackNet (U-Net encoder-decoder) for more accurate ball detection.
Matches padel_analytics implementation exactly.
"""

import cv2
import pickle
import pandas as pd
import numpy as np
import os
import torch
from collections import deque

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tracknet import TrackNet


def predict_location(heatmap, img_scaler=(1.0, 1.0), threshold=0.5):
    """
    Extract ball coordinates from heatmap using contour detection.
    Matches padel_analytics predict_modified function.

    Args:
        heatmap: (H, W) numpy array with ball probability
        img_scaler: (scale_x, scale_y) to convert to original frame size
        threshold: binarization threshold (default 0.5)

    Returns:
        (x, y, visibility) or (0, 0, 0) if no detection
    """
    # Binarize heatmap
    binary = (heatmap > threshold).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0, 0, 0

    # Find largest contour by area
    max_area = 0
    best_bbox = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            best_bbox = (x, y, w, h)

    if best_bbox is None:
        return 0, 0, 0

    # Get center of bounding box
    x, y, w, h = best_bbox
    cx = int((x + w / 2) * img_scaler[0])
    cy = int((y + h / 2) * img_scaler[1])

    return cx, cy, 1


class TrackNetBallTracker:
    """
    Ball tracker using TrackNet model.
    Matches padel_analytics implementation.

    Uses 8 frames + 1 background frame (bg_mode=concat) = 27 input channels.
    """

    def __init__(self, model_path, device=None, input_size=(288, 512)):
        """
        Args:
            model_path: Path to TrackNet weights (.pt file)
            device: 'cuda' or 'cpu' (auto-detect if None)
            input_size: (height, width) for model input
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading TrackNet on {self.device}...")
        self.model = self._load_model(model_path)
        self.input_size = input_size  # (H, W)
        self.seq_len = 8  # Number of frames in sequence

        # Frame buffer for 8-frame input
        self.frame_buffer = deque(maxlen=self.seq_len)

        # Background/median frame (will be computed from first N frames)
        self.bg_frame = None
        self.bg_frames_for_median = []
        self.bg_frames_needed = 30  # Frames to compute median

        # Detection threshold
        self.threshold = 0.5

    def _load_model(self, model_path):
        """Load TrackNet model with weights"""
        model = TrackNet(in_dim=27, out_dim=8)  # 9 frames × 3 RGB = 27

        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        print("TrackNet model loaded successfully!")
        return model

    def _preprocess_frame(self, frame):
        """Preprocess frame for TrackNet input"""
        h, w = self.input_size
        resized = cv2.resize(frame, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return normalized

    def _compute_background(self, frames):
        """Compute median background from frames"""
        if len(frames) == 0:
            return None
        stacked = np.stack(frames, axis=0)
        median = np.median(stacked, axis=0)
        return median.astype(np.float32)

    def detect_frame(self, frame):
        """
        Detect ball in a single frame.
        """
        # Preprocess frame
        processed = self._preprocess_frame(frame)

        # Collect frames for background median
        if len(self.bg_frames_for_median) < self.bg_frames_needed:
            self.bg_frames_for_median.append(processed.copy())
            if len(self.bg_frames_for_median) == self.bg_frames_needed:
                self.bg_frame = self._compute_background(self.bg_frames_for_median)
                print(f"Background frame computed from {self.bg_frames_needed} frames")

        # Add to frame buffer
        self.frame_buffer.append(processed)

        # Need background and full buffer for detection
        if self.bg_frame is None or len(self.frame_buffer) < self.seq_len:
            return {}

        # Stack: background frame + 8 sequence frames = 9 frames
        # bg_mode='concat' prepends background
        frames_list = [self.bg_frame] + list(self.frame_buffer)

        # Concatenate along channel axis: (9, H, W, 3) -> (H, W, 27) -> (27, H, W)
        stacked = np.concatenate(frames_list, axis=2)  # (H, W, 27)
        stacked = np.transpose(stacked, (2, 0, 1))  # (27, H, W)

        # Convert to tensor
        input_tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Get heatmap - use last channel (index 7) for current frame prediction
        # The 8 output channels correspond to the 8 input frames
        heatmap = output[0, -1].cpu().numpy()  # Use last channel for current frame

        # Extract ball position using contour method
        img_scaler = (frame.shape[1] / self.input_size[1],
                      frame.shape[0] / self.input_size[0])
        cx, cy, visibility = predict_location(heatmap, img_scaler, self.threshold)

        if visibility == 0:
            return {}

        # Return as bbox
        ball_radius = 10
        return {1: [cx - ball_radius, cy - ball_radius,
                    cx + ball_radius, cy + ball_radius]}

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """Detect ball in all frames."""
        ball_detections = []

        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'rb') as f:
                    ball_detections = pickle.load(f)
                print(f"Loaded ball detections from {stub_path}")
                return ball_detections
            except FileNotFoundError:
                print(f"Stub not found, running TrackNet detection...")
            except Exception as e:
                print(f"Error loading stub: {e}")

        # Reset state
        self.frame_buffer.clear()
        self.bg_frame = None
        self.bg_frames_for_median = []

        print(f"Running TrackNet ball detection on {len(frames)} frames...")

        for i, frame in enumerate(frames):
            if i % 100 == 0:
                print(f"  Processing frame {i}/{len(frames)}")
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            try:
                with open(stub_path, 'wb') as f:
                    pickle.dump(ball_detections, f)
                print(f"Saved ball detections to {stub_path}")
            except Exception as e:
                print(f"Error saving stub: {e}")

        return ball_detections

    def interpolate_ball_positions(self, ball_positions):
        """Interpolate missing ball positions with outlier filtering"""
        # First pass: extract positions and filter outliers (sudden jumps)
        ball_positions_filtered = self._filter_ball_outliers(ball_positions)

        # Convert to DataFrame for interpolation
        positions = [x.get(1, []) for x in ball_positions_filtered]
        df = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.bfill()
        df = df.ffill()

        ball_positions = [{1: x} for x in df.to_numpy().tolist()]
        return ball_positions

    def _filter_ball_outliers(self, ball_positions):
        """Filter out sudden jumps in ball position (false positives)"""
        max_distance_per_frame = 120  # Maximum pixels ball can move per frame
        max_absolute_distance = 400   # Hard cap on distance regardless of missing frames
        min_consecutive_frames = 3    # Need 3+ consistent frames to trust position

        # First pass: basic distance filtering
        filtered = []
        last_valid_pos = None
        last_valid_velocity = (0, 0)
        consecutive_count = 0

        for i, bp in enumerate(ball_positions):
            if 1 not in bp or not bp[1]:
                filtered.append({})
                consecutive_count = 0
                continue

            bbox = bp[1]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if last_valid_pos is None:
                # First detection
                last_valid_pos = (cx, cy)
                filtered.append(bp)
                consecutive_count = 1
                continue

            # Calculate distance from last valid position
            dx = cx - last_valid_pos[0]
            dy = cy - last_valid_pos[1]
            distance = (dx**2 + dy**2)**0.5

            # Predict expected position based on velocity
            predicted_x = last_valid_pos[0] + last_valid_velocity[0]
            predicted_y = last_valid_pos[1] + last_valid_velocity[1]
            pred_distance = ((cx - predicted_x)**2 + (cy - predicted_y)**2)**0.5

            # Check if this position is valid
            # Allow slightly larger jumps if ball was missing for a while, but cap it
            frames_since_valid = 1
            for j in range(i-1, max(0, i-10), -1):
                if 1 in ball_positions[j] and ball_positions[j][1]:
                    break
                frames_since_valid += 1

            # Cap the adjustment to prevent allowing huge jumps
            adjusted_max_dist = min(max_distance_per_frame * frames_since_valid, max_absolute_distance)

            # Position is valid if:
            # 1. Distance from last position is reasonable, OR
            # 2. Distance from predicted position is reasonable
            is_valid = (distance < adjusted_max_dist) or (pred_distance < adjusted_max_dist)

            if is_valid:
                # Update velocity estimate
                if consecutive_count >= min_consecutive_frames - 1:
                    last_valid_velocity = (dx * 0.7 + last_valid_velocity[0] * 0.3,
                                           dy * 0.7 + last_valid_velocity[1] * 0.3)

                last_valid_pos = (cx, cy)
                filtered.append(bp)
                consecutive_count += 1
            else:
                # Outlier detected - mark as missing
                filtered.append({})
                consecutive_count = 0

        # Second pass: remove isolated detections (single frame spikes)
        # A valid detection should have at least one neighbor within reasonable distance
        second_pass = []
        neighbor_max_dist = 150

        for i, bp in enumerate(filtered):
            if 1 not in bp or not bp[1]:
                second_pass.append({})
                continue

            bbox = bp[1]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # Check neighbors in a small window
            has_valid_neighbor = False
            for j in range(max(0, i-3), min(len(filtered), i+4)):
                if j == i:
                    continue
                if 1 in filtered[j] and filtered[j][1]:
                    neighbor_bbox = filtered[j][1]
                    ncx = (neighbor_bbox[0] + neighbor_bbox[2]) / 2
                    ncy = (neighbor_bbox[1] + neighbor_bbox[3]) / 2
                    dist = ((cx - ncx)**2 + (cy - ncy)**2)**0.5
                    # Distance should scale with frame gap
                    frame_gap = abs(j - i)
                    if dist < neighbor_max_dist * frame_gap:
                        has_valid_neighbor = True
                        break

            if has_valid_neighbor:
                second_pass.append(bp)
            else:
                second_pass.append({})

        # Third pass: check trajectory consistency after gaps
        # If a detection appears after many missing frames, check if it fits
        # between earlier and later detections
        final_filtered = []
        gap_threshold = 10  # Consider this a significant gap
        cluster_skip_dist = 100  # Skip nearby positions to avoid cluster self-validation

        for i, bp in enumerate(second_pass):
            if 1 not in bp or not bp[1]:
                final_filtered.append({})
                continue

            bbox = bp[1]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # Find previous valid detection (far enough to be independent)
            prev_pos = None
            prev_gap = 0
            for j in range(i-1, max(0, i-50), -1):
                if 1 in second_pass[j] and second_pass[j][1]:
                    prev_bbox = second_pass[j][1]
                    px = (prev_bbox[0] + prev_bbox[2]) / 2
                    py = (prev_bbox[1] + prev_bbox[3]) / 2
                    # Skip if too close (might be part of same outlier cluster)
                    dist_to_curr = ((px - cx)**2 + (py - cy)**2)**0.5
                    if dist_to_curr > cluster_skip_dist:
                        prev_pos = (px, py)
                        prev_gap = i - j
                        break

            # Find next valid detection (far enough to be independent)
            next_pos = None
            next_gap = 0
            for j in range(i+1, min(len(second_pass), i+50)):
                if 1 in second_pass[j] and second_pass[j][1]:
                    next_bbox = second_pass[j][1]
                    nx = (next_bbox[0] + next_bbox[2]) / 2
                    ny = (next_bbox[1] + next_bbox[3]) / 2
                    # Skip if too close (might be part of same outlier cluster)
                    dist_to_curr = ((nx - cx)**2 + (ny - cy)**2)**0.5
                    if dist_to_curr > cluster_skip_dist:
                        next_pos = (nx, ny)
                        next_gap = j - i
                        break

            # If there's a significant gap on either side, check consistency
            if prev_pos and next_pos and (prev_gap >= gap_threshold or next_gap >= gap_threshold):
                # Calculate expected position based on interpolation
                total_gap = prev_gap + next_gap
                t = prev_gap / total_gap
                expected_x = prev_pos[0] + t * (next_pos[0] - prev_pos[0])
                expected_y = prev_pos[1] + t * (next_pos[1] - prev_pos[1])

                # Distance from expected interpolated position
                interp_dist = ((cx - expected_x)**2 + (cy - expected_y)**2)**0.5

                # If current position is far from interpolated path, reject it
                max_deviation = 150  # Maximum allowed deviation from interpolated path
                if interp_dist > max_deviation:
                    final_filtered.append({})
                    continue

            final_filtered.append(bp)

        return final_filtered

    def get_ball_shot_frames(self, ball_positions, player_detections=None):
        """Detect frames where a player hits the ball.

        Strategy: Find trajectory turning points where ball Y direction reverses,
        then find the closest player who could have made that hit.

        Args:
            ball_positions: List of ball detection dicts
            player_detections: Optional list of player detection dicts for validation
        """
        # Extract ball center positions
        positions = []
        for i, bp in enumerate(ball_positions):
            if 1 in bp and bp[1]:
                bbox = bp[1]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                positions.append((i, cx, cy))
            else:
                positions.append((i, None, None))

        df = pd.DataFrame(positions, columns=['frame', 'x', 'y'])

        # Interpolate to fill gaps (for trajectory analysis only)
        df['y_interp'] = df['y'].interpolate(method='linear', limit_direction='both')
        df['y_smooth'] = df['y_interp'].rolling(window=7, min_periods=1, center=True).mean()

        # Find turning points (local min/max in Y)
        turning_points = []
        window = 8
        min_y_change = 40  # Minimum Y change to count as real direction change

        for i in range(window, len(df) - window):
            y_curr = df['y_smooth'].iloc[i]
            y_before = df['y_smooth'].iloc[i-window:i].values
            y_after = df['y_smooth'].iloc[i+1:i+window+1].values

            if pd.isna(y_curr):
                continue

            is_local_min = np.all(y_curr <= y_before) and np.all(y_curr <= y_after)
            is_local_max = np.all(y_curr >= y_before) and np.all(y_curr >= y_after)

            if is_local_min or is_local_max:
                y_change = abs(np.nanmean(y_before) - y_curr) + abs(np.nanmean(y_after) - y_curr)
                if y_change > min_y_change:
                    turning_points.append((i, 'max' if is_local_max else 'min'))

        if not player_detections:
            return [tp[0] for tp in turning_points]

        # For each turning point, find who hit the ball
        # Turning point is where ball ARRIVES, so shooter hit BEFORE this
        shot_frames = []
        min_gap = 35  # Minimum ~1.5 seconds between detected shots

        for turn_frame, turn_type in turning_points:
            # Search backward to find the shooter
            # Turning point = where ball changes direction = where player hit it
            # Max Y (bottom of screen) = ball at bottom = hit by bottom player (P1, P2)
            # Min Y (top of screen) = ball at top = hit by top player (P3, P4)

            if turn_type == 'max':
                # Ball at bottom - was hit by bottom player (P1 or P2)
                expected_players = [1, 2]
            else:
                # Ball at top - was hit by top player (P3 or P4)
                expected_players = [3, 4]

            # Search around the turning point for when ball was near expected player
            best_frame = None
            best_dist = float('inf')

            # Search window around turning point (player hits ball near turning point)
            for search_frame in range(max(0, turn_frame - 10), min(len(ball_positions), turn_frame + 5)):
                if search_frame >= len(player_detections):
                    continue

                ball_pos = ball_positions[search_frame].get(1)
                if not ball_pos:
                    continue

                ball_cx = (ball_pos[0] + ball_pos[2]) / 2
                ball_cy = (ball_pos[1] + ball_pos[3]) / 2

                for player_id in expected_players:
                    if player_id not in player_detections[search_frame]:
                        continue
                    bbox = player_detections[search_frame][player_id]
                    player_cx = (bbox[0] + bbox[2]) / 2
                    player_cy = (bbox[1] + bbox[3]) / 2
                    dist = ((ball_cx - player_cx)**2 + (ball_cy - player_cy)**2)**0.5

                    if dist < 250 and dist < best_dist:
                        best_dist = dist
                        best_frame = search_frame

            if best_frame is not None:
                # Check gap from last shot
                if not shot_frames or (best_frame - shot_frames[-1]) >= min_gap:
                    shot_frames.append(best_frame)

        return sorted(shot_frames)

    def draw_bboxes(self, video_frames, ball_detections, ball_shot_frames=None):
        """Draw yellow glowing ball tracking visualization"""
        output_video_frames = []
        trajectory_points = []
        trajectory_length = 12
        frames_without_ball = 0

        for frame_idx, (frame, ball_dict) in enumerate(zip(video_frames, ball_detections)):
            if not ball_dict:
                frames_without_ball += 1
                if frames_without_ball > 5:
                    trajectory_points.clear()
                output_video_frames.append(frame)
                continue

            frames_without_ball = 0

            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Check for position jumps
                if trajectory_points:
                    last_x, last_y = trajectory_points[-1]
                    distance = ((center_x - last_x)**2 + (center_y - last_y)**2)**0.5
                    if distance > 200:
                        trajectory_points.clear()

                trajectory_points.append((center_x, center_y))
                if len(trajectory_points) > trajectory_length:
                    trajectory_points.pop(0)

                # Gradient trail (cyan to yellow)
                if len(trajectory_points) >= 2:
                    for i in range(1, len(trajectory_points)):
                        alpha = i / len(trajectory_points)
                        r = int(255 * alpha)
                        g = int(255 * (0.8 + 0.2 * alpha))
                        b = int(200 + 55 * alpha)
                        color = (b, g, r)
                        thickness = max(1, int(4 * alpha))
                        cv2.line(frame, trajectory_points[i-1], trajectory_points[i],
                                color, thickness, cv2.LINE_AA)

                # Glowing ball effect - multiple layers
                for glow_radius, glow_alpha in [(18, 0.15), (14, 0.25), (10, 0.4)]:
                    glow_overlay = frame.copy()
                    cv2.circle(glow_overlay, (center_x, center_y), glow_radius,
                              (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.addWeighted(glow_overlay, glow_alpha, frame, 1 - glow_alpha, 0, frame)

                # Core ball - bright yellow
                cv2.circle(frame, (center_x, center_y), 7, (0, 255, 255), -1, cv2.LINE_AA)

                # White highlight
                cv2.circle(frame, (center_x - 2, center_y - 2), 3, (255, 255, 255), -1, cv2.LINE_AA)

            # Draw shot indicator - subtle professional style
            if ball_shot_frames and frame_idx in ball_shot_frames:
                if ball_dict:
                    bbox = list(ball_dict.values())[0]
                    cx, cy = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)

                    # Soft outer glow
                    glow_overlay = frame.copy()
                    cv2.circle(glow_overlay, (cx, cy), 25, (100, 220, 255), -1, cv2.LINE_AA)
                    cv2.addWeighted(glow_overlay, 0.2, frame, 0.8, 0, frame)

                    # Thin elegant ring
                    cv2.circle(frame, (cx, cy), 20, (150, 230, 255), 2, cv2.LINE_AA)

            output_video_frames.append(frame)

        return output_video_frames
