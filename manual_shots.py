"""
Manual Shot Assignment and Visualization Script

This script uses manually assigned shot data (ground truth) to visualize
shots correctly, bypassing the automatic detection.
"""

import cv2
import numpy as np
import pickle
import os
from ultralytics import YOLO

from utils import (read_video, save_video, measure_distance, draw_player_stats,
                   convert_pixel_distance_to_meters, get_center_of_bbox)
import constants
from trackers import PlayerTracker
from trackers.tracknet_ball_tracker import TrackNetBallTracker
from court_line_detector import PadelCourtDetectorColor
from mini_court import MiniCourt
from copy import deepcopy
import pandas as pd


# ============================================================================
# GROUND TRUTH SHOTS - EDIT THIS SECTION WITH YOUR MANUAL ANNOTATIONS
# Format: (second, player_id, shot_type)
# player_id: 1=bottom-left, 2=bottom-right, 3=top-left, 4=top-right
# shot_type: 'Forehand' or 'Backhand'
# ============================================================================

# Winning point reason (shown in highlight reel)
WINNING_REASON = "Ball hit P3 body"

# Precise timestamps (skipping 1.4, starting from 2.3)
GROUND_TRUTH_SHOTS = [
    (1.900, 3, 'Forehand'),   # P3 Forehand
    (2.600, 2, 'Backhand'),   # P2 Backhand
    (3.433, 4, 'Backhand'),   # P4 Backhand
    (4.200, 1, 'Forehand'),   # P1 Forehand
    (5.300, 4, 'Lob'),        # P4 Lob
    (7.067, 2, 'Smash'),      # P2 Smash
    (7.833, 3, 'Backhand'),   # P3 Backhand
    (8.933, 2, 'Backhand'),   # P2 Backhand
    (10.300, 3, 'Lob'),       # P3 Lob
    (12.900, 1, 'Forehand'),  # P1 Forehand
    (15.867, 4, 'Lob'),       # P4 Lob
    (17.533, 1, 'Smash'),     # P1 Smash
    (19.367, 4, 'Forehand'),  # P4 Forehand
]

# ============================================================================

# Body-part specific colors (BGR)
PART_COLORS = {
    'head': (50, 50, 255),
    'eye': (0, 255, 255),
    'shoulder': (255, 170, 0),
    'arm': (0, 255, 170),
    'hip': (200, 100, 255),
    'leg': (180, 50, 255),
    'foot': (128, 128, 255),
    'torso': (170, 255, 170)
}

# COCO skeleton connections
POSE_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
    [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
]

KEYPOINT_PARTS = [
    'head', 'eye', 'eye', 'head', 'head',
    'shoulder', 'shoulder',
    'arm', 'arm',
    'arm', 'arm',
    'hip', 'hip',
    'leg', 'leg',
    'foot', 'foot'
]

SKELETON_PART_MAP = {
    'shoulder': [5, 6],
    'arm': [7, 8, 9, 10],
    'hip': [11, 12],
    'leg': [13, 14],
    'foot': [15, 16],
    'torso': [0, 1, 2, 3, 4]
}


def get_skeleton_part(idx):
    for part, indices in SKELETON_PART_MAP.items():
        if idx in indices:
            return part
    return 'torso'


def draw_pose_on_frame(frame, keypoints, alpha=1.0, thickness=2):
    """Draw pose skeleton on frame with optional transparency"""
    if keypoints is None:
        return frame

    head_indices = {0, 1, 2, 3, 4}
    ignored_indices = {0, 1, 2, 3, 4}

    # Create overlay for alpha blending
    overlay = frame.copy()

    # Draw skeleton connections
    for joint in POSE_SKELETON:
        if joint[0] in head_indices or joint[1] in head_indices:
            continue
        pt1, pt2 = keypoints[joint[0]], keypoints[joint[1]]
        if pt1[2] > 0.5 and pt2[2] > 0.5:
            part = get_skeleton_part(joint[0])
            line_color = PART_COLORS.get(part, (150, 150, 150))
            cv2.line(overlay, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                     line_color, thickness + 2, cv2.LINE_AA)

    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5 and i < len(KEYPOINT_PARTS):
            if i in ignored_indices:
                continue
            part = KEYPOINT_PARTS[i]
            kpt_color = PART_COLORS.get(part, (100, 100, 100))
            cv2.circle(overlay, (int(x), int(y)), 5, kpt_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y)), 6, (30, 30, 30), 2, cv2.LINE_AA)

    # Blend with original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_trajectory_arc(frame, start_pos, end_pos, color=(255, 165, 0), num_points=30):
    """Draw a smooth curved arc trajectory"""
    if start_pos is None or end_pos is None:
        return frame

    x1, y1 = start_pos
    x2, y2 = end_pos

    distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
    arc_height = min(distance * 0.3, 150)

    points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = x1 + t * (x2 - x1)
        base_y = y1 + t * (y2 - y1)
        arc_offset = arc_height * np.sin(t * np.pi)
        y = base_y - arc_offset
        points.append((int(x), int(y)))

    for i in range(1, len(points)):
        alpha = i / len(points)
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, points[i-1], points[i], color, thickness, cv2.LINE_AA)

    return frame


def draw_landing_marker(frame, landing_pos, color=(0, 255, 255)):
    """Draw a target marker at the ball landing position"""
    x, y = int(landing_pos[0]), int(landing_pos[1])
    cv2.circle(frame, (x, y), 25, color, 2)
    cv2.circle(frame, (x, y), 15, color, 2)
    cv2.circle(frame, (x, y), 5, color, -1)
    cv2.line(frame, (x - 30, y), (x + 30, y), color, 2)
    cv2.line(frame, (x, y - 30), (x, y + 30), color, 2)
    return frame


def find_ball_landing_spot(ball_detections, shot_frame, next_shot_frame, shooter_id):
    """Find where the ball bounces after a shot"""
    if next_shot_frame is None:
        next_shot_frame = min(shot_frame + 60, len(ball_detections) - 1)

    positions = []
    for frame_idx in range(shot_frame + 3, next_shot_frame):
        ball_pos = ball_detections[frame_idx].get(1)
        if ball_pos:
            ball_center_x = (ball_pos[0] + ball_pos[2]) / 2
            ball_center_y = (ball_pos[1] + ball_pos[3]) / 2
            positions.append((frame_idx, ball_center_x, ball_center_y))

    if len(positions) < 5:
        return None

    # Players 1,2 at bottom (high Y), Players 3,4 at top (low Y)
    if shooter_id in [1, 2]:
        # Ball goes up (Y decreases) then down - bounce at min Y
        min_y = float('inf')
        bounce_point = None
        for frame_idx, x, y in positions:
            if y < min_y:
                min_y = y
                bounce_point = (int(x), int(y), frame_idx)
    else:
        # Ball goes down (Y increases) then up - bounce at max Y
        max_y = -1
        bounce_point = None
        for frame_idx, x, y in positions:
            if y > max_y:
                max_y = y
                bounce_point = (int(x), int(y), frame_idx)

    return bounce_point


def main():
    fps = 30
    input_video_path = "inputs/padel_match_20s.mp4"

    print("Reading video...")
    video_frames = read_video(input_video_path)
    print(f"Loaded {len(video_frames)} frames")

    # Convert ground truth to frame numbers
    shot_data = []
    for sec, player_id, shot_type in GROUND_TRUTH_SHOTS:
        frame_num = int(sec * fps)
        if frame_num < len(video_frames):
            shot_data.append({
                'frame': frame_num,
                'player_id': player_id,
                'shot_type': shot_type,
                'second': sec
            })

    print(f"\nManual shots loaded: {len(shot_data)}")
    for shot in shot_data:
        print(f"  sec {shot['second']}: P{shot['player_id']} {shot['shot_type']}")

    # Load player tracker
    print("\nDetecting players...")
    player_tracker = PlayerTracker(model_path='yolov8x.pt')

    stub_path = 'tracker_stubs/manual_player_detections_20s.pkl'
    if os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            player_detections = pickle.load(f)
        print(f"Loaded player detections from cache")
    else:
        player_detections = player_tracker.detect_frames(video_frames)
        os.makedirs('tracker_stubs', exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(player_detections, f)

    # Load ball tracker
    print("\nDetecting ball...")
    ball_tracker = TrackNetBallTracker(
        model_path='models/weights/ball_detection/TrackNet_best.pt',
        device='cuda'
    )

    ball_stub_path = 'tracker_stubs/manual_ball_detections_20s.pkl'
    if os.path.exists(ball_stub_path):
        with open(ball_stub_path, 'rb') as f:
            ball_detections = pickle.load(f)
        print(f"Loaded ball detections from cache")
    else:
        ball_detections = ball_tracker.detect_frames(video_frames)
        with open(ball_stub_path, 'wb') as f:
            pickle.dump(ball_detections, f)

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court detection
    print("\nDetecting court...")
    court_line_detector = PadelCourtDetectorColor(use_calibrated=True)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Filter players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Mini court
    mini_court = MiniCourt(video_frames[0])

    # Load pose model
    print("\nLoading pose model...")
    pose_model = YOLO('yolo11m-pose.pt')

    # Build shot player mapping and counts
    shot_frames = [s['frame'] for s in shot_data]
    shot_player_mapping = {s['frame']: s['player_id'] for s in shot_data}
    player_shot_counts = {}
    for s in shot_data:
        pid = s['player_id']
        player_shot_counts[pid] = player_shot_counts.get(pid, 0) + 1

    # Convert to mini court coordinates
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints)

    # Calculate player stats
    print("\nCalculating player stats...")

    # Initialize shot type counters
    shot_types = ['Forehand', 'Backhand', 'Lob', 'Smash']
    initial_stats = {
        'frame_num': 0,
        'player_1_number_of_shots': 0, 'player_1_total_shot_speed': 0, 'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0, 'player_1_last_player_speed': 0,
        'player_2_number_of_shots': 0, 'player_2_total_shot_speed': 0, 'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0, 'player_2_last_player_speed': 0,
        'player_3_number_of_shots': 0, 'player_3_total_shot_speed': 0, 'player_3_last_shot_speed': 0,
        'player_3_total_player_speed': 0, 'player_3_last_player_speed': 0,
        'player_4_number_of_shots': 0, 'player_4_total_shot_speed': 0, 'player_4_last_shot_speed': 0,
        'player_4_total_player_speed': 0, 'player_4_last_player_speed': 0,
    }
    # Add shot type counters for each player
    for pid in [1, 2, 3, 4]:
        for st in shot_types:
            initial_stats[f'player_{pid}_{st.lower()}'] = 0

    # Add best shot tracking (fastest shot per type)
    for st in shot_types:
        initial_stats[f'best_{st.lower()}_speed'] = 0
        initial_stats[f'best_{st.lower()}_player'] = 0

    player_stats_data = [initial_stats]

    team1_players = [1, 2]
    team2_players = [3, 4]

    for i in range(len(shot_data) - 1):
        start_frame = shot_data[i]['frame']
        end_frame = shot_data[i + 1]['frame']
        ball_shot_time = (end_frame - start_frame) / fps

        if 1 not in ball_mini_court_detections[start_frame] or 1 not in ball_mini_court_detections[end_frame]:
            continue

        dist_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                       ball_mini_court_detections[end_frame][1])
        dist_meters = convert_pixel_distance_to_meters(dist_pixels, constants.DOUBLE_LINE_WIDTH,
                                                       mini_court.get_width_of_mini_court())
        speed = dist_meters / ball_shot_time * 3.6

        player_shot_ball = shot_data[i]['player_id']
        shot_type = shot_data[i].get('shot_type', 'Forehand')
        opponent_players = team2_players if player_shot_ball in team1_players else team1_players

        current_stats = deepcopy(player_stats_data[-1])
        current_stats['frame_num'] = start_frame
        current_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed
        current_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed
        # Track shot type
        if shot_type.lower() in ['forehand', 'backhand', 'lob', 'smash']:
            current_stats[f'player_{player_shot_ball}_{shot_type.lower()}'] += 1
            # Track best/fastest shot per type
            if speed > current_stats[f'best_{shot_type.lower()}_speed']:
                current_stats[f'best_{shot_type.lower()}_speed'] = speed
                current_stats[f'best_{shot_type.lower()}_player'] = player_shot_ball

        for opp_id in opponent_players:
            if opp_id in player_mini_court_detections[start_frame] and opp_id in player_mini_court_detections[end_frame]:
                opp_dist = measure_distance(player_mini_court_detections[start_frame][opp_id],
                                           player_mini_court_detections[end_frame][opp_id])
                opp_dist_m = convert_pixel_distance_to_meters(opp_dist, constants.DOUBLE_LINE_WIDTH,
                                                              mini_court.get_width_of_mini_court())
                opp_speed = opp_dist_m / ball_shot_time * 3.6
            else:
                opp_speed = 0
            current_stats[f'player_{opp_id}_total_player_speed'] += opp_speed
            current_stats[f'player_{opp_id}_last_player_speed'] = opp_speed

        player_stats_data.append(current_stats)

    player_stats_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_df = pd.merge(frames_df, player_stats_df, on='frame_num', how='left')
    player_stats_df = player_stats_df.ffill()

    for pid in [1, 2, 3, 4]:
        player_stats_df[f'player_{pid}_average_shot_speed'] = (
            player_stats_df[f'player_{pid}_total_shot_speed'] /
            player_stats_df[f'player_{pid}_number_of_shots'].replace(0, 1)
        )

    # Add progressive stats
    court_height = mini_court.court_end_y - mini_court.court_start_y
    half_court = court_height / 2
    baseline_zone = half_court * 0.35
    net_zone = half_court * 0.25
    net_y = (mini_court.court_start_y + mini_court.court_end_y) / 2

    player_distance_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_baseline_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_midcourt_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_net_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_live_speed_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_avg_speed_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_top_speed_arr = {pid: [] for pid in [1, 2, 3, 4]}

    player_total_dist = {pid: 0 for pid in [1, 2, 3, 4]}
    player_zones = {pid: {'baseline': 0, 'midcourt': 0, 'net': 0} for pid in [1, 2, 3, 4]}
    player_speeds = {pid: [] for pid in [1, 2, 3, 4]}
    player_top_speed = {pid: 0 for pid in [1, 2, 3, 4]}

    # Rally tracking
    rally_lengths = []  # List of shots per rally
    current_rally_shots = 0
    rally_length_arr = []
    avg_rally_length_arr = []

    for frame_idx in range(len(player_mini_court_detections)):
        frame_speeds = {pid: 0 for pid in [1, 2, 3, 4]}

        if frame_idx > 0:
            for pid in [1, 2, 3, 4]:
                if pid in player_mini_court_detections[frame_idx] and pid in player_mini_court_detections[frame_idx - 1]:
                    dist = measure_distance(player_mini_court_detections[frame_idx][pid],
                                           player_mini_court_detections[frame_idx - 1][pid])
                    dist_m = convert_pixel_distance_to_meters(dist, constants.DOUBLE_LINE_WIDTH,
                                                              mini_court.get_width_of_mini_court())
                    player_total_dist[pid] += dist_m
                    frame_speeds[pid] = dist_m * fps * 3.6
                    player_speeds[pid].append(frame_speeds[pid])
                    # Track top speed
                    if frame_speeds[pid] > player_top_speed[pid]:
                        player_top_speed[pid] = frame_speeds[pid]

        for pid in [1, 2, 3, 4]:
            prev_speed = player_live_speed_arr[pid][-1] if player_live_speed_arr[pid] else 0
            player_live_speed_arr[pid].append(0.3 * frame_speeds[pid] + 0.7 * prev_speed)
            player_avg_speed_arr[pid].append(sum(player_speeds[pid]) / len(player_speeds[pid]) if player_speeds[pid] else 0)
            player_top_speed_arr[pid].append(player_top_speed[pid])

        # Track rally length (count shots in current rally)
        if frame_idx in shot_frames:
            current_rally_shots += 1
        rally_length_arr.append(current_rally_shots)
        avg_rally_length_arr.append(sum(rally_lengths) / len(rally_lengths) if rally_lengths else current_rally_shots)

        for pid in [1, 2]:
            if pid in player_mini_court_detections[frame_idx]:
                pos_y = player_mini_court_detections[frame_idx][pid][1]
                if pos_y > mini_court.court_end_y - baseline_zone:
                    player_zones[pid]['baseline'] += 1
                elif pos_y >= net_y and pos_y < net_y + net_zone:
                    player_zones[pid]['net'] += 1
                else:
                    player_zones[pid]['midcourt'] += 1

        for pid in [3, 4]:
            if pid in player_mini_court_detections[frame_idx]:
                pos_y = player_mini_court_detections[frame_idx][pid][1]
                if pos_y < mini_court.court_start_y + baseline_zone:
                    player_zones[pid]['baseline'] += 1
                elif pos_y <= net_y and pos_y > net_y - net_zone:
                    player_zones[pid]['net'] += 1
                else:
                    player_zones[pid]['midcourt'] += 1

        for pid in [1, 2, 3, 4]:
            player_distance_arr[pid].append(player_total_dist[pid])
            total_zone = sum(player_zones[pid].values()) or 1
            player_baseline_arr[pid].append(player_zones[pid]['baseline'] / total_zone * 100)
            player_midcourt_arr[pid].append(player_zones[pid]['midcourt'] / total_zone * 100)
            player_net_arr[pid].append(player_zones[pid]['net'] / total_zone * 100)

    for pid in [1, 2, 3, 4]:
        player_stats_df[f'player_{pid}_distance_meters'] = player_distance_arr[pid]
        player_stats_df[f'player_{pid}_baseline_pct'] = player_baseline_arr[pid]
        player_stats_df[f'player_{pid}_midcourt_pct'] = player_midcourt_arr[pid]
        player_stats_df[f'player_{pid}_net_pct'] = player_net_arr[pid]
        player_stats_df[f'player_{pid}_last_player_speed'] = player_live_speed_arr[pid]
        player_stats_df[f'player_{pid}_average_player_speed'] = player_avg_speed_arr[pid]
        player_stats_df[f'player_{pid}_top_speed'] = player_top_speed_arr[pid]

    # Add rally stats
    player_stats_df['current_rally_length'] = rally_length_arr
    player_stats_df['average_rally_length'] = avg_rally_length_arr

    # Draw all visualizations
    print("\nDrawing visualizations...")
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections, shot_frames)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court_with_live_heatmap(
        output_video_frames, player_mini_court_detections, ball_mini_court_detections,
        shot_frames, shot_player_mapping, alternate_interval=90)
    output_video_frames = draw_player_stats(output_video_frames, player_stats_df)

    # Pre-compute poses and landing spots
    print("\nComputing poses and trajectories...")
    shot_poses = {}
    shot_landings = {}

    for i, shot in enumerate(shot_data):
        frame_num = shot['frame']
        player_id = shot['player_id']

        # Get pose at shot frame
        bbox = player_detections[frame_num].get(player_id)
        pose = None
        if bbox:
            x1, y1, x2, y2 = bbox
            padding = max(x2 - x1, y2 - y1) * 0.5
            crop_x1 = max(0, int(x1 - padding))
            crop_y1 = max(0, int(y1 - padding))
            crop_x2 = min(video_frames[frame_num].shape[1], int(x2 + padding))
            crop_y2 = min(video_frames[frame_num].shape[0], int(y2 + padding))
            cropped = video_frames[frame_num][crop_y1:crop_y2, crop_x1:crop_x2].copy()

            h, w = cropped.shape[:2]
            scale = 1.0
            if w < 300 or h < 300:
                scale = 300 / min(w, h)
                cropped = cv2.resize(cropped, (int(w * scale), int(h * scale)))

            results = pose_model(cropped, verbose=False)[0]
            if results.keypoints is not None and len(results.keypoints.data) > 0:
                kpts = results.keypoints.data[0]
                adjusted = []
                for idx in range(17):
                    kx, ky, conf = float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])
                    orig_x = (kx / scale) + crop_x1
                    orig_y = (ky / scale) + crop_y1
                    adjusted.append([orig_x, orig_y, conf])
                pose = np.array(adjusted)

        shot_poses[frame_num] = pose

        # Get landing spot
        next_frame = shot_data[i + 1]['frame'] if i + 1 < len(shot_data) else None
        landing = find_ball_landing_spot(ball_detections, frame_num, next_frame, player_id)
        shot_landings[frame_num] = landing

    # Create final video with 1-second pause and flashing pose
    print("\nCreating final video with effects...")
    final_frames = []
    pause_duration = fps  # 1 second pause = 30 frames at 30fps
    flash_cycles = 3  # Number of fade in/out cycles

    shot_frames_dict = {shot['frame']: shot for shot in shot_data}

    team_colors = {
        1: (0, 255, 0),
        2: (0, 200, 100),
        3: (0, 0, 255),
        4: (0, 100, 255),
    }

    shot_type_colors = {
        'Forehand': (100, 255, 100),
        'Backhand': (100, 100, 255),
    }

    for frame_idx in range(len(output_video_frames)):
        frame = output_video_frames[frame_idx].copy()

        if frame_idx in shot_frames_dict:
            shot = shot_frames_dict[frame_idx]
            shooter_id = shot['player_id']
            shot_type = shot['shot_type']
            pose = shot_poses.get(frame_idx)

            # Create pause frames with flashing pose
            for pause_frame in range(pause_duration):
                pause_img = output_video_frames[frame_idx].copy()

                # Calculate flash alpha (sine wave for smooth fade in/out)
                cycle_progress = (pause_frame / pause_duration) * flash_cycles * 2 * np.pi
                alpha = (np.sin(cycle_progress) + 1) / 2  # 0 to 1

                # Draw pose with flashing effect
                if pose is not None:
                    draw_pose_on_frame(pause_img, pose, alpha=alpha * 0.8 + 0.2, thickness=3)

                # Draw shot type label
                bbox = player_detections[frame_idx].get(shooter_id)
                if bbox:
                    label_x = int((bbox[0] + bbox[2]) / 2)
                    label_y = int(bbox[1]) - 60

                    text = shot_type.upper()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2
                    thickness = 3
                    color = shot_type_colors.get(shot_type, (255, 255, 255))

                    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

                    # Background
                    cv2.rectangle(pause_img,
                                 (label_x - text_w//2 - 10, label_y - text_h - 10),
                                 (label_x + text_w//2 + 10, label_y + 10),
                                 (0, 0, 0), -1)
                    cv2.rectangle(pause_img,
                                 (label_x - text_w//2 - 10, label_y - text_h - 10),
                                 (label_x + text_w//2 + 10, label_y + 10),
                                 color, 2)

                    # Text
                    cv2.putText(pause_img, text,
                               (label_x - text_w//2, label_y),
                               font, font_scale, color, thickness, cv2.LINE_AA)

                    # Player indicator
                    player_text = f"P{shooter_id}"
                    cv2.putText(pause_img, player_text,
                               (label_x - 15, label_y + 35),
                               font, 0.8, team_colors[shooter_id], 2, cv2.LINE_AA)

                final_frames.append(pause_img)
        else:
            final_frames.append(frame)

    # =========================================================================
    # PROFESSIONAL HIGHLIGHT REEL - TEAM 1 WINNING POINT
    # =========================================================================
    print("\nCreating professional highlight reel...")

    # Load logo
    logo = None
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        print(f"  Loaded logo: {logo.shape}")

    # Find the last Team 1 shot (winning point) - Team 1 is P1 and P2
    last_team1_shot = None
    for shot in reversed(shot_data):
        if shot['player_id'] in [1, 2]:
            last_team1_shot = shot
            break

    if last_team1_shot:
        highlight_start_frame = last_team1_shot['frame']
        highlight_end_frame = len(video_frames) - 1

        frame_h, frame_w = video_frames[0].shape[:2]

        # Prepare logo for overlay (resize to fit)
        logo_small = None
        logo_large = None
        if logo is not None:
            # Small logo for replay watermark (corner)
            logo_h_small = 60
            aspect = logo.shape[1] / logo.shape[0]
            logo_w_small = int(logo_h_small * aspect)
            logo_small = cv2.resize(logo, (logo_w_small, logo_h_small))

            # Large logo for transition
            logo_h_large = 200
            logo_w_large = int(logo_h_large * aspect)
            logo_large = cv2.resize(logo, (logo_w_large, logo_h_large))

        def overlay_logo(frame, logo_img, x, y, alpha=1.0):
            """Overlay logo with alpha channel support"""
            if logo_img is None:
                return frame
            h, w = logo_img.shape[:2]
            if y + h > frame.shape[0] or x + w > frame.shape[1]:
                return frame
            if logo_img.shape[2] == 4:  # Has alpha channel
                logo_alpha = (logo_img[:, :, 3] / 255.0) * alpha
                logo_alpha = np.dstack([logo_alpha] * 3)
                logo_rgb = logo_img[:, :, :3]
                roi = frame[y:y+h, x:x+w].astype(np.float32)
                blended = logo_rgb * logo_alpha + roi * (1 - logo_alpha)
                frame[y:y+h, x:x+w] = blended.astype(np.uint8)
            else:
                frame[y:y+h, x:x+w] = cv2.addWeighted(
                    logo_img, alpha, frame[y:y+h, x:x+w], 1 - alpha, 0)
            return frame

        # --- TRANSITION 1: Smooth fade to black ---
        print("  Adding transition effect...")
        transition_frames = 60  # Longer transition (~2 sec)
        last_main_frame = final_frames[-1].copy()

        for i in range(transition_frames):
            progress = i / transition_frames
            # Smooth ease-in-out curve
            ease = 0.5 - 0.5 * np.cos(progress * np.pi)

            # Fade to black
            black_frame = np.zeros_like(last_main_frame)
            frame = cv2.addWeighted(last_main_frame, 1 - ease, black_frame, ease, 0)

            final_frames.append(frame.astype(np.uint8))

        # --- TITLE CARD: "TEAM 1 - WINNING POINT" ---
        print("  Adding title card...")
        title_duration = int(fps * 1.5)

        for i in range(title_duration):
            title_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

            # Radial gradient background
            center_x, center_y = frame_w // 2, frame_h // 2
            Y, X = np.ogrid[:frame_h, :frame_w]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            gradient = 1 - (dist_from_center / max_dist) * 0.7
            gradient = np.clip(gradient * 40, 10, 50).astype(np.uint8)

            title_frame[:, :, 0] = gradient
            title_frame[:, :, 1] = gradient
            title_frame[:, :, 2] = gradient

            progress = i / title_duration
            text_alpha = min(1.0, progress * 3) if progress < 0.3 else (1.0 if progress < 0.7 else max(0, 1 - (progress - 0.7) * 3))

            # Team indicator
            team_text = "TEAM 1"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (ttw, tth), _ = cv2.getTextSize(team_text, font, 1.5, 4)
            team_x = (frame_w - ttw) // 2
            team_y = frame_h // 2 - 80

            if text_alpha > 0:
                team_color = (int(100 * text_alpha), int(255 * text_alpha), int(100 * text_alpha))  # Green
                cv2.putText(title_frame, team_text, (team_x, team_y),
                           font, 1.5, team_color, 4, cv2.LINE_AA)

            # Main title
            title_text = "WINNING POINT"
            font_scale = 3.0
            thickness = 8

            (tw, th), _ = cv2.getTextSize(title_text, font, font_scale, thickness)
            title_x = (frame_w - tw) // 2
            title_y = frame_h // 2 + 20

            if text_alpha > 0:
                for glow in range(3, 0, -1):
                    glow_color = (int(50 * text_alpha), int(180 * text_alpha), int(255 * text_alpha))
                    cv2.putText(title_frame, title_text, (title_x, title_y),
                               font, font_scale, glow_color, thickness + glow * 4, cv2.LINE_AA)

                main_color = (int(100 * text_alpha), int(215 * text_alpha), int(255 * text_alpha))
                cv2.putText(title_frame, title_text, (title_x, title_y),
                           font, font_scale, main_color, thickness, cv2.LINE_AA)

            # Winning reason
            reason_scale = 1.0
            (rw, rh), _ = cv2.getTextSize(WINNING_REASON, font, reason_scale, 2)
            reason_x = (frame_w - rw) // 2
            reason_y = title_y + th + 50

            if text_alpha > 0:
                # Subtle background for reason text
                reason_color = (int(150 * text_alpha), int(220 * text_alpha), int(150 * text_alpha))
                cv2.putText(title_frame, WINNING_REASON, (reason_x, reason_y),
                           font, reason_scale, reason_color, 2, cv2.LINE_AA)

            # Subtitle
            subtitle = "REPLAY"
            sub_scale = 0.9
            (sw, sh), _ = cv2.getTextSize(subtitle, font, sub_scale, 2)
            sub_x = (frame_w - sw) // 2
            sub_y = reason_y + rh + 35

            if text_alpha > 0:
                sub_color = (int(180 * text_alpha), int(180 * text_alpha), int(180 * text_alpha))
                cv2.putText(title_frame, subtitle, (sub_x, sub_y),
                           font, sub_scale, sub_color, 2, cv2.LINE_AA)

            # Letterbox bars
            bar_height = int(frame_h * 0.08)
            cv2.rectangle(title_frame, (0, 0), (frame_w, bar_height), (0, 0, 0), -1)
            cv2.rectangle(title_frame, (0, frame_h - bar_height), (frame_w, frame_h), (0, 0, 0), -1)

            # Add logo at bottom center
            if logo_large is not None and text_alpha > 0:
                logo_x = (frame_w - logo_large.shape[1]) // 2
                logo_y = frame_h - bar_height - logo_large.shape[0] - 30
                title_frame = overlay_logo(title_frame, logo_large, logo_x, logo_y, text_alpha * 0.85)

            final_frames.append(title_frame)

        # --- TRANSITION 2: Fade from black to highlight ---
        fade_in_frames = 40  # Longer fade in
        first_highlight_frame = video_frames[highlight_start_frame].copy()

        for i in range(fade_in_frames):
            progress = i / fade_in_frames
            ease = 0.5 - 0.5 * np.cos(progress * np.pi)

            black_frame = np.zeros_like(first_highlight_frame)
            frame = cv2.addWeighted(black_frame, 1 - ease, first_highlight_frame, ease, 0)

            bar_height = int(frame_h * 0.08)
            cv2.rectangle(frame, (0, 0), (frame_w, bar_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, frame_h - bar_height), (frame_w, frame_h), (0, 0, 0), -1)

            final_frames.append(frame)

        # --- SLOW MOTION HIGHLIGHT (raw video, only P1 and P3 visible) ---
        print("  Creating slow-motion highlight...")
        slowmo_factor = 4

        # Pre-compute poses for P1 and P3 (players involved in final exchange)
        highlight_poses = {1: {}, 3: {}}
        print("    Computing poses for P1 and P3...")
        for frame_idx in range(highlight_start_frame, highlight_end_frame + 1):
            for pid in [1, 3]:
                bbox = player_detections[frame_idx].get(pid)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    padding = max(x2 - x1, y2 - y1) * 0.5
                    crop_x1 = max(0, int(x1 - padding))
                    crop_y1 = max(0, int(y1 - padding))
                    crop_x2 = min(frame_w, int(x2 + padding))
                    crop_y2 = min(frame_h, int(y2 + padding))
                    cropped = video_frames[frame_idx][crop_y1:crop_y2, crop_x1:crop_x2].copy()

                    h, w = cropped.shape[:2]
                    if h > 0 and w > 0:
                        scale = 1.0
                        if w < 300 or h < 300:
                            scale = 300 / min(w, h)
                            cropped = cv2.resize(cropped, (int(w * scale), int(h * scale)))

                        results = pose_model(cropped, verbose=False)[0]
                        if results.keypoints is not None and len(results.keypoints.data) > 0:
                            kpts = results.keypoints.data[0]
                            adjusted = []
                            for idx in range(17):
                                kx, ky, conf = float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])
                                orig_x = (kx / scale) + crop_x1
                                orig_y = (ky / scale) + crop_y1
                                adjusted.append([orig_x, orig_y, conf])
                            highlight_poses[pid][frame_idx] = np.array(adjusted)

        # Track frame count for flashing effect
        replay_frame_count = 0

        for frame_idx in range(highlight_start_frame, highlight_end_frame + 1):
            # Use RAW video frame (no stats/heatmaps)
            raw_frame = video_frames[frame_idx].copy()

            # Create dimmed/grayed out background (not fully black)
            dimmed_frame = (raw_frame * 0.35).astype(np.uint8)

            # Create mask for P1 and P3 to make them brighter
            mask = np.zeros((frame_h, frame_w), dtype=np.float32)

            for pid in [1, 3]:
                bbox = player_detections[frame_idx].get(pid)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    # Expand bbox
                    pad = 50
                    bx1 = max(0, int(x1 - pad))
                    by1 = max(0, int(y1 - pad))
                    bx2 = min(frame_w, int(x2 + pad))
                    by2 = min(frame_h, int(y2 + pad))

                    # Create soft elliptical spotlight for player
                    center_x = (bx1 + bx2) // 2
                    center_y = (by1 + by2) // 2
                    width = (bx2 - bx1) // 2
                    height = (by2 - by1) // 2

                    Y, X = np.ogrid[:frame_h, :frame_w]
                    ellipse = ((X - center_x) / (width + 30))**2 + ((Y - center_y) / (height + 30))**2
                    player_mask = np.clip(1 - ellipse, 0, 1)
                    mask = np.maximum(mask, player_mask)

            # Feather the mask edges
            mask = cv2.GaussianBlur(mask, (61, 61), 0)
            mask = np.clip(mask, 0, 1)
            mask_3ch = np.dstack([mask] * 3)

            # Blend: bright players on dimmed background
            highlight_frame = (raw_frame * mask_3ch + dimmed_frame * (1 - mask_3ch)).astype(np.uint8)

            # Slight contrast boost
            highlight_frame = cv2.convertScaleAbs(highlight_frame, alpha=1.1, beta=5)

            # Letterbox bars
            bar_height = int(frame_h * 0.08)
            cv2.rectangle(highlight_frame, (0, 0), (frame_w, bar_height), (0, 0, 0), -1)
            cv2.rectangle(highlight_frame, (0, frame_h - bar_height), (frame_w, frame_h), (0, 0, 0), -1)

            # Flashing "REPLAY" indicator (pulses in and out)
            flash_alpha = (np.sin(replay_frame_count * 0.3) + 1) / 2  # 0 to 1 pulsing
            replay_color = (int(100 + 155 * flash_alpha), int(215), int(255))
            cv2.putText(highlight_frame, "REPLAY", (frame_w - 180, bar_height + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, replay_color, 2, cv2.LINE_AA)

            # Logo watermark in bottom-left corner
            if logo_small is not None:
                logo_x = 30
                logo_y = frame_h - bar_height - logo_small.shape[0] - 20
                highlight_frame = overlay_logo(highlight_frame, logo_small, logo_x, logo_y, 0.7)

            # Draw poses for P1 and P3
            for pid in [1, 3]:
                pose = highlight_poses[pid].get(frame_idx)
                if pose is not None:
                    draw_pose_on_frame(highlight_frame, pose, alpha=0.9, thickness=3)

            # Repeat frame for slow motion
            for _ in range(slowmo_factor):
                final_frames.append(highlight_frame.copy())
                replay_frame_count += 1

        # --- ENDING: Fade to black ---
        print("  Adding ending fade...")
        fade_out_frames = 30
        last_highlight = final_frames[-1].copy()

        for i in range(fade_out_frames):
            progress = i / fade_out_frames
            ease = 0.5 - 0.5 * np.cos(progress * np.pi)

            black_frame = np.zeros_like(last_highlight)
            frame = cv2.addWeighted(last_highlight, 1 - ease, black_frame, ease, 0)
            final_frames.append(frame)

        # Hold black
        for _ in range(15):
            final_frames.append(np.zeros((frame_h, frame_w, 3), dtype=np.uint8))

    print(f"\nFinal video: {len(final_frames)} frames (original: {len(output_video_frames)})")

    # Save video
    output_path = "output_videos/manual_shots_20s_full.mp4"
    save_video(final_frames, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
