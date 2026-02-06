"""
Multi-Video Padel Analysis Script

Processes multiple videos with cumulative stats, side switching support,
and scoreboard display.
"""

import cv2
import numpy as np
import pickle
import os
from ultralytics import YOLO
import pandas as pd
from copy import deepcopy

from utils import (read_video, save_video, measure_distance, draw_player_stats,
                   convert_pixel_distance_to_meters, get_center_of_bbox)
import constants
from trackers import PlayerTracker
from trackers.tracknet_ball_tracker import TrackNetBallTracker
from court_line_detector import PadelCourtDetectorColor
from mini_court import MiniCourt

# ============================================================================
# VIDEO CONFIGURATION
# ============================================================================
VIDEO_CONFIGS = [
    {
        'video_path': 'inputs/video_1.mp4',
        'shot_data_path': 'shot_data_video_1.csv',
        'sides_switched': False,  # Green team (P1,P2) near, Red team (P3,P4) far
        'score_team1': 0,  # Team 1 (Green) score
        'score_team2': 0,  # Team 2 (Red) score
        'point_winner': 1,  # Team 1 wins this point
        'winning_reason': "Ball hit P3 body",
        'highlight_players': [1, 3],  # Players to highlight in replay
    },
    {
        'video_path': 'inputs/video_3.mp4',
        'shot_data_path': 'shot_data_video_3.csv',
        'sides_switched': True,  # Red team (P3,P4) near, Green team (P1,P2) far
        'score_team1': 15,  # Team 1 won video_1, now 15-0
        'score_team2': 0,  # Score after previous point
        'point_winner': 2,  # Team 2 wins this point
        'winning_reason': "Ball bounced out",
        'highlight_players': [3],  # Players to highlight in replay
    },
]

# ============================================================================
# POSE AND SKELETON SETTINGS
# ============================================================================
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

POSE_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
    [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
]

KEYPOINT_PARTS = [
    'head', 'eye', 'eye', 'head', 'head',
    'shoulder', 'shoulder', 'arm', 'arm', 'arm', 'arm',
    'hip', 'hip', 'leg', 'leg', 'foot', 'foot'
]

SKELETON_PART_MAP = {
    'shoulder': [5, 6], 'arm': [7, 8, 9, 10], 'hip': [11, 12],
    'leg': [13, 14], 'foot': [15, 16], 'torso': [0, 1, 2, 3, 4]
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
    overlay = frame.copy()

    for joint in POSE_SKELETON:
        if joint[0] in head_indices or joint[1] in head_indices:
            continue
        pt1, pt2 = keypoints[joint[0]], keypoints[joint[1]]
        if pt1[2] > 0.5 and pt2[2] > 0.5:
            part = get_skeleton_part(joint[0])
            line_color = PART_COLORS.get(part, (150, 150, 150))
            cv2.line(overlay, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                     line_color, thickness + 2, cv2.LINE_AA)

    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5 and i < len(KEYPOINT_PARTS) and i not in head_indices:
            part = KEYPOINT_PARTS[i]
            kpt_color = PART_COLORS.get(part, (100, 100, 100))
            cv2.circle(overlay, (int(x), int(y)), 5, kpt_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y)), 6, (30, 30, 30), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_scoreboard(frame, score_team1, score_team2, frame_w, frame_h):
    """Draw professional scoreboard at bottom center"""
    # Tennis scoring format: 0, 15, 30, 40
    score_map = {0: '0', 15: '15', 30: '30', 40: '40'}
    s1 = score_map.get(score_team1, str(score_team1))
    s2 = score_map.get(score_team2, str(score_team2))

    # Larger scoreboard dimensions
    sb_width = 400
    sb_height = 85
    sb_x = (frame_w - sb_width) // 2
    sb_y = frame_h - 110

    # Outer glow/shadow effect
    for i in range(3, 0, -1):
        cv2.rectangle(frame, (sb_x - i*2, sb_y - i*2),
                     (sb_x + sb_width + i*2, sb_y + sb_height + i*2),
                     (0, 0, 0), -1)

    # Main background with gradient effect (dark)
    cv2.rectangle(frame, (sb_x, sb_y), (sb_x + sb_width, sb_y + sb_height), (25, 25, 25), -1)

    # Team color strips at top
    strip_height = 8
    # Team 1 (Green) - left half
    cv2.rectangle(frame, (sb_x, sb_y), (sb_x + sb_width//2 - 3, sb_y + strip_height), (80, 200, 80), -1)
    # Team 2 (Red) - right half
    cv2.rectangle(frame, (sb_x + sb_width//2 + 3, sb_y), (sb_x + sb_width, sb_y + strip_height), (80, 80, 200), -1)

    # Outer border
    cv2.rectangle(frame, (sb_x, sb_y), (sb_x + sb_width, sb_y + sb_height), (180, 180, 180), 2)

    # Center divider with decorative elements
    center_x = sb_x + sb_width // 2
    cv2.line(frame, (center_x, sb_y + strip_height + 5), (center_x, sb_y + sb_height - 5), (120, 120, 120), 2)

    # "VS" or dash in center
    cv2.putText(frame, "-", (center_x - 10, sb_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Team 1 (Green) - Left side
    t1_center = sb_x + sb_width // 4
    cv2.putText(frame, "TEAM 1", (t1_center - 40, sb_y + 30), font, 0.55, (120, 220, 120), 1, cv2.LINE_AA)
    # Score - larger and centered, moved down
    (sw, sh), _ = cv2.getTextSize(s1, font, 1.5, 3)
    cv2.putText(frame, s1, (t1_center - sw//2, sb_y + 70), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    # Team 2 (Red) - Right side
    t2_center = sb_x + 3 * sb_width // 4
    cv2.putText(frame, "TEAM 2", (t2_center - 40, sb_y + 30), font, 0.55, (120, 120, 220), 1, cv2.LINE_AA)
    # Score - larger and centered, moved down
    (sw, sh), _ = cv2.getTextSize(s2, font, 1.5, 3)
    cv2.putText(frame, s2, (t2_center - sw//2, sb_y + 70), font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    return frame


def find_ball_landing_spot(ball_detections, shot_frame, next_shot_frame, shooter_id, sides_switched):
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

    # Adjust for side switching
    if sides_switched:
        # Red team (3,4) is near (high Y), Green team (1,2) is far (low Y)
        near_team = [3, 4]
        far_team = [1, 2]
    else:
        # Green team (1,2) is near (high Y), Red team (3,4) is far (low Y)
        near_team = [1, 2]
        far_team = [3, 4]

    if shooter_id in near_team:
        # Ball goes up (Y decreases) then down
        min_y = float('inf')
        bounce_point = None
        for frame_idx, x, y in positions:
            if y < min_y:
                min_y = y
                bounce_point = (int(x), int(y), frame_idx)
    else:
        # Ball goes down (Y increases) then up
        max_y = -1
        bounce_point = None
        for frame_idx, x, y in positions:
            if y > max_y:
                max_y = y
                bounce_point = (int(x), int(y), frame_idx)

    return bounce_point


def load_shot_data(csv_path, fps=30):
    """Load shot data from CSV file"""
    df = pd.read_csv(csv_path)
    shot_data = []
    for _, row in df.iterrows():
        frame_num = int(row['timestamp'] * fps)
        shot_data.append({
            'frame': frame_num,
            'player_id': int(row['player_id']),
            'shot_type': row['shot_type'],
            'second': row['timestamp']
        })
    return shot_data


def process_video(video_config, cumulative_stats, pose_model, fps=30):
    """Process a single video and return frames with stats"""
    video_path = video_config['video_path']
    shot_data_path = video_config['shot_data_path']
    sides_switched = video_config['sides_switched']
    score_t1 = video_config['score_team1']
    score_t2 = video_config['score_team2']

    print(f"\n{'='*60}")
    print(f"Processing: {video_path}")
    print(f"Sides switched: {sides_switched}")
    print(f"Score: {score_t1}-{score_t2}")
    print(f"{'='*60}")

    # Read video
    print("Reading video...")
    video_frames = read_video(video_path)
    print(f"Loaded {len(video_frames)} frames")

    # Load shot data
    shot_data = load_shot_data(shot_data_path, fps)
    print(f"Shots loaded: {len(shot_data)}")
    for shot in shot_data:
        print(f"  sec {shot['second']}: P{shot['player_id']} {shot['shot_type']}")

    # Setup trackers
    print("\nDetecting players...")
    player_tracker = PlayerTracker(model_path='yolov8x.pt')

    # Handle side switching in player tracker if needed
    if sides_switched:
        # Swap court polygon for switched sides
        player_tracker.COURT_POLYGON = np.array([
            [583, 323], [1343, 323], [1696, 942], [228, 944]
        ], dtype=np.float32)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    stub_path = f'tracker_stubs/{video_name}_player_detections.pkl'

    if os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            player_detections = pickle.load(f)
        print("Loaded player detections from cache")
    else:
        player_detections = player_tracker.detect_frames(video_frames)
        os.makedirs('tracker_stubs', exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(player_detections, f)

    # Swap player IDs if sides are switched
    # In video_3: Red team (P3,P4) is at bottom, Green team (P1,P2) is at top
    # But tracker assigns P1,P2 to bottom by default, so we swap
    if sides_switched:
        print("Swapping player IDs for switched sides...")
        swapped_detections = []
        swap_map = {1: 3, 2: 4, 3: 1, 4: 2}  # P1↔P3, P2↔P4
        for frame_dict in player_detections:
            new_dict = {}
            for pid, bbox in frame_dict.items():
                new_pid = swap_map.get(pid, pid)
                new_dict[new_pid] = bbox
            swapped_detections.append(new_dict)
        player_detections = swapped_detections

    # Ball detection
    print("\nDetecting ball...")
    ball_tracker = TrackNetBallTracker(
        model_path='models/weights/ball_detection/TrackNet_best.pt',
        device='cuda'
    )

    ball_stub_path = f'tracker_stubs/{video_name}_ball_detections.pkl'
    if os.path.exists(ball_stub_path):
        with open(ball_stub_path, 'rb') as f:
            ball_detections = pickle.load(f)
        print("Loaded ball detections from cache")
    else:
        ball_detections = ball_tracker.detect_frames(video_frames)
        with open(ball_stub_path, 'wb') as f:
            pickle.dump(ball_detections, f)

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court detection
    print("\nDetecting court...")
    court_line_detector = PadelCourtDetectorColor(use_calibrated=True)
    court_keypoints = court_line_detector.predict(video_frames[0])

    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Mini court
    mini_court = MiniCourt(video_frames[0])

    # Convert coordinates
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints)

    # Calculate stats (using cumulative)
    print("\nCalculating player stats...")
    shot_frames = [s['frame'] for s in shot_data]
    shot_player_mapping = {s['frame']: s['player_id'] for s in shot_data}

    shot_types = ['Forehand', 'Backhand', 'Lob', 'Smash', 'Serve']

    # Start with cumulative stats or initialize
    if cumulative_stats is None:
        current_stats = {
            'frame_num': 0,
        }
        for pid in [1, 2, 3, 4]:
            current_stats[f'player_{pid}_number_of_shots'] = 0
            current_stats[f'player_{pid}_total_shot_speed'] = 0
            current_stats[f'player_{pid}_last_shot_speed'] = 0
            current_stats[f'player_{pid}_total_player_speed'] = 0
            current_stats[f'player_{pid}_last_player_speed'] = 0
            for st in shot_types:
                current_stats[f'player_{pid}_{st.lower()}'] = 0
        for st in shot_types:
            current_stats[f'best_{st.lower()}_speed'] = 0
            current_stats[f'best_{st.lower()}_player'] = 0
    else:
        current_stats = deepcopy(cumulative_stats)
        current_stats['frame_num'] = 0  # Reset frame number for this video

    player_stats_data = [deepcopy(current_stats)]

    # Determine teams based on side switching
    if sides_switched:
        team1_players = [1, 2]  # Green team (far side in video_3)
        team2_players = [3, 4]  # Red team (near side in video_3)
    else:
        team1_players = [1, 2]  # Green team (near side)
        team2_players = [3, 4]  # Red team (far side)

    # Process shots
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

        frame_stats = deepcopy(player_stats_data[-1])
        frame_stats['frame_num'] = start_frame
        frame_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        frame_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed
        frame_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed

        if shot_type.lower() in ['forehand', 'backhand', 'lob', 'smash', 'serve']:
            frame_stats[f'player_{player_shot_ball}_{shot_type.lower()}'] += 1
            if speed > frame_stats[f'best_{shot_type.lower()}_speed']:
                frame_stats[f'best_{shot_type.lower()}_speed'] = speed
                frame_stats[f'best_{shot_type.lower()}_player'] = player_shot_ball

        for opp_id in opponent_players:
            if opp_id in player_mini_court_detections[start_frame] and opp_id in player_mini_court_detections[end_frame]:
                opp_dist = measure_distance(player_mini_court_detections[start_frame][opp_id],
                                           player_mini_court_detections[end_frame][opp_id])
                opp_dist_m = convert_pixel_distance_to_meters(opp_dist, constants.DOUBLE_LINE_WIDTH,
                                                              mini_court.get_width_of_mini_court())
                opp_speed = opp_dist_m / ball_shot_time * 3.6
            else:
                opp_speed = 0
            frame_stats[f'player_{opp_id}_total_player_speed'] += opp_speed
            frame_stats[f'player_{opp_id}_last_player_speed'] = opp_speed

        player_stats_data.append(frame_stats)

    # Build stats DataFrame
    player_stats_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_df = pd.merge(frames_df, player_stats_df, on='frame_num', how='left')
    player_stats_df = player_stats_df.ffill().bfill().fillna(0)

    for pid in [1, 2, 3, 4]:
        player_stats_df[f'player_{pid}_average_shot_speed'] = (
            player_stats_df[f'player_{pid}_total_shot_speed'] /
            player_stats_df[f'player_{pid}_number_of_shots'].replace(0, 1)
        )

    # Add progressive stats (distance, zones, etc.)
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

    rally_length_arr = []
    current_rally_shots = 0

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
                    if frame_speeds[pid] > player_top_speed[pid]:
                        player_top_speed[pid] = frame_speeds[pid]

        for pid in [1, 2, 3, 4]:
            prev_speed = player_live_speed_arr[pid][-1] if player_live_speed_arr[pid] else 0
            player_live_speed_arr[pid].append(0.3 * frame_speeds[pid] + 0.7 * prev_speed)
            player_avg_speed_arr[pid].append(sum(player_speeds[pid]) / len(player_speeds[pid]) if player_speeds[pid] else 0)
            player_top_speed_arr[pid].append(player_top_speed[pid])

        if frame_idx in shot_frames:
            current_rally_shots += 1
        rally_length_arr.append(current_rally_shots)

        # Zone tracking (adjusted for side switching)
        if sides_switched:
            # In video_3: P3,P4 are near (high Y), P1,P2 are far (low Y)
            for pid in [3, 4]:
                if pid in player_mini_court_detections[frame_idx]:
                    pos_y = player_mini_court_detections[frame_idx][pid][1]
                    if pos_y > mini_court.court_end_y - baseline_zone:
                        player_zones[pid]['baseline'] += 1
                    elif pos_y >= net_y and pos_y < net_y + net_zone:
                        player_zones[pid]['net'] += 1
                    else:
                        player_zones[pid]['midcourt'] += 1

            for pid in [1, 2]:
                if pid in player_mini_court_detections[frame_idx]:
                    pos_y = player_mini_court_detections[frame_idx][pid][1]
                    if pos_y < mini_court.court_start_y + baseline_zone:
                        player_zones[pid]['baseline'] += 1
                    elif pos_y <= net_y and pos_y > net_y - net_zone:
                        player_zones[pid]['net'] += 1
                    else:
                        player_zones[pid]['midcourt'] += 1
        else:
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

    player_stats_df['current_rally_length'] = rally_length_arr
    player_stats_df['average_rally_length'] = rally_length_arr

    # Draw visualizations
    print("\nDrawing visualizations...")
    output_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_frames = ball_tracker.draw_bboxes(output_frames, ball_detections, shot_frames)
    output_frames = court_line_detector.draw_keypoints_on_video(output_frames, court_keypoints)
    output_frames = mini_court.draw_mini_court_with_live_heatmap(
        output_frames, player_mini_court_detections, ball_mini_court_detections,
        shot_frames, shot_player_mapping, alternate_interval=90)
    output_frames = draw_player_stats(output_frames, player_stats_df)

    # Add scoreboard to all frames
    frame_h, frame_w = output_frames[0].shape[:2]
    for frame in output_frames:
        draw_scoreboard(frame, score_t1, score_t2, frame_w, frame_h)

    # Create frames with shot pauses
    print("\nAdding shot pauses...")
    final_frames = []
    pause_duration = fps
    flash_cycles = 3
    shot_frames_dict = {shot['frame']: shot for shot in shot_data}

    team_colors = {1: (0, 255, 0), 2: (0, 200, 100), 3: (0, 0, 255), 4: (0, 100, 255)}
    shot_type_colors = {'Forehand': (100, 255, 100), 'Backhand': (100, 100, 255),
                        'Lob': (255, 200, 100), 'Smash': (100, 200, 255), 'Serve': (255, 255, 100)}

    # Pre-compute poses for shots
    shot_poses = {}
    for shot in shot_data:
        frame_num = shot['frame']
        player_id = shot['player_id']
        bbox = player_detections[frame_num].get(player_id)
        pose = None
        if bbox:
            x1, y1, x2, y2 = bbox
            padding = max(x2 - x1, y2 - y1) * 0.5
            crop_x1 = max(0, int(x1 - padding))
            crop_y1 = max(0, int(y1 - padding))
            crop_x2 = min(frame_w, int(x2 + padding))
            crop_y2 = min(frame_h, int(y2 + padding))
            cropped = video_frames[frame_num][crop_y1:crop_y2, crop_x1:crop_x2].copy()

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
                    pose = np.array(adjusted)
        shot_poses[frame_num] = pose

    for frame_idx in range(len(output_frames)):
        frame = output_frames[frame_idx].copy()

        if frame_idx in shot_frames_dict:
            shot = shot_frames_dict[frame_idx]
            shooter_id = shot['player_id']
            shot_type = shot['shot_type']
            pose = shot_poses.get(frame_idx)

            for pause_frame in range(pause_duration):
                pause_img = output_frames[frame_idx].copy()
                cycle_progress = (pause_frame / pause_duration) * flash_cycles * 2 * np.pi
                alpha = (np.sin(cycle_progress) + 1) / 2

                if pose is not None:
                    draw_pose_on_frame(pause_img, pose, alpha=alpha * 0.8 + 0.2, thickness=3)

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

                    cv2.rectangle(pause_img, (label_x - text_w//2 - 10, label_y - text_h - 10),
                                 (label_x + text_w//2 + 10, label_y + 10), (0, 0, 0), -1)
                    cv2.rectangle(pause_img, (label_x - text_w//2 - 10, label_y - text_h - 10),
                                 (label_x + text_w//2 + 10, label_y + 10), color, 2)
                    cv2.putText(pause_img, text, (label_x - text_w//2, label_y),
                               font, font_scale, color, thickness, cv2.LINE_AA)

                    player_text = f"P{shooter_id}"
                    cv2.putText(pause_img, player_text, (label_x - 15, label_y + 35),
                               font, 0.8, team_colors[shooter_id], 2, cv2.LINE_AA)

                final_frames.append(pause_img)
        else:
            final_frames.append(frame)

    # Get final cumulative stats
    final_stats = player_stats_data[-1] if player_stats_data else current_stats

    return {
        'frames': final_frames,
        'video_frames': video_frames,
        'player_detections': player_detections,
        'shot_data': shot_data,
        'cumulative_stats': final_stats,
        'frame_h': frame_h,
        'frame_w': frame_w,
        'pose_model': pose_model,
    }


def create_replay(video_data, video_config, pose_model, logo):
    """Create a highlight replay for a video"""
    video_frames = video_data['video_frames']
    player_detections = video_data['player_detections']
    shot_data = video_data['shot_data']
    frame_h = video_data['frame_h']
    frame_w = video_data['frame_w']
    point_winner = video_config['point_winner']
    winning_reason = video_config['winning_reason']
    highlight_players = video_config.get('highlight_players', None)

    fps = 30
    replay_frames = []

    # Determine winning team for title card
    if point_winner == 1:
        winning_team_players = [1, 2]
        team_name = "TEAM 1"
        team_color = (100, 255, 100)  # Green
    else:
        winning_team_players = [3, 4]
        team_name = "TEAM 2"
        team_color = (100, 100, 255)  # Red

    # Use custom highlight_players if specified, otherwise use winning team
    if highlight_players is None:
        highlight_players = winning_team_players

    last_winning_shot = None
    for shot in reversed(shot_data):
        if shot['player_id'] in winning_team_players:
            last_winning_shot = shot
            break

    if not last_winning_shot:
        last_winning_shot = shot_data[-1] if shot_data else None

    if not last_winning_shot:
        return []

    highlight_start = last_winning_shot['frame']
    highlight_end = len(video_frames) - 1

    # Prepare logos
    logo_small = None
    logo_large = None
    if logo is not None:
        logo_h_small = 60
        aspect = logo.shape[1] / logo.shape[0]
        logo_w_small = int(logo_h_small * aspect)
        logo_small = cv2.resize(logo, (logo_w_small, logo_h_small))

        logo_h_large = 200
        logo_w_large = int(logo_h_large * aspect)
        logo_large = cv2.resize(logo, (logo_w_large, logo_h_large))

    def overlay_logo(frame, logo_img, x, y, alpha=1.0):
        if logo_img is None:
            return frame
        h, w = logo_img.shape[:2]
        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            return frame
        if logo_img.shape[2] == 4:
            logo_alpha = (logo_img[:, :, 3] / 255.0) * alpha
            logo_alpha = np.dstack([logo_alpha] * 3)
            logo_rgb = logo_img[:, :, :3]
            roi = frame[y:y+h, x:x+w].astype(np.float32)
            blended = logo_rgb * logo_alpha + roi * (1 - logo_alpha)
            frame[y:y+h, x:x+w] = blended.astype(np.uint8)
        return frame

    # Transition to black
    print("  Adding transition...")
    last_frame = video_data['frames'][-1].copy()
    for i in range(60):
        progress = i / 60
        ease = 0.5 - 0.5 * np.cos(progress * np.pi)
        black = np.zeros_like(last_frame)
        frame = cv2.addWeighted(last_frame, 1 - ease, black, ease, 0)
        replay_frames.append(frame.astype(np.uint8))

    # Title card
    print("  Adding title card...")
    for i in range(int(fps * 1.5)):
        title_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        center_x, center_y = frame_w // 2, frame_h // 2
        Y, X = np.ogrid[:frame_h, :frame_w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        gradient = 1 - (dist / max_dist) * 0.7
        gradient = np.clip(gradient * 40, 10, 50).astype(np.uint8)
        title_frame[:, :, 0] = gradient
        title_frame[:, :, 1] = gradient
        title_frame[:, :, 2] = gradient

        progress = i / (fps * 1.5)
        text_alpha = min(1.0, progress * 3) if progress < 0.3 else (1.0 if progress < 0.7 else max(0, 1 - (progress - 0.7) * 3))

        if text_alpha > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Team name
            (ttw, tth), _ = cv2.getTextSize(team_name, font, 1.5, 4)
            cv2.putText(title_frame, team_name, ((frame_w - ttw) // 2, frame_h // 2 - 80),
                       font, 1.5, tuple(int(c * text_alpha) for c in team_color), 4, cv2.LINE_AA)

            # Winning point
            title = "WINNING POINT"
            (tw, th), _ = cv2.getTextSize(title, font, 3.0, 8)
            main_color = (int(100 * text_alpha), int(215 * text_alpha), int(255 * text_alpha))
            cv2.putText(title_frame, title, ((frame_w - tw) // 2, frame_h // 2 + 20),
                       font, 3.0, main_color, 8, cv2.LINE_AA)

            # Reason
            (rw, rh), _ = cv2.getTextSize(winning_reason, font, 1.0, 2)
            cv2.putText(title_frame, winning_reason, ((frame_w - rw) // 2, frame_h // 2 + th + 50),
                       font, 1.0, (int(150 * text_alpha), int(220 * text_alpha), int(150 * text_alpha)), 2, cv2.LINE_AA)

        bar_height = int(frame_h * 0.08)
        cv2.rectangle(title_frame, (0, 0), (frame_w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(title_frame, (0, frame_h - bar_height), (frame_w, frame_h), (0, 0, 0), -1)

        if logo_large is not None and text_alpha > 0:
            logo_x = (frame_w - logo_large.shape[1]) // 2
            logo_y = frame_h - bar_height - logo_large.shape[0] - 30
            title_frame = overlay_logo(title_frame, logo_large, logo_x, logo_y, text_alpha * 0.85)

        replay_frames.append(title_frame)

    # Fade in to highlight
    first_highlight = video_frames[highlight_start].copy()
    for i in range(40):
        progress = i / 40
        ease = 0.5 - 0.5 * np.cos(progress * np.pi)
        black = np.zeros_like(first_highlight)
        frame = cv2.addWeighted(black, 1 - ease, first_highlight, ease, 0)
        bar_height = int(frame_h * 0.08)
        cv2.rectangle(frame, (0, 0), (frame_w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, frame_h - bar_height), (frame_w, frame_h), (0, 0, 0), -1)
        replay_frames.append(frame)

    # Slow motion highlight
    print("  Creating slow-motion highlight...")
    slowmo = 4
    replay_count = 0

    # Pre-compute poses for key players
    highlight_poses = {}
    for pid in highlight_players:
        highlight_poses[pid] = {}
        for frame_idx in range(highlight_start, highlight_end + 1):
            bbox = player_detections[frame_idx].get(pid)
            if bbox:
                x1, y1, x2, y2 = bbox
                padding = max(x2 - x1, y2 - y1) * 0.5
                crop_x1, crop_y1 = max(0, int(x1 - padding)), max(0, int(y1 - padding))
                crop_x2, crop_y2 = min(frame_w, int(x2 + padding)), min(frame_h, int(y2 + padding))
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
                        adjusted = [[float(kpts[idx][0]) / scale + crop_x1,
                                    float(kpts[idx][1]) / scale + crop_y1,
                                    float(kpts[idx][2])] for idx in range(17)]
                        highlight_poses[pid][frame_idx] = np.array(adjusted)

    for frame_idx in range(highlight_start, highlight_end + 1):
        raw = video_frames[frame_idx].copy()
        dimmed = (raw * 0.35).astype(np.uint8)

        mask = np.zeros((frame_h, frame_w), dtype=np.float32)
        for pid in highlight_players:
            bbox = player_detections[frame_idx].get(pid)
            if bbox:
                x1, y1, x2, y2 = bbox
                pad = 50
                bx1, by1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
                bx2, by2 = min(frame_w, int(x2 + pad)), min(frame_h, int(y2 + pad))
                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                w, h = (bx2 - bx1) // 2, (by2 - by1) // 2
                Y, X = np.ogrid[:frame_h, :frame_w]
                ellipse = ((X - cx) / (w + 30))**2 + ((Y - cy) / (h + 30))**2
                mask = np.maximum(mask, np.clip(1 - ellipse, 0, 1))

        mask = cv2.GaussianBlur(mask, (61, 61), 0)
        mask = np.clip(mask, 0, 1)
        mask_3ch = np.dstack([mask] * 3)

        highlight_frame = (raw * mask_3ch + dimmed * (1 - mask_3ch)).astype(np.uint8)
        highlight_frame = cv2.convertScaleAbs(highlight_frame, alpha=1.1, beta=5)

        bar_height = int(frame_h * 0.08)
        cv2.rectangle(highlight_frame, (0, 0), (frame_w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(highlight_frame, (0, frame_h - bar_height), (frame_w, frame_h), (0, 0, 0), -1)

        flash_alpha = (np.sin(replay_count * 0.3) + 1) / 2
        replay_color = (int(100 + 155 * flash_alpha), 215, 255)
        cv2.putText(highlight_frame, "REPLAY", (frame_w - 180, bar_height + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, replay_color, 2, cv2.LINE_AA)

        if logo_small is not None:
            highlight_frame = overlay_logo(highlight_frame, logo_small, 30,
                                          frame_h - bar_height - logo_small.shape[0] - 20, 0.7)

        for pid in highlight_players:
            pose = highlight_poses[pid].get(frame_idx)
            if pose is not None:
                draw_pose_on_frame(highlight_frame, pose, alpha=0.9, thickness=3)

        for _ in range(slowmo):
            replay_frames.append(highlight_frame.copy())
            replay_count += 1

    # Fade out
    print("  Adding ending fade...")
    last = replay_frames[-1].copy()
    for i in range(30):
        progress = i / 30
        ease = 0.5 - 0.5 * np.cos(progress * np.pi)
        black = np.zeros_like(last)
        replay_frames.append(cv2.addWeighted(last, 1 - ease, black, ease, 0))

    for _ in range(15):
        replay_frames.append(np.zeros((frame_h, frame_w, 3), dtype=np.uint8))

    return replay_frames


def main():
    fps = 30

    # Load pose model once
    print("Loading pose model...")
    pose_model = YOLO('yolo11m-pose.pt')

    # Load logo
    logo = None
    if os.path.exists("logo.png"):
        logo = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)
        print(f"Loaded logo: {logo.shape}")

    all_frames = []
    cumulative_stats = None

    for i, video_config in enumerate(VIDEO_CONFIGS):
        print(f"\n\n{'#'*60}")
        print(f"# VIDEO {i+1}: {video_config['video_path']}")
        print(f"{'#'*60}")

        # Process video
        video_data = process_video(video_config, cumulative_stats, pose_model, fps)

        # Add main video frames
        all_frames.extend(video_data['frames'])

        # Update cumulative stats
        cumulative_stats = video_data['cumulative_stats']

        # Create and add replay
        print("\nCreating replay...")
        replay_frames = create_replay(video_data, video_config, pose_model, logo)
        all_frames.extend(replay_frames)

    print(f"\n\nFinal combined video: {len(all_frames)} frames")

    # Save
    output_path = "output_videos/multi_video_analysis.mp4"
    os.makedirs("output_videos", exist_ok=True)
    save_video(all_frames, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
