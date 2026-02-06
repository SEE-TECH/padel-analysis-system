import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils import (read_video,
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters,
                   get_center_of_bbox
                   )
import constants
from trackers import PlayerTracker, BallTracker, TrackNetBallTracker
from court_line_detector import CourtLineDetector, CourtLineDetectorCV, PadelCourtLineDetectorCV, PadelCourtDetectorColor
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy
from ultralytics import YOLO
import numpy as np

# Body-part specific colors (BGR) - matching visualizer exactly
PART_COLORS = {
    'head': (50, 50, 255),          # red-ish
    'eye': (0, 255, 255),           # yellow
    'shoulder': (255, 170, 0),      # blue-orange
    'arm': (0, 255, 170),           # greenish
    'hip': (200, 100, 255),         # purple-pink
    'leg': (180, 50, 255),          # magenta
    'foot': (128, 128, 255),        # lavender
    'torso': (170, 255, 170)        # pale green
}

# COCO skeleton connections (same as visualizer)
POSE_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
    [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
]

# Keypoint to body part mapping
KEYPOINT_PARTS = [
    'head', 'eye', 'eye', 'head', 'head',           # 0-4: nose, left/right eye, left/right ear
    'shoulder', 'shoulder',                         # 5-6
    'arm', 'arm',                                   # 7-8
    'arm', 'arm',                                   # 9-10
    'hip', 'hip',                                   # 11-12
    'leg', 'leg',                                   # 13-14
    'foot', 'foot'                                  # 15-16
]

# Part mapping for skeleton lines
SKELETON_PART_MAP = {
    'shoulder': [5, 6],
    'arm': [7, 8, 9, 10],
    'hip': [11, 12],
    'leg': [13, 14],
    'foot': [15, 16],
    'torso': [0, 1, 2, 3, 4]
}

def get_skeleton_part(idx):
    """Get body part for a keypoint index"""
    for part, indices in SKELETON_PART_MAP.items():
        if idx in indices:
            return part
    return 'torso'

def draw_pose_on_frame(frame, keypoints, color=None, thickness=2):
    """Draw pose skeleton on frame - matching visualizer exactly"""
    # keypoints shape: (17, 3) - x, y, confidence
    head_indices = {0, 1, 2, 3, 4}
    ignored_indices = {0, 1, 2, 3, 4}  # Skip nose, eyes, ears

    # Draw skeleton connections first
    for joint in POSE_SKELETON:
        # Skip lines involving head parts
        if joint[0] in head_indices or joint[1] in head_indices:
            continue

        pt1, pt2 = keypoints[joint[0]], keypoints[joint[1]]
        if pt1[2] > 0.5 and pt2[2] > 0.5:
            part = get_skeleton_part(joint[0])
            line_color = PART_COLORS.get(part, (150, 150, 150))
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                     line_color, thickness, cv2.LINE_AA)

    # Draw keypoints (skip head keypoints for cleaner look)
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5 and i < len(KEYPOINT_PARTS):
            if i in ignored_indices:
                continue  # skip head parts
            part = KEYPOINT_PARTS[i]
            kpt_color = PART_COLORS.get(part, (100, 100, 100))
            # Filled circle with dark border (radius 3, border radius 4)
            cv2.circle(frame, (int(x), int(y)), 3, kpt_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 4, (30, 30, 30), 1, cv2.LINE_AA)

    return frame

def get_player_who_shot(ball_position, player_detections):
    """Determine which player is closest to the ball (likely hit it)"""
    if not ball_position or not player_detections:
        return None

    ball_center = ((ball_position[0] + ball_position[2]) / 2,
                   (ball_position[1] + ball_position[3]) / 2)

    min_dist = float('inf')
    closest_player = None

    for player_id, bbox in player_detections.items():
        player_center = get_center_of_bbox(bbox)
        dist = measure_distance(ball_center, player_center)
        if dist < min_dist:
            min_dist = dist
            closest_player = player_id

    return closest_player


def find_ball_landing_spot(ball_detections, shot_frame, next_shot_frame, shooter_id):
    """Find where the ball bounces after a shot (detect direction change)"""
    if next_shot_frame is None:
        next_shot_frame = min(shot_frame + 50, len(ball_detections) - 1)

    # Collect ball positions
    positions = []
    for frame_idx in range(shot_frame + 3, next_shot_frame):
        ball_pos = ball_detections[frame_idx].get(1)
        if ball_pos:
            ball_center_x = (ball_pos[0] + ball_pos[2]) / 2
            ball_center_y = (ball_pos[1] + ball_pos[3]) / 2
            positions.append((frame_idx, ball_center_x, ball_center_y))

    if len(positions) < 5:
        return None

    # Find where vertical direction changes (bounce point)
    # Player 1 is at bottom (high Y), Player 2 is at top (low Y)
    # Player 1 hits: ball goes up (Y decreases) then down (Y increases) - bounce is at min Y
    # Player 2 hits: ball goes down (Y increases) then up (Y decreases) - bounce is at max Y

    if shooter_id == 1:
        # Player 1 (bottom) hits - ball goes toward top, bounce is where Y is minimum
        # then ball comes back down toward player 1's side
        min_y = float('inf')
        bounce_point = None
        for frame_idx, x, y in positions:
            if y < min_y:
                min_y = y
                bounce_point = (int(x), int(y), frame_idx)
    else:
        # Player 2 (top) hits - ball goes toward bottom, bounce is where Y is maximum
        max_y = -1
        bounce_point = None
        for frame_idx, x, y in positions:
            if y > max_y:
                max_y = y
                bounce_point = (int(x), int(y), frame_idx)

    return bounce_point


def draw_landing_marker(frame, landing_pos, color=(0, 255, 255)):
    """Draw a target marker at the ball landing position"""
    x, y = landing_pos[0], landing_pos[1]
    # Draw concentric circles (target marker)
    cv2.circle(frame, (x, y), 25, color, 2)
    cv2.circle(frame, (x, y), 15, color, 2)
    cv2.circle(frame, (x, y), 5, color, -1)
    # Draw crosshairs
    cv2.line(frame, (x - 30, y), (x + 30, y), color, 2)
    cv2.line(frame, (x, y - 30), (x, y + 30), color, 2)
    return frame


def classify_shot_type(pose_keypoints, player_bbox, ball_position, shot_index, player_position_y,
                       court_height, player_id=None, ball_start_height=None):
    """
    Classify tennis shot type based on pose and context.

    Player orientation:
    - Player 1 (bottom court): shows BACK to camera - screen-left is their right side
    - Player 2 (top court): shows FACE to camera - screen-right is their right side

    COCO keypoint indices:
    0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 7-8: elbows,
    9-10: wrists, 11-12: hips, 13-14: knees, 15-16: ankles

    Returns: (shot_type, confidence)
    """
    if pose_keypoints is None:
        return ("Unknown", 0.0)

    # Extract key points with confidence check
    def get_point(idx):
        if idx < len(pose_keypoints) and pose_keypoints[idx][2] > 0.3:
            return (pose_keypoints[idx][0], pose_keypoints[idx][1])
        return None

    left_shoulder = get_point(5)
    right_shoulder = get_point(6)
    left_wrist = get_point(9)
    right_wrist = get_point(10)
    left_hip = get_point(11)
    right_hip = get_point(12)

    # Calculate body center
    body_center_x = None
    if left_shoulder and right_shoulder:
        body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    elif left_hip and right_hip:
        body_center_x = (left_hip[0] + right_hip[0]) / 2

    # Get shoulder height (average of both shoulders)
    shoulder_height = None
    if left_shoulder and right_shoulder:
        shoulder_height = (left_shoulder[1] + right_shoulder[1]) / 2
    elif left_shoulder:
        shoulder_height = left_shoulder[1]
    elif right_shoulder:
        shoulder_height = right_shoulder[1]

    # Check if player is near the net (middle 30% of the court)
    court_center = court_height * 0.45
    net_zone_range = court_height * 0.15
    is_at_net = abs(player_position_y - court_center) < net_zone_range

    # Determine which arm is the hitting arm (further from body center, more extended)
    hitting_wrist = None
    hitting_side = None  # 'left' or 'right' in SCREEN coordinates

    if left_wrist and right_wrist and body_center_x:
        left_dist = abs(left_wrist[0] - body_center_x)
        right_dist = abs(right_wrist[0] - body_center_x)
        if left_dist > right_dist:
            hitting_wrist = left_wrist
            hitting_side = 'screen_left'
        else:
            hitting_wrist = right_wrist
            hitting_side = 'screen_right'
    elif left_wrist:
        hitting_wrist = left_wrist
        hitting_side = 'screen_left'
    elif right_wrist:
        hitting_wrist = right_wrist
        hitting_side = 'screen_right'

    # === FOREHAND vs BACKHAND ONLY ===
    # Simplified classification for better accuracy
    # Account for player orientation:
    # - Players 1, 2 (bottom court): show BACK to camera
    # - Players 3, 4 (top court): show FACE to camera
    #
    # For right-handed players:
    # - Player showing BACK: their right arm is on screen-LEFT side
    # - Player facing camera: their right arm is on screen-RIGHT side

    if hitting_wrist and body_center_x:
        wrist_offset = hitting_wrist[0] - body_center_x  # positive = screen-right

        # Assume right-handed players (most common)
        # Forehand: hitting with dominant (right) hand on that side
        # Backhand: reaching across body

        if player_id in [1, 2]:
            # Players 1, 2 show BACK to camera
            # Their right arm appears on screen-LEFT
            # When wrist extends to screen-LEFT (negative offset) = their forehand
            # When wrist extends to screen-RIGHT (positive offset) = their backhand
            if wrist_offset <= 0:
                return ("Forehand", 0.75)
            else:
                return ("Backhand", 0.75)
        else:
            # Players 3, 4 show FACE to camera
            # Their right arm appears on screen-RIGHT
            # When wrist extends to screen-RIGHT (positive offset) = their forehand
            # When wrist extends to screen-LEFT (negative offset) = their backhand
            if wrist_offset >= 0:
                return ("Forehand", 0.75)
            else:
                return ("Backhand", 0.75)

    # Default: can't determine - don't show label
    return ("Unknown", 0.0)


def draw_trajectory_arc(frame, start_pos, end_pos, color=(255, 165, 0), num_points=30):
    """Draw a smooth curved arc trajectory (parabolic path)"""
    if not start_pos or not end_pos:
        return frame

    x1, y1 = int(start_pos[0]), int(start_pos[1])
    x2, y2 = int(end_pos[0]), int(end_pos[1])

    # Calculate control point for quadratic bezier (arc apex)
    mid_x = (x1 + x2) / 2
    # Arc height - higher arc for longer shots
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    arc_height = min(150, distance * 0.3)  # Cap the arc height

    # Control point is above the midpoint (lower Y value)
    ctrl_y = min(y1, y2) - arc_height

    # Generate points along quadratic bezier curve
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        # Quadratic bezier formula: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        bx = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
        by = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
        points.append((int(bx), int(by)))

    # Draw the curved path
    for i in range(len(points) - 1):
        # Gradient color effect - fade along the path
        alpha = i / len(points)
        thickness = max(2, int(4 * (1 - alpha * 0.5)))
        cv2.line(frame, points[i], points[i+1], color, thickness)

    # Draw small circles along the path for dotted effect
    for i in range(0, len(points), 3):
        cv2.circle(frame, points[i], 4, color, -1)

    return frame


def main():
    # Read Video
    input_video_path = "inputs/padel_match_35s.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    # Use TrackNet for more accurate ball detection (U-Net encoder-decoder)
    ball_tracker = TrackNetBallTracker(model_path='models/weights/ball_detection/TrackNet_best.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/padel_player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=False,
                                                 stub_path="tracker_stubs/padel_tracknet_ball_detections.pkl"
                                                 )

    print(f"Detected {len(ball_detections)} ball positions")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Line Detector - using calibrated padel court detector
    # Uses pre-calibrated keypoints for consistent detection
    court_line_detector = PadelCourtDetectorColor(use_calibrated=True)
    print("Using Padel court detection with calibrated keypoints")
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0])
    
    # Detect ball shots (pass player detections to validate shots by player proximity)
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections, player_detections)
    print(f"Detected {len(ball_shot_frames)} shots at frames: {ball_shot_frames}")

    # Load pose model for shot visualization
    pose_model = YOLO('yolo11m-pose.pt')

    # Attribute each shot to a player and count shots per player
    player_shot_counts = {}  # {player_id: shot_count}
    shot_player_mapping = {}  # {frame_num: player_id}

    for shot_frame in ball_shot_frames:
        ball_pos = ball_detections[shot_frame].get(1)
        player_dets = player_detections[shot_frame]
        shooter = get_player_who_shot(ball_pos, player_dets)
        if shooter:
            player_shot_counts[shooter] = player_shot_counts.get(shooter, 0) + 1
            shot_player_mapping[shot_frame] = shooter
            print(f"  Frame {shot_frame}: Player {shooter} hit the ball (total: {player_shot_counts[shooter]})")

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections,
        ball_detections,
        court_keypoints)

    player_stats_data = [{
        'frame_num': 0,
        # Team 1 - Players 1 and 2 (near/bottom)
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,

        # Team 2 - Players 3 and 4 (far/top)
        'player_3_number_of_shots': 0,
        'player_3_total_shot_speed': 0,
        'player_3_last_shot_speed': 0,
        'player_3_total_player_speed': 0,
        'player_3_last_player_speed': 0,

        'player_4_number_of_shots': 0,
        'player_4_total_shot_speed': 0,
        'player_4_last_shot_speed': 0,
        'player_4_total_player_speed': 0,
        'player_4_last_player_speed': 0,
    }]

    # Team mappings for padel doubles
    team1_players = [1, 2]  # Near/bottom
    team2_players = [3, 4]  # Far/top

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps

        # Get distance covered by the ball
        if 1 not in ball_mini_court_detections[start_frame] or 1 not in ball_mini_court_detections[end_frame]:
            continue

        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           )

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # player who hit the ball - find closest player among all 4
        player_positions = player_mini_court_detections[start_frame]
        if not player_positions:
            continue

        ball_pos = ball_mini_court_detections[start_frame][1]
        player_shot_ball = min(player_positions.keys(),
                               key=lambda player_id: measure_distance(player_positions[player_id], ball_pos))

        # Get opponents (the other team)
        if player_shot_ball in team1_players:
            opponent_players = team2_players
        else:
            opponent_players = team1_players

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        # Calculate speed for all opponents (other team)
        for opponent_id in opponent_players:
            opponent_in_start = opponent_id in player_mini_court_detections[start_frame]
            opponent_in_end = opponent_id in player_mini_court_detections[end_frame]

            if opponent_in_start and opponent_in_end:
                distance_covered_by_opponent_pixels = measure_distance(
                    player_mini_court_detections[start_frame][opponent_id],
                    player_mini_court_detections[end_frame][opponent_id])
                distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
                    distance_covered_by_opponent_pixels,
                    constants.DOUBLE_LINE_WIDTH,
                    mini_court.get_width_of_mini_court())
                speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6
            else:
                speed_of_opponent = 0

            current_player_stats[f'player_{opponent_id}_total_player_speed'] += speed_of_opponent
            current_player_stats[f'player_{opponent_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # Calculate averages for all 4 players
    for pid in [1, 2, 3, 4]:
        player_stats_data_df[f'player_{pid}_average_shot_speed'] = (
            player_stats_data_df[f'player_{pid}_total_shot_speed'] /
            player_stats_data_df[f'player_{pid}_number_of_shots'].replace(0, 1)
        )
        # Get opponent team's total shots for player speed average
        if pid in [1, 2]:
            opponent_shots = player_stats_data_df['player_3_number_of_shots'] + player_stats_data_df['player_4_number_of_shots']
        else:
            opponent_shots = player_stats_data_df['player_1_number_of_shots'] + player_stats_data_df['player_2_number_of_shots']
        player_stats_data_df[f'player_{pid}_average_player_speed'] = (
            player_stats_data_df[f'player_{pid}_total_player_speed'] /
            opponent_shots.replace(0, 1)
        )

    # ===== CALCULATE ALL STATS PROGRESSIVELY PER FRAME =====
    # Get court boundaries
    court_top_y = mini_court.court_start_y
    court_bottom_y = mini_court.court_end_y
    court_height = court_bottom_y - court_top_y
    net_y = (court_top_y + court_bottom_y) / 2
    court_center_x = (mini_court.court_start_x + mini_court.court_end_x) / 2

    # Zone depths as percentage of HALF court (each player's side)
    half_court_height = court_height / 2
    baseline_zone_depth = half_court_height * 0.35  # 35% of half-court near baseline
    net_zone_depth = half_court_height * 0.25       # 25% of half-court near net
    movement_threshold = 5  # pixels for reaction time

    # Pre-calculate reaction times for each shot (for padel doubles)
    shot_reaction_data = {}  # {shot_frame: {'receiver_id': id, 'reaction_ms': ms}}
    for i, shot_frame in enumerate(ball_shot_frames[:-1]):
        shooter_id = shot_player_mapping.get(shot_frame)
        if shooter_id is None:
            continue

        # Get opponent team players
        if shooter_id in [1, 2]:  # Team 1 shot
            opponent_team = [3, 4]
        else:  # Team 2 shot
            opponent_team = [1, 2]

        next_shot_frame = ball_shot_frames[i + 1] if i + 1 < len(ball_shot_frames) else len(player_mini_court_detections) - 1

        # Find which opponent reacted first (closest to ball or first to move)
        for receiver_id in opponent_team:
            if receiver_id in player_mini_court_detections[shot_frame]:
                initial_pos = player_mini_court_detections[shot_frame][receiver_id]
                reaction_frame = 0
                for check_frame in range(shot_frame + 1, min(shot_frame + 30, next_shot_frame)):
                    if receiver_id in player_mini_court_detections[check_frame]:
                        current_pos = player_mini_court_detections[check_frame][receiver_id]
                        dist = measure_distance(initial_pos, current_pos)
                        if dist > movement_threshold:
                            reaction_frame = check_frame - shot_frame
                            break
                shot_reaction_data[shot_frame] = {'receiver_id': receiver_id, 'reaction_ms': (reaction_frame / 24) * 1000}
                break  # Only track first responding opponent

    # Pre-calculate tactical patterns for each shot
    shot_tactical_data = {}  # {shot_frame: {'shooter_id': id, 'is_crosscourt': bool}}
    for i, shot_frame in enumerate(ball_shot_frames[:-1]):
        shooter_id = shot_player_mapping.get(shot_frame)
        if shooter_id is None or shot_frame >= len(ball_mini_court_detections):
            continue
        if 1 not in ball_mini_court_detections[shot_frame]:
            continue
        start_pos = ball_mini_court_detections[shot_frame][1]
        next_frame = ball_shot_frames[i + 1]
        if next_frame >= len(ball_mini_court_detections) or 1 not in ball_mini_court_detections[next_frame]:
            continue
        end_pos = ball_mini_court_detections[next_frame][1]
        start_side = 'left' if start_pos[0] < court_center_x else 'right'
        end_side = 'left' if end_pos[0] < court_center_x else 'right'
        shot_tactical_data[shot_frame] = {'shooter_id': shooter_id, 'is_crosscourt': start_side != end_side}

    # Initialize progressive arrays for all 4 players
    player_distance_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_baseline_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_midcourt_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_net_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_reaction_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_fatigue_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_crosscourt_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_live_speed_arr = {pid: [] for pid in [1, 2, 3, 4]}
    player_avg_speed_arr = {pid: [] for pid in [1, 2, 3, 4]}

    # Running totals
    player_total_dist = {pid: 0 for pid in [1, 2, 3, 4]}
    player_zones = {pid: {'baseline': 0, 'midcourt': 0, 'net': 0} for pid in [1, 2, 3, 4]}
    player_reactions = {pid: [] for pid in [1, 2, 3, 4]}
    player_speeds = {pid: [] for pid in [1, 2, 3, 4]}
    player_cc = {pid: 0 for pid in [1, 2, 3, 4]}
    player_dl = {pid: 0 for pid in [1, 2, 3, 4]}

    speed_smoothing = 0.3  # Higher = more responsive to current frame

    for frame_idx in range(len(player_mini_court_detections)):
        frame_speeds = {pid: 0 for pid in [1, 2, 3, 4]}

        # Update distance and live speed for all 4 players
        if frame_idx > 0:
            for pid in [1, 2, 3, 4]:
                if pid in player_mini_court_detections[frame_idx] and pid in player_mini_court_detections[frame_idx - 1]:
                    dist = measure_distance(
                        player_mini_court_detections[frame_idx][pid],
                        player_mini_court_detections[frame_idx - 1][pid]
                    )
                    dist_m = convert_pixel_distance_to_meters(dist, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court())
                    player_total_dist[pid] += dist_m
                    frame_speeds[pid] = dist_m * 24 * 3.6  # km/h
                    player_speeds[pid].append(frame_speeds[pid])

        # Live speed with smoothing for all players
        for pid in [1, 2, 3, 4]:
            prev_speed = player_live_speed_arr[pid][-1] if player_live_speed_arr[pid] else 0
            player_live_speed_arr[pid].append(speed_smoothing * frame_speeds[pid] + (1 - speed_smoothing) * prev_speed)
            # Running average speed
            player_avg_speed_arr[pid].append(sum(player_speeds[pid]) / len(player_speeds[pid]) if player_speeds[pid] else 0)

        # Update zones - Team 1 (P1, P2) at bottom/near, Team 2 (P3, P4) at top/far
        # In padel mini court: P1,P2 have higher Y (bottom), P3,P4 have lower Y (top)
        for pid in [1, 2]:  # Team 1 - near/bottom side
            if pid in player_mini_court_detections[frame_idx]:
                pos_y = player_mini_court_detections[frame_idx][pid][1]
                if pos_y > court_bottom_y - baseline_zone_depth:
                    player_zones[pid]['baseline'] += 1
                elif pos_y >= net_y and pos_y < net_y + net_zone_depth:
                    player_zones[pid]['net'] += 1
                else:
                    player_zones[pid]['midcourt'] += 1

        for pid in [3, 4]:  # Team 2 - far/top side
            if pid in player_mini_court_detections[frame_idx]:
                pos_y = player_mini_court_detections[frame_idx][pid][1]
                if pos_y < court_top_y + baseline_zone_depth:
                    player_zones[pid]['baseline'] += 1
                elif pos_y <= net_y and pos_y > net_y - net_zone_depth:
                    player_zones[pid]['net'] += 1
                else:
                    player_zones[pid]['midcourt'] += 1

        # Check if this frame is a shot frame - update reaction times
        if frame_idx in shot_reaction_data:
            data = shot_reaction_data[frame_idx]
            receiver_id = data['receiver_id']
            if receiver_id in player_reactions:
                player_reactions[receiver_id].append(data['reaction_ms'])

        # Update tactical patterns
        if frame_idx in shot_tactical_data:
            data = shot_tactical_data[frame_idx]
            shooter_id = data['shooter_id']
            if shooter_id in player_cc:
                if data['is_crosscourt']:
                    player_cc[shooter_id] += 1
                else:
                    player_dl[shooter_id] += 1

        # Calculate current stats for all 4 players
        smoothing_alpha = 0.05

        for pid in [1, 2, 3, 4]:
            player_distance_arr[pid].append(player_total_dist[pid])

            total_zone = sum(player_zones[pid].values()) or 1
            player_baseline_arr[pid].append(player_zones[pid]['baseline'] / total_zone * 100)
            player_midcourt_arr[pid].append(player_zones[pid]['midcourt'] / total_zone * 100)
            player_net_arr[pid].append(player_zones[pid]['net'] / total_zone * 100)

            # Reaction time
            player_reaction_arr[pid].append(
                sum(player_reactions[pid]) / len(player_reactions[pid]) if player_reactions[pid] else 0
            )

            # Fatigue calculation
            speeds = player_speeds[pid]
            mid_idx = len(speeds) // 2 if len(speeds) > 10 else len(speeds)
            if mid_idx > 0 and len(speeds) > mid_idx:
                first_avg = sum(speeds[:mid_idx]) / mid_idx
                second_avg = sum(speeds[mid_idx:]) / (len(speeds) - mid_idx)
                raw_fatigue = ((first_avg - second_avg) / first_avg * 100) if first_avg > 0 else 0
                raw_fatigue = max(0, min(100, raw_fatigue))
                prev_fatigue = player_fatigue_arr[pid][-1] if player_fatigue_arr[pid] else 0
                smoothed_fatigue = smoothing_alpha * raw_fatigue + (1 - smoothing_alpha) * prev_fatigue
                player_fatigue_arr[pid].append(smoothed_fatigue)
            else:
                prev_fatigue = player_fatigue_arr[pid][-1] if player_fatigue_arr[pid] else 0
                player_fatigue_arr[pid].append(prev_fatigue)

            # Cross-court percentage
            total_shots = player_cc[pid] + player_dl[pid] or 1
            player_crosscourt_arr[pid].append(player_cc[pid] / total_shots * 100)

    # Add all progressive stats to dataframe for all 4 players
    for pid in [1, 2, 3, 4]:
        player_stats_data_df[f'player_{pid}_distance_meters'] = player_distance_arr[pid]
        player_stats_data_df[f'player_{pid}_baseline_pct'] = player_baseline_arr[pid]
        player_stats_data_df[f'player_{pid}_midcourt_pct'] = player_midcourt_arr[pid]
        player_stats_data_df[f'player_{pid}_net_pct'] = player_net_arr[pid]
        player_stats_data_df[f'player_{pid}_reaction_ms'] = player_reaction_arr[pid]
        player_stats_data_df[f'player_{pid}_fatigue_pct'] = player_fatigue_arr[pid]
        player_stats_data_df[f'player_{pid}_crosscourt_pct'] = player_crosscourt_arr[pid]
        player_stats_data_df[f'player_{pid}_last_player_speed'] = player_live_speed_arr[pid]
        player_stats_data_df[f'player_{pid}_average_player_speed'] = player_avg_speed_arr[pid]

    # Draw output
    ## Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections, ball_shot_frames)

    ## Draw court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court with Live Heatmaps (alternating between player movement and shots)
    output_video_frames = mini_court.draw_mini_court_with_live_heatmap(
        output_video_frames,
        player_mini_court_detections,
        ball_mini_court_detections,
        ball_shot_frames,
        shot_player_mapping,  # Pass player mapping for shot colors
        alternate_interval=72  # Switch every 3 seconds at 24fps
    )

    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)


    # Pre-compute ball landing spots and trajectories for each shot
    print("\nAnalyzing shot trajectories...")
    shot_data = {}  # {shot_frame: {'landing': (x,y,frame), 'start_pos': (x,y), 'shooter_id': id}}

    for i, shot_frame in enumerate(ball_shot_frames):
        next_shot = ball_shot_frames[i + 1] if i + 1 < len(ball_shot_frames) else None
        shooter_id = shot_player_mapping.get(shot_frame)

        # Find where ball lands after this shot (pass shooter_id for correct direction)
        landing = find_ball_landing_spot(ball_detections, shot_frame, next_shot, shooter_id)

        # Get ball position at shot time
        ball_pos = ball_detections[shot_frame].get(1)
        start_pos = None
        if ball_pos:
            start_pos = ((ball_pos[0] + ball_pos[2]) / 2, (ball_pos[1] + ball_pos[3]) / 2)

        shot_data[shot_frame] = {
            'landing': landing,
            'start_pos': start_pos,
            'shooter_id': shooter_id
        }

        if landing:
            print(f"  Shot at frame {shot_frame} (Player {shooter_id}): lands at ({landing[0]}, {landing[1]}) on frame {landing[2]}")

    # Create final video with slow-motion replays at shot frames
    print("\nCreating final video with slow-motion replays and visualizations...")
    final_video_frames = []
    slowmo_window = 12  # frames before and after shot to include in slow-mo
    slowmo_factor = 2   # how many times to repeat each frame (2x = 1/2 speed)
    pause_frames = 18   # extra pause at shot moment (0.75 sec at 24fps)

    # Get mapping of original player IDs to display IDs (1 and 2)
    player_ids = sorted(player_shot_counts.keys())
    player_display_map = {pid: idx+1 for idx, pid in enumerate(player_ids[:2])}

    # Track which frames are part of slow-mo sequences to avoid duplicates
    slowmo_ranges = []
    for shot_frame in ball_shot_frames:
        start = max(0, shot_frame - slowmo_window)
        end = min(len(output_video_frames), shot_frame + slowmo_window + 1)
        slowmo_ranges.append((start, end, shot_frame))

    frame_idx = 0
    while frame_idx < len(output_video_frames):
        # Check if this frame is the start of a slow-mo sequence
        slowmo_match = None
        for start, end, shot_frame in slowmo_ranges:
            if frame_idx == start:
                slowmo_match = (start, end, shot_frame)
                break

        if slowmo_match:
            start, end, shot_frame = slowmo_match
            shooter_id = shot_data[shot_frame]['shooter_id']
            landing = shot_data[shot_frame]['landing']
            start_pos = shot_data[shot_frame]['start_pos']

            # Detect pose BEFORE shot frame (during backswing) for CLASSIFICATION
            pose_frame = max(0, shot_frame - 5)
            pose_bbox = player_detections[pose_frame].get(shooter_id) if shooter_id else None
            classification_pose = None

            if pose_bbox:
                x1, y1, x2, y2 = pose_bbox
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                padding = max(bbox_width, bbox_height) * 0.5
                crop_x1 = max(0, int(x1 - padding))
                crop_y1 = max(0, int(y1 - padding))
                crop_x2 = min(video_frames[pose_frame].shape[1], int(x2 + padding))
                crop_y2 = min(video_frames[pose_frame].shape[0], int(y2 + padding))
                cropped_frame = video_frames[pose_frame][crop_y1:crop_y2, crop_x1:crop_x2].copy()

                crop_height, crop_width = cropped_frame.shape[:2]
                min_size = 300
                scale_factor = 1.0
                if crop_width < min_size or crop_height < min_size:
                    scale_factor = min_size / min(crop_width, crop_height)
                    new_width = int(crop_width * scale_factor)
                    new_height = int(crop_height * scale_factor)
                    cropped_frame = cv2.resize(cropped_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                pose_results = pose_model(cropped_frame, verbose=False)[0]
                if pose_results.keypoints is not None and len(pose_results.keypoints.data) > 0:
                    kpts = pose_results.keypoints.data[0]
                    adjusted_kpts = []
                    for idx in range(17):
                        kx, ky, conf = float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])
                        orig_x = (kx / scale_factor) + crop_x1
                        orig_y = (ky / scale_factor) + crop_y1
                        adjusted_kpts.append([orig_x, orig_y, conf])
                    classification_pose = np.array(adjusted_kpts)

            # Detect pose AT shot frame for VISUALIZATION
            shooter_bbox = player_detections[shot_frame].get(shooter_id) if shooter_id else None
            shooter_pose = None

            if shooter_bbox:
                x1, y1, x2, y2 = shooter_bbox
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                padding = max(bbox_width, bbox_height) * 0.5
                crop_x1 = max(0, int(x1 - padding))
                crop_y1 = max(0, int(y1 - padding))
                crop_x2 = min(video_frames[shot_frame].shape[1], int(x2 + padding))
                crop_y2 = min(video_frames[shot_frame].shape[0], int(y2 + padding))
                cropped_frame = video_frames[shot_frame][crop_y1:crop_y2, crop_x1:crop_x2].copy()

                crop_height, crop_width = cropped_frame.shape[:2]
                min_size = 300
                scale_factor = 1.0
                if crop_width < min_size or crop_height < min_size:
                    scale_factor = min_size / min(crop_width, crop_height)
                    new_width = int(crop_width * scale_factor)
                    new_height = int(crop_height * scale_factor)
                    cropped_frame = cv2.resize(cropped_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                display_id = player_display_map.get(shooter_id, shooter_id)
                print(f"    Player {display_id} shot at frame {shot_frame}")

                pose_results = pose_model(cropped_frame, verbose=False)[0]
                if pose_results.keypoints is not None and len(pose_results.keypoints.data) > 0:
                    kpts = pose_results.keypoints.data[0]
                    adjusted_kpts = []
                    for idx in range(17):
                        kx, ky, conf = float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])
                        orig_x = (kx / scale_factor) + crop_x1
                        orig_y = (ky / scale_factor) + crop_y1
                        adjusted_kpts.append([orig_x, orig_y, conf])
                    shooter_pose = np.array(adjusted_kpts)

            # Classify shot type
            shot_idx = ball_shot_frames.index(shot_frame)
            # Use shot_frame position (not pose_frame) for court position detection
            shot_frame_bbox = player_detections[shot_frame].get(shooter_id)
            player_y = (shot_frame_bbox[1] + shot_frame_bbox[3]) / 2 if shot_frame_bbox else 0
            frame_height = video_frames[0].shape[0]

            # Get display player ID (1 or 2 after normalization)
            display_player_id = player_display_map.get(shooter_id, shooter_id)

            shot_type, shot_confidence = classify_shot_type(
                pose_keypoints=classification_pose,  # Use earlier pose for classification
                player_bbox=shooter_bbox,
                ball_position=start_pos,
                shot_index=shot_idx,
                player_position_y=player_y,
                court_height=frame_height,
                player_id=display_player_id
            )
            shot_data[shot_frame]['shot_type'] = shot_type
            shot_data[shot_frame]['shot_confidence'] = shot_confidence
            print(f"      Shot type: {shot_type} ({shot_confidence*100:.0f}% confidence)")

            # Create slow-motion sequence
            for slowmo_frame_idx in range(start, end):
                slowmo_frame = output_video_frames[slowmo_frame_idx].copy()

                # At and after shot frame, show trajectory arc
                if slowmo_frame_idx >= shot_frame and start_pos and landing:
                    landing_pos = (landing[0], landing[1])
                    draw_trajectory_arc(slowmo_frame, start_pos, landing_pos, color=(255, 165, 0))

                # Show landing marker when ball reaches landing spot
                if landing and slowmo_frame_idx >= landing[2]:
                    draw_landing_marker(slowmo_frame, landing, color=(0, 255, 255))

                # At shot frame, add pose visualization and PAUSE
                if slowmo_frame_idx == shot_frame:
                    if shooter_pose is not None:
                        draw_pose_on_frame(slowmo_frame, shooter_pose)

                    # Draw shot type label
                    if shot_type and shot_type != "Unknown":
                        # Position label near the player
                        label_x = int(shooter_bbox[0]) if shooter_bbox else 100
                        label_y = int(shooter_bbox[1] - 60) if shooter_bbox else 100

                        # Shot type colors
                        type_colors = {
                            'Forehand': (100, 255, 100),   # Green
                            'Backhand': (100, 100, 255),   # Red
                        }
                        type_color = type_colors.get(shot_type, (255, 255, 255))

                        # Draw badge background
                        text = shot_type.upper()
                        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        badge_x1 = label_x - 5
                        badge_y1 = label_y - text_h - 10
                        badge_x2 = label_x + text_w + 10
                        badge_y2 = label_y + 5

                        # Semi-transparent background
                        overlay = slowmo_frame.copy()
                        cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (30, 30, 30), -1)
                        cv2.addWeighted(overlay, 0.7, slowmo_frame, 0.3, 0, slowmo_frame)

                        # Border and text
                        cv2.rectangle(slowmo_frame, (badge_x1, badge_y1), (badge_x2, badge_y2), type_color, 2)
                        cv2.putText(slowmo_frame, text, (label_x, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, type_color, 2, cv2.LINE_AA)

                    # PAUSE at shot frame
                    for _ in range(pause_frames):
                        final_video_frames.append(slowmo_frame.copy())

                # Repeat frame for slow-motion effect
                for _ in range(slowmo_factor):
                    final_video_frames.append(slowmo_frame.copy())

            frame_idx = end  # Skip past the slow-mo sequence
        else:
            final_video_frames.append(output_video_frames[frame_idx])
            frame_idx += 1

    print(f"Final video: {len(final_video_frames)} frames (original: {len(output_video_frames)}, added slow-mo frames)")
    save_video(final_video_frames, "output_videos/output_video.mp4")


if __name__ == "__main__":
    main()
