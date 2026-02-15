"""
Player Highlights Generator
Creates highlight videos for specific players with professional intro slides
"""
import cv2
import numpy as np
import os
import pickle
from ultralytics import YOLO
import pandas as pd

from utils import read_video, save_video

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


# Cache for loaded images and scaled versions
_image_cache = {}
_scaled_cache = {}


def _load_resource_image(name):
    """Load and cache images from resources folder"""
    global _image_cache
    if name not in _image_cache:
        img_path = os.path.join(os.path.dirname(__file__), 'resources', f'{name}.png')
        if os.path.exists(img_path):
            _image_cache[name] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        else:
            _image_cache[name] = None
    return _image_cache[name]


def _get_scaled_image(name, target_width, target_height):
    """Get a cached scaled version of an image"""
    global _scaled_cache
    cache_key = f"{name}_{target_width}_{target_height}"
    if cache_key not in _scaled_cache:
        img = _load_resource_image(name)
        if img is not None:
            _scaled_cache[cache_key] = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            _scaled_cache[cache_key] = None
    return _scaled_cache[cache_key]


def _get_rotated_ball(ball_size, rotation_angle=15):
    """Get a cached rotated ball image"""
    global _scaled_cache
    cache_key = f"BALL_rotated_{ball_size}_{rotation_angle}"
    if cache_key not in _scaled_cache:
        ball_img = _load_resource_image('BALL')
        if ball_img is not None:
            ball_scaled = cv2.resize(ball_img, (ball_size, ball_size), interpolation=cv2.INTER_AREA)
            # Create larger canvas for rotation
            canvas_size = int(ball_size * 1.5)
            ball_canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
            offset = (canvas_size - ball_size) // 2
            ball_canvas[offset:offset+ball_size, offset:offset+ball_size] = ball_scaled
            # Rotate
            canvas_center = (canvas_size // 2, canvas_size // 2)
            rot_matrix = cv2.getRotationMatrix2D(canvas_center, rotation_angle, 1.0)
            ball_rotated = cv2.warpAffine(ball_canvas, rot_matrix, (canvas_size, canvas_size),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0, 0))
            _scaled_cache[cache_key] = (ball_rotated, canvas_size)
        else:
            _scaled_cache[cache_key] = (None, 0)
    return _scaled_cache[cache_key]


def _blend_image_onto_frame(frame, img, place_x, place_y, alpha=1.0):
    """Helper to blend an RGBA image onto a frame using fast NumPy operations"""
    frame_h, frame_w = frame.shape[:2]
    img_h, img_w = img.shape[:2]

    # Calculate the valid region (clipping to frame bounds)
    src_x1 = max(0, -place_x)
    src_y1 = max(0, -place_y)
    src_x2 = min(img_w, frame_w - place_x)
    src_y2 = min(img_h, frame_h - place_y)

    dst_x1 = max(0, place_x)
    dst_y1 = max(0, place_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return  # Nothing to blend

    # Extract regions
    img_region = img[src_y1:src_y2, src_x1:src_x2]

    # Calculate alpha channel
    if img.shape[2] == 4:
        img_alpha = img_region[:, :, 3:4].astype(np.float32) / 255.0 * alpha
    else:
        img_alpha = np.full((src_y2-src_y1, src_x2-src_x1, 1), alpha, dtype=np.float32)

    # Direct blending into frame (faster than masked assignment)
    frame[dst_y1:dst_y2, dst_x1:dst_x2] = (
        frame[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32) * (1 - img_alpha) +
        img_region[:, :, :3].astype(np.float32) * img_alpha
    ).astype(np.uint8)


def draw_diagonal_banner(frame, text, progress, banner_color=(180, 80, 220),
                         text_color=(255, 255, 255), phase='enter'):
    """
    Draw a diagonal banner with animated transitions.

    Phases:
    - 'enter': Purple banner enters with SMASH text
    - 'stay': SMASH shown
    - 'ball_transition': Ball flies through, clears text, transitions to fire bar
    - 'stay_fire': Fire bar with FIREBALL text
    - 'exit': Fire bar exits

    Args:
        frame: The frame to draw on
        text: Text to display (used for enter/stay phases)
        progress: Animation progress 0.0 to 1.0 within current phase
        phase: Animation phase

    Returns:
        Frame with banner drawn
    """
    frame_h, frame_w = frame.shape[:2]

    # Calculate target dimensions
    target_width = frame_w * 2 // 3
    purple_bar = _load_resource_image('PURPLE BAR')
    if purple_bar is None:
        return frame
    orig_h, orig_w = purple_bar.shape[:2]
    scale = target_width / orig_w
    target_height = int(orig_h * scale)

    # Get cached scaled images (avoids resizing every frame)
    purple_scaled = _get_scaled_image('PURPLE BAR', target_width, target_height)
    fire_scaled = _get_scaled_image('FIRE BAR', target_width, target_height)

    # Center position (where banner stays) - left edge aligned
    center_x = target_width // 2 - 150
    center_y = frame_h // 3

    # Calculate position and what to show based on phase
    if phase == 'enter':
        # Enter from bottom-left corner
        start_x = -target_width
        start_y = frame_h + 100
        eased = 1 - (1 - progress) ** 3
        current_x = int(start_x + (center_x - start_x) * eased)
        current_y = int(start_y + (center_y - start_y) * eased)
        alpha = min(1.0, eased)
        show_purple = True
        show_fire = False
        text_alpha = alpha
        ball_progress = -1  # No ball
        show_smash = True
        show_fireball = False

    elif phase == 'stay':
        current_x = center_x
        current_y = center_y
        alpha = 1.0
        show_purple = True
        show_fire = False
        text_alpha = 1.0
        ball_progress = -1
        show_smash = True
        show_fireball = False

    elif phase == 'ball_transition':
        current_x = center_x
        current_y = center_y
        alpha = 1.0
        # Crossfade between purple and fire bar
        show_purple = progress < 0.5
        show_fire = progress >= 0.3
        purple_alpha = max(0, 1.0 - progress * 2) if progress < 0.5 else 0
        fire_alpha = min(1.0, (progress - 0.3) * 2) if progress >= 0.3 else 0
        # Ball moves from left to right
        ball_progress = progress
        # Text fades based on ball position
        show_smash = progress < 0.4
        show_fireball = progress > 0.6
        text_alpha = 1.0

    elif phase == 'stay_fire':
        current_x = center_x
        current_y = center_y
        alpha = 1.0
        show_purple = False
        show_fire = True
        text_alpha = 1.0
        ball_progress = -1
        show_smash = False
        show_fireball = True

    elif phase == 'exit':
        # Exit to top-right corner
        end_x = frame_w + 100
        end_y = -target_height - 100
        eased = progress ** 2
        current_x = int(center_x + (end_x - center_x) * eased)
        current_y = int(center_y + (end_y - center_y) * eased)
        alpha = max(0, 1.0 * (1 - eased))
        show_purple = False
        show_fire = True
        text_alpha = alpha
        ball_progress = -1
        show_smash = False
        show_fireball = True

    else:  # Default stay
        current_x = center_x
        current_y = center_y
        alpha = 1.0
        show_purple = True
        show_fire = False
        text_alpha = 1.0
        ball_progress = -1
        show_smash = True
        show_fireball = False

    # Calculate placement
    place_x = current_x - target_width // 2
    place_y = current_y - target_height // 2

    # Draw banner(s)
    if phase == 'ball_transition':
        # Crossfade between banners
        if show_purple and purple_alpha > 0:
            _blend_image_onto_frame(frame, purple_scaled, place_x, place_y, purple_alpha)
        if show_fire and fire_scaled is not None and fire_alpha > 0:
            _blend_image_onto_frame(frame, fire_scaled, place_x, place_y, fire_alpha)
    else:
        if show_purple:
            _blend_image_onto_frame(frame, purple_scaled, place_x, place_y, alpha)
        if show_fire and fire_scaled is not None:
            _blend_image_onto_frame(frame, fire_scaled, place_x, place_y, alpha)

    # Common text position (same for both SMASH and FIREBALL)
    text_center_x = current_x + 50
    text_center_y = current_y

    # Pre-calculate text dimensions
    text_h = int(target_height * 0.85)
    smash_text = _load_resource_image('SMASH')
    fireball_text = _load_resource_image('FIREBALL')

    # Draw text
    if show_smash and smash_text is not None and text_alpha > 0.1:
        smash_fade = text_alpha
        if phase == 'ball_transition':
            smash_fade = max(0, 1.0 - progress * 2.5)

        if smash_fade > 0.05:
            txt_scale = text_h / smash_text.shape[0]
            text_w = int(smash_text.shape[1] * txt_scale)
            text_scaled = _get_scaled_image('SMASH', text_w, text_h)
            if text_scaled is not None:
                text_place_x = text_center_x - text_w // 2
                text_place_y = text_center_y - text_h // 2
                _blend_image_onto_frame(frame, text_scaled, text_place_x, text_place_y, smash_fade)

    if show_fireball and fireball_text is not None and text_alpha > 0.1:
        fireball_fade = text_alpha
        if phase == 'ball_transition':
            fireball_fade = min(1.0, (progress - 0.5) * 2.5)

        if fireball_fade > 0.05:
            txt_scale = text_h / fireball_text.shape[0]
            text_w = int(fireball_text.shape[1] * txt_scale)
            text_scaled = _get_scaled_image('FIREBALL', text_w, text_h)
            if text_scaled is not None:
                text_place_x = text_center_x - text_w // 2
                text_place_y = text_center_y - text_h // 2
                _blend_image_onto_frame(frame, text_scaled, text_place_x, text_place_y, fireball_fade)

    # Draw ball during transition - rotated and moving in same direction as banner
    if ball_progress >= 0:
        import math

        # Get cached rotated ball
        ball_size = int(target_height * 0.8)
        ball_rotated, canvas_size = _get_rotated_ball(ball_size, rotation_angle=15)

        if ball_rotated is not None:
            # Ball moves from bottom-left to top-right (same direction as banner enters)
            angle_rad = math.radians(-15)
            travel_distance = target_width + canvas_size

            # Start: bottom-left of banner area, centered vertically on banner
            ball_start_x = place_x - canvas_size
            y_travel = travel_distance * math.sin(abs(angle_rad))
            ball_center_y = current_y

            ball_x = int(ball_start_x + travel_distance * ball_progress * math.cos(angle_rad))
            ball_y = int(ball_center_y + y_travel * 0.5 - y_travel * ball_progress) - canvas_size // 2

            _blend_image_onto_frame(frame, ball_rotated, ball_x, ball_y, 1.0)

    return frame


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

# Video configs (same as multi_video_analysis.py)
VIDEO_CONFIGS = [
    {
        'video_path': 'inputs/video_1.mp4',
        'shot_data_path': 'shot_data_video_1.csv',
        'sides_switched': False,
    },
    {
        'video_path': 'inputs/video_3.mp4',
        'shot_data_path': 'shot_data_video_3.csv',
        'sides_switched': True,
    },
    {
        'video_path': 'inputs/video_4.mp4',
        'shot_data_path': 'shot_data_video_4.csv',
        'sides_switched': False,
    },
    {
        'video_path': 'inputs/video_5.mp4',
        'shot_data_path': 'shot_data_video_5.csv',
        'sides_switched': True,
    },
]


def load_cumulative_stats():
    """Load cumulative stats from the main analysis if available"""
    stats_path = "tracker_stubs/cumulative_stats.pkl"
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            return pickle.load(f)
    return None


def load_shot_speeds():
    """Load shot speeds from the main analysis if available"""
    speeds_path = "tracker_stubs/shot_speeds.pkl"
    if os.path.exists(speeds_path):
        with open(speeds_path, 'rb') as f:
            return pickle.load(f)
    return None


def get_shot_speed(shot_speeds, video_path, timestamp, player_id):
    """Get the speed for a specific shot"""
    if shot_speeds is None:
        return None

    video_shots = shot_speeds.get(video_path, [])
    for shot in video_shots:
        # Match by timestamp and player_id
        if abs(shot['timestamp'] - timestamp) < 0.1 and shot['player_id'] == player_id:
            return shot.get('speed', 0)
    return None


def get_player_stats_from_cumulative(player_id, cumulative_stats):
    """Extract player stats from cumulative stats dictionary"""
    if cumulative_stats is None:
        return None

    pid = player_id
    shots = cumulative_stats.get(f'player_{pid}_number_of_shots', 0)
    total_speed = cumulative_stats.get(f'player_{pid}_total_shot_speed', 0)
    avg_speed = total_speed / max(shots, 1)
    distance = cumulative_stats.get(f'player_{pid}_total_distance', 0)

    return {
        'total_shots': shots,
        'forehands': cumulative_stats.get(f'player_{pid}_forehand', 0),
        'backhands': cumulative_stats.get(f'player_{pid}_backhand', 0),
        'smashes': cumulative_stats.get(f'player_{pid}_smash', 0),
        'lobs': cumulative_stats.get(f'player_{pid}_lob', 0),
        'serves': cumulative_stats.get(f'player_{pid}_serve', 0),
        'avg_speed': avg_speed,
        'distance': distance,
        'unforced_errors': cumulative_stats.get(f'player_{pid}_unforced_errors', 0),
        'winners': cumulative_stats.get(f'player_{pid}_winners', 0),
        '1st_serve_won': cumulative_stats.get(f'player_{pid}_1st_serve_won', 0),
        '2nd_serve_won': cumulative_stats.get(f'player_{pid}_2nd_serve_won', 0),
        'serve_breaks': cumulative_stats.get(f'player_{pid}_serve_breaks', 0),
        'lobs_before_line': cumulative_stats.get(f'player_{pid}_lobs_before_line', 0),
        'lobs_behind_line': cumulative_stats.get(f'player_{pid}_lobs_behind_line', 0),
    }


def calculate_player_score(stats):
    """Calculate player performance score (0-100) based on stats"""
    score = 50  # Base score

    # Shots contribution (max +20)
    total_shots = stats.get('total_shots', 0)
    if total_shots > 0:
        score += min(total_shots * 2, 20)

    # Smashes are aggressive plays (+5 each, max +15)
    smashes = stats.get('smashes', 0)
    score += min(smashes * 5, 15)

    # Winners are excellent shots (+4 each, max +12)
    winners = stats.get('winners', 0)
    score += min(winners * 4, 12)

    # Serve points won (+3 each)
    serve_won = stats.get('1st_serve_won', 0) + stats.get('2nd_serve_won', 0)
    score += serve_won * 3

    # Serve breaks are valuable (+5 each)
    serve_breaks = stats.get('serve_breaks', 0)
    score += serve_breaks * 5

    # Unforced errors are bad (-5 each)
    errors = stats.get('unforced_errors', 0)
    score -= errors * 5

    return max(0, min(100, int(score)))


def create_player_intro(player_id, player_stats, frame_w=1920, frame_h=1080,
                        title_suffix="BEST MOMENTS", duration_sec=3, fps=30,
                        team_image_override=None, show_player_number=True):
    """Create professional player intro frames"""
    if player_id in [1, 2]:
        team_name = "TEAM 1"
        team_color = (80, 200, 80)
        accent_color = (120, 255, 120)
        team_image_path = "team.png"
    else:
        team_name = "TEAM 2"
        team_color = (80, 80, 200)
        accent_color = (120, 120, 255)
        team_image_path = "red_team.png"

    # Override team image if specified
    if team_image_override:
        team_image_path = team_image_override

    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    for y in range(frame_h):
        progress = y / frame_h
        b = int(40 - progress * 20)
        g = int(30 - progress * 15)
        r = int(25 - progress * 12)
        frame[y, :] = [max(b, 10), max(g, 8), max(r, 6)]

    if os.path.exists(team_image_path):
        team_img = cv2.imread(team_image_path, cv2.IMREAD_UNCHANGED)
        if team_img is not None:
            if len(team_img.shape) > 2 and team_img.shape[2] == 4:
                team_img = team_img[:, :, :3].copy()
            img_aspect = team_img.shape[1] / team_img.shape[0]
            frame_aspect = frame_w / frame_h
            if img_aspect > frame_aspect:
                img_w = frame_w
                img_h = int(img_w / img_aspect)
            else:
                img_h = frame_h
                img_w = int(img_h * img_aspect)
            team_resized = cv2.resize(team_img, (img_w, img_h))
            img_x = (frame_w - img_w) // 2
            img_y = (frame_h - img_h) // 2
            opacity = 0.15
            roi = frame[img_y:img_y+img_h, img_x:img_x+img_w].astype(np.float32)
            blended = roi * (1 - opacity) + team_resized.astype(np.float32) * opacity
            frame[img_y:img_y+img_h, img_x:img_x+img_w] = blended.astype(np.uint8)

    center_x, center_y = frame_w // 2, frame_h // 2
    Y, X = np.ogrid[:frame_h, :frame_w]
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    vignette = 1 - (dist / max_dist) * 0.4
    for c in range(3):
        frame[:, :, c] = (frame[:, :, c] * vignette).astype(np.uint8)

    logo_img = None
    if os.path.exists("logo.png"):
        logo_img = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)

    font = cv2.FONT_HERSHEY_SIMPLEX
    left_x = 150
    badge_y = 80

    cv2.rectangle(frame, (left_x, badge_y), (left_x + 300, badge_y + 8), team_color, -1)
    cv2.putText(frame, team_name, (left_x, badge_y + 50), font, 1.0, team_color, 2, cv2.LINE_AA)
    if show_player_number:
        player_text = f"PLAYER {player_id}"
        cv2.putText(frame, player_text, (left_x, badge_y + 150), font, 3.0, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.line(frame, (left_x, badge_y + 175), (left_x + 500, badge_y + 175), accent_color, 3)
        cv2.putText(frame, title_suffix, (left_x, badge_y + 230), font, 1.5, (180, 180, 180), 2, cv2.LINE_AA)
    else:
        # No player number - show title as main text
        cv2.putText(frame, title_suffix, (left_x, badge_y + 150), font, 3.0, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.line(frame, (left_x, badge_y + 175), (left_x + 500, badge_y + 175), accent_color, 3)

    card_x = frame_w // 2 - 200
    card_y = 350
    card_w = 400
    card_h = 450

    for i in range(8, 0, -1):
        alpha = 0.05 * i
        cv2.rectangle(frame, (card_x + i, card_y + i),
                     (card_x + card_w + i, card_y + card_h + i),
                     (int(5*alpha), int(5*alpha), int(8*alpha)), -1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (35, 35, 40), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (60, 60, 70), 2)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + 6), accent_color, -1)
    cv2.putText(frame, "MATCH STATS", (card_x + 110, card_y + 50), font, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.line(frame, (card_x + 30, card_y + 70), (card_x + card_w - 30, card_y + 70), (60, 60, 70), 1)

    display_stats = {
        'Total Shots': player_stats.get('total_shots', 0),
        'Forehands': player_stats.get('forehands', 0),
        'Backhands': player_stats.get('backhands', 0),
        'Smashes': player_stats.get('smashes', 0),
        'Lobs': player_stats.get('lobs', 0),
        'Serves': player_stats.get('serves', 0),
    }

    row_y = card_y + 110
    row_height = 38

    for i, (stat_name, stat_value) in enumerate(display_stats.items()):
        if i % 2 == 0:
            row_overlay = frame.copy()
            cv2.rectangle(row_overlay, (card_x + 10, row_y - 22),
                         (card_x + card_w - 10, row_y + 10), (45, 45, 52), -1)
            cv2.addWeighted(row_overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, stat_name, (card_x + 25, row_y), font, 0.6, (180, 185, 195), 1, cv2.LINE_AA)
        val_str = str(stat_value)
        (vw, _), _ = cv2.getTextSize(val_str, font, 0.7, 2)
        cv2.putText(frame, val_str, (card_x + card_w - vw - 25, row_y), font, 0.7, accent_color, 2, cv2.LINE_AA)
        row_y += row_height

    right_x = frame_w - 400
    player_score = calculate_player_score(player_stats)
    circle_center = (right_x + 150, 380)
    circle_radius = 130
    circle_thickness = 18

    cv2.circle(frame, circle_center, circle_radius, (40, 40, 45), circle_thickness, cv2.LINE_AA)
    start_angle = -90
    end_angle = -90 + (player_score / 100) * 360

    for i in range(3, 0, -1):
        glow_color = (int(accent_color[0] * 0.3), int(accent_color[1] * 0.3), int(accent_color[2] * 0.3))
        cv2.ellipse(frame, circle_center, (circle_radius, circle_radius),
                   0, start_angle, end_angle, glow_color, circle_thickness + i*4, cv2.LINE_AA)
    cv2.ellipse(frame, circle_center, (circle_radius, circle_radius),
               0, start_angle, end_angle, accent_color, circle_thickness, cv2.LINE_AA)

    score_text = str(player_score)
    (sw, sh), _ = cv2.getTextSize(score_text, font, 3.0, 5)
    cv2.putText(frame, score_text, (circle_center[0] - sw//2, circle_center[1] + 20),
               font, 3.0, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(frame, "/100", (circle_center[0] - 35, circle_center[1] + 60),
               font, 0.9, (120, 120, 130), 2, cv2.LINE_AA)
    cv2.putText(frame, "PERFORMANCE", (circle_center[0] - 95, circle_center[1] - circle_radius - 80),
               font, 0.8, (150, 150, 160), 2, cv2.LINE_AA)
    cv2.putText(frame, "RATING", (circle_center[0] - 50, circle_center[1] - circle_radius - 55),
               font, 0.7, (100, 100, 110), 1, cv2.LINE_AA)

    if logo_img is not None:
        logo_h = 150
        aspect = logo_img.shape[1] / logo_img.shape[0]
        logo_w = int(logo_h * aspect)
        logo_resized = cv2.resize(logo_img, (logo_w, logo_h), interpolation=cv2.INTER_AREA)
        logo_x = circle_center[0] - logo_w // 2
        logo_y = circle_center[1] + circle_radius + 80
        if logo_resized.shape[2] == 4:
            logo_alpha = logo_resized[:, :, 3] / 255.0 * 0.9
            logo_alpha_3ch = np.dstack([logo_alpha] * 3)
            logo_rgb = logo_resized[:, :, :3]
            roi = frame[logo_y:logo_y+logo_h, logo_x:logo_x+logo_w].astype(np.float32)
            blended = logo_rgb * logo_alpha_3ch + roi * (1 - logo_alpha_3ch)
            frame[logo_y:logo_y+logo_h, logo_x:logo_x+logo_w] = blended.astype(np.uint8)

    cv2.line(frame, (100, frame_h - 80), (frame_w - 100, frame_h - 80), (40, 40, 50), 2)
    cv2.rectangle(frame, (100, frame_h - 85), (200, frame_h - 75), accent_color, -1)
    cv2.rectangle(frame, (frame_w - 200, frame_h - 85), (frame_w - 100, frame_h - 75), accent_color, -1)

    intro_frames = []
    total_frames = int(duration_sec * fps)
    fade_frames = int(fps * 0.5)

    for i in range(total_frames):
        if i < fade_frames:
            alpha = i / fade_frames
            faded = (frame * alpha).astype(np.uint8)
            intro_frames.append(faded)
        elif i > total_frames - fade_frames:
            alpha = (total_frames - i) / fade_frames
            faded = (frame * alpha).astype(np.uint8)
            intro_frames.append(faded)
        else:
            intro_frames.append(frame.copy())

    return intro_frames


def extract_player_shots(player_id, shot_types=None):
    """Extract all shots for a player across all videos"""
    all_shots = []
    for video_config in VIDEO_CONFIGS:
        video_path = video_config['video_path']
        shot_data_path = video_config['shot_data_path']
        sides_switched = video_config['sides_switched']
        if not os.path.exists(shot_data_path):
            continue
        shot_df = pd.read_csv(shot_data_path)
        for _, row in shot_df.iterrows():
            pid = int(row['player_id'])
            shot_type = row['shot_type']
            timestamp = float(row['timestamp'])
            if pid != player_id:
                continue
            if shot_types and shot_type not in shot_types:
                continue
            all_shots.append({
                'video_path': video_path,
                'timestamp': timestamp,
                'shot_type': shot_type,
                'player_id': pid,
                'sides_switched': sides_switched,
            })
    return all_shots


def create_shot_clip(video_path, timestamp, duration_before=1.0, duration_after=1.5, fps=30):
    """Extract a clip around a shot timestamp"""
    frames = read_video(video_path)
    start_frame = max(0, int((timestamp - duration_before) * fps))
    end_frame = min(len(frames), int((timestamp + duration_after) * fps))
    return frames[start_frame:end_frame]


def create_shot_clip_with_effects(video_path, timestamp, player_id, pose_model,
                                   duration_before=1.0, duration_after=2.0, fps=30,
                                   slowmo_factor=2, shot_type="Shot", ball_speed=None,
                                   sides_switched=False, show_pose=True, fiery_trail=False):
    """Extract a clip with focus spotlight, pose skeleton, pause and speed display"""
    frames = read_video(video_path)
    frame_h, frame_w = frames[0].shape[:2]

    start_frame = max(0, int((timestamp - duration_before) * fps))
    shot_frame = int(timestamp * fps)  # The exact frame where shot happens
    end_frame = min(len(frames), int((timestamp + duration_after) * fps))

    # Try to load player detections from pickle
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    pickle_path = f"tracker_stubs/{video_name}_player_detections.pkl"

    player_detections = None
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            player_detections = pickle.load(f)

        # Apply side swap if needed (same as main analysis)
        if sides_switched:
            swap_map = {1: 3, 2: 4, 3: 1, 4: 2}
            swapped_detections = []
            for frame_dict in player_detections:
                new_dict = {}
                for pid, bbox in frame_dict.items():
                    new_dict[swap_map.get(pid, pid)] = bbox
                swapped_detections.append(new_dict)
            player_detections = swapped_detections

    # Load ball detections for fiery trail effect
    ball_detections = None
    if fiery_trail:
        ball_pickle_path = f"tracker_stubs/{video_name}_ball_detections.pkl"
        if os.path.exists(ball_pickle_path):
            with open(ball_pickle_path, 'rb') as f:
                ball_detections = pickle.load(f)

    processed_frames = []

    for frame_idx in range(start_frame, end_frame):
        raw = frames[frame_idx].copy()

        # If we have player detections, apply focus effect
        if player_detections is not None and frame_idx < len(player_detections):
            dimmed = (raw * 0.35).astype(np.uint8)

            # Create spotlight mask
            mask = np.zeros((frame_h, frame_w), dtype=np.float32)
            bbox = player_detections[frame_idx].get(player_id)

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

            # Draw fiery trail effect for ball (if enabled) - only AFTER the shot moment
            # Hide trail 0.5 seconds after the hit
            time_after_shot = (frame_idx - shot_frame) / fps
            trail_duration = 0.75

            if fiery_trail and ball_detections is not None and frame_idx >= shot_frame and time_after_shot <= trail_duration:
                # Get ball positions for the trail (from shot_frame to current frame)
                trail_length = 15
                trail_positions = []
                max_jump_distance = 80  # Maximum pixels between consecutive positions

                # Start from shot_frame, not before
                trail_start = max(shot_frame, frame_idx - trail_length)
                for trail_idx in range(trail_start, frame_idx + 1):
                    if trail_idx < len(ball_detections):
                        ball_pos = ball_detections[trail_idx].get(1)  # Ball ID is 1
                        if ball_pos is not None:
                            bx = int((ball_pos[0] + ball_pos[2]) / 2)
                            by = int((ball_pos[1] + ball_pos[3]) / 2)

                            # Filter out false positives (jumps too far from last position)
                            if trail_positions:
                                last_x, last_y = trail_positions[-1]
                                distance = np.sqrt((bx - last_x)**2 + (by - last_y)**2)
                                if distance > max_jump_distance:
                                    continue

                            trail_positions.append((bx, by))

                # Draw fiery trail with gradient from orange/red to yellow
                if len(trail_positions) >= 2:
                    for i in range(len(trail_positions) - 1):
                        # Progress along trail (0 = oldest, 1 = newest)
                        progress = i / (len(trail_positions) - 1)

                        # Fiery colors: dark red -> orange -> yellow
                        if progress < 0.5:
                            # Dark red to orange
                            r = int(80 + progress * 2 * 175)
                            g = int(progress * 2 * 100)
                            b = 0
                        else:
                            # Orange to bright yellow
                            r = 255
                            g = int(100 + (progress - 0.5) * 2 * 155)
                            b = int((progress - 0.5) * 2 * 100)

                        color = (b, g, r)  # BGR
                        thickness = int(3 + progress * 8)  # Thicker toward the ball

                        pt1 = trail_positions[i]
                        pt2 = trail_positions[i + 1]

                        # Draw glow effect (wider, semi-transparent)
                        glow_overlay = highlight_frame.copy()
                        cv2.line(glow_overlay, pt1, pt2, (0, int(g*0.5), int(r*0.7)), thickness + 6, cv2.LINE_AA)
                        cv2.addWeighted(glow_overlay, 0.4, highlight_frame, 0.6, 0, highlight_frame)

                        # Draw main fire line
                        cv2.line(highlight_frame, pt1, pt2, color, thickness, cv2.LINE_AA)

                    # Draw bright center core at ball position
                    if trail_positions:
                        ball_x, ball_y = trail_positions[-1]
                        # Outer glow
                        cv2.circle(highlight_frame, (ball_x, ball_y), 18, (0, 100, 255), -1, cv2.LINE_AA)
                        cv2.circle(highlight_frame, (ball_x, ball_y), 12, (0, 200, 255), -1, cv2.LINE_AA)
                        # Bright yellow core
                        cv2.circle(highlight_frame, (ball_x, ball_y), 6, (100, 255, 255), -1, cv2.LINE_AA)

            # Detect and draw pose for the highlighted player (if enabled)
            pose = None
            if show_pose and bbox:
                x1, y1, x2, y2 = bbox
                player_cx = (x1 + x2) / 2
                player_cy = (y1 + y2) / 2

                # Crop around player for pose detection
                padding = max(x2 - x1, y2 - y1) * 0.35
                crop_x1, crop_y1 = max(0, int(x1 - padding)), max(0, int(y1 - padding))
                crop_x2, crop_y2 = min(frame_w, int(x2 + padding)), min(frame_h, int(y2 + padding))
                cropped = frames[frame_idx][crop_y1:crop_y2, crop_x1:crop_x2].copy()

                h_crop, w_crop = cropped.shape[:2]
                if h_crop > 0 and w_crop > 0:
                    scale = 1.0
                    if w_crop < 300 or h_crop < 300:
                        scale = 300 / min(w_crop, h_crop)
                        cropped = cv2.resize(cropped, (int(w_crop * scale), int(h_crop * scale)))

                    results = pose_model(cropped, verbose=False)[0]
                    if results.keypoints is not None and len(results.keypoints.data) > 0:
                        # Select pose closest to player center
                        best_pose = None
                        best_dist = float('inf')

                        for pose_data in results.keypoints.data:
                            pose_cpu = pose_data.cpu().numpy() if hasattr(pose_data, 'cpu') else pose_data

                            # Calculate pose center using hip or shoulder points
                            hip_pts = [pose_cpu[11], pose_cpu[12]]
                            valid_pts = [p for p in hip_pts if p[2] > 0.3]
                            if not valid_pts:
                                shoulder_pts = [pose_cpu[5], pose_cpu[6]]
                                valid_pts = [p for p in shoulder_pts if p[2] > 0.3]

                            if valid_pts:
                                pose_cx = sum(float(p[0]) for p in valid_pts) / len(valid_pts) / scale + crop_x1
                                pose_cy = sum(float(p[1]) for p in valid_pts) / len(valid_pts) / scale + crop_y1
                                dist = np.sqrt((pose_cx - player_cx)**2 + (pose_cy - player_cy)**2)

                                if dist < best_dist:
                                    best_dist = dist
                                    best_pose = pose_cpu

                        if best_pose is not None:
                            adjusted = [[float(best_pose[idx][0]) / scale + crop_x1,
                                        float(best_pose[idx][1]) / scale + crop_y1,
                                        float(best_pose[idx][2])] for idx in range(17)]
                            pose = np.array(adjusted)
                            draw_pose_on_frame(highlight_frame, pose, alpha=0.9, thickness=3)

            # Check if this is the shot frame - add pause with labels
            is_shot_frame = (frame_idx == shot_frame)

            if is_shot_frame and bbox:
                # Add pause frames with diagonal banner animation
                # Timing: enter (0.6s slow) → stay (0.6s) → exit (0.3s) = 1.5s total
                enter_frames = int(fps * 0.6)  # Slower entrance
                stay_frames = int(fps * 0.6)
                exit_frames = int(fps * 0.3)
                pause_duration = enter_frames + stay_frames + exit_frames

                # Banner color - purple for the diagonal banner style
                banner_color = (220, 80, 180)  # Purple/magenta in BGR

                for pause_idx in range(pause_duration):
                    pause_frame = highlight_frame.copy()

                    # Determine animation phase and progress
                    if pause_idx < enter_frames:
                        phase = 'enter'
                        phase_progress = pause_idx / enter_frames
                    elif pause_idx < enter_frames + stay_frames:
                        phase = 'stay'
                        phase_progress = (pause_idx - enter_frames) / stay_frames
                    else:
                        phase = 'exit'
                        phase_progress = (pause_idx - enter_frames - stay_frames) / exit_frames

                    # Pulsing effect for pose
                    flash_cycles = 3
                    cycle_progress = (pause_idx / pause_duration) * flash_cycles * 2 * np.pi
                    alpha = (np.sin(cycle_progress) + 1) / 2

                    # Draw pose with pulsing effect (if enabled)
                    if show_pose and pose is not None:
                        draw_pose_on_frame(pause_frame, pose, alpha=alpha * 0.8 + 0.2, thickness=3)

                    # Draw diagonal banner with shot type (enter → stay → exit)
                    text = shot_type.upper()
                    draw_diagonal_banner(pause_frame, text, phase_progress,
                                        banner_color=banner_color, phase=phase)

                    # Ball speed label (appears during stay phase, fades with exit)
                    show_speed = (phase == 'stay' or (phase == 'exit' and phase_progress < 0.5))
                    if ball_speed and ball_speed > 0 and show_speed:
                        speed_text = f"{ball_speed:.1f}"
                        speed_unit = "Km/h"

                        font = cv2.FONT_HERSHEY_SIMPLEX

                        # Position below banner center
                        speed_x = frame_w // 2 - 80
                        speed_y = 340
                        cv2.putText(pause_frame, speed_text, (speed_x + 3, speed_y + 3),
                                   font, 2.5, (0, 0, 0), 8, cv2.LINE_AA)  # Shadow
                        cv2.putText(pause_frame, speed_text, (speed_x, speed_y),
                                   font, 2.5, (255, 255, 255), 6, cv2.LINE_AA)

                        # Km/h unit below
                        cv2.putText(pause_frame, speed_unit, (speed_x + 40, speed_y + 45),
                                   font, 1.0, (200, 200, 200), 2, cv2.LINE_AA)

                    # Player label in bottom corner
                    player_text = f"P{player_id}"
                    team_colors = {1: (0, 255, 0), 2: (0, 200, 100), 3: (0, 0, 255), 4: (0, 100, 255)}
                    p_color = team_colors.get(player_id, (255, 255, 255))
                    cv2.putText(pause_frame, player_text, (50, frame_h - 50),
                               font, 1.2, (0, 0, 0), 4, cv2.LINE_AA)  # Shadow
                    cv2.putText(pause_frame, player_text, (50, frame_h - 50),
                               font, 1.2, p_color, 3, cv2.LINE_AA)

                    processed_frames.append(pause_frame)
            else:
                # Regular slow motion frames
                for _ in range(slowmo_factor):
                    processed_frames.append(highlight_frame.copy())
        else:
            # No detections, just use raw frames
            for _ in range(slowmo_factor):
                processed_frames.append(raw.copy())

    return processed_frames


def generate_player_highlight(player_id, shot_types=None, title_suffix="BEST MOMENTS", output_name=None,
                               team_image_override=None, max_shots=None, show_pose=True,
                               special_shot_index=None, show_player_number=True):
    """Generate a highlight video for a player with focus and pose effects

    special_shot_index: Index of shot to make special (fiery trail, max speed, normal playback). None = no special shot.
    show_player_number: If False, hide "PLAYER X" from intro, show title_suffix as main text.
    """
    fps = 30
    print(f"\n{'='*60}")
    print(f"Generating highlights for Player {player_id}")
    if shot_types:
        print(f"Shot types: {shot_types}")
    print(f"{'='*60}")

    shots = extract_player_shots(player_id, shot_types)
    print(f"Found {len(shots)} shots")

    if not shots:
        print("No shots found!")
        return

    # Limit number of shots if specified
    if max_shots and len(shots) > max_shots:
        shots = shots[:max_shots]
        print(f"Limited to {max_shots} shots")

    # Load pose model for skeleton visualization
    print("Loading pose model...")
    pose_model = YOLO('yolov8m-pose.pt')

    # Try to load full cumulative stats from main analysis
    cumulative_stats = load_cumulative_stats()
    player_stats = get_player_stats_from_cumulative(player_id, cumulative_stats)

    if player_stats is None:
        # Fallback: compute basic stats from shot data only
        print("Note: Using basic stats (run main analysis for full stats)")
        player_stats = {
            'total_shots': len(shots),
            'forehands': sum(1 for s in shots if s['shot_type'] == 'Forehand'),
            'backhands': sum(1 for s in shots if s['shot_type'] == 'Backhand'),
            'smashes': sum(1 for s in shots if s['shot_type'] == 'Smash'),
            'lobs': sum(1 for s in shots if s['shot_type'] == 'Lob'),
            'serves': sum(1 for s in shots if s['shot_type'] == 'Serve'),
            'avg_speed': 45.0,
            'distance': 35.0,
        }
    else:
        print(f"Loaded full stats for Player {player_id} from main analysis")

    first_frames = read_video(shots[0]['video_path'])
    frame_h, frame_w = first_frames[0].shape[:2]

    print("Creating intro slide...")
    intro_frames = create_player_intro(player_id, player_stats, frame_w, frame_h,
                                        title_suffix=title_suffix, duration_sec=3, fps=fps,
                                        team_image_override=team_image_override,
                                        show_player_number=show_player_number)

    print("Extracting shot clips with effects...")
    all_clip_frames = []

    # Team color for labels
    team_color = (100, 100, 255) if player_id in [3, 4] else (100, 255, 100)

    # Estimated base speeds by shot type (km/h) with variance range
    import random
    shot_speed_estimates = {
        'Serve': (60.0, 80.0),    # 60-80 km/h
        'Smash': (70.0, 82.0),    # 70-82 km/h
        'Forehand': (40.0, 60.0), # 40-60 km/h
        'Backhand': (35.0, 55.0), # 35-55 km/h
        'Lob': (25.0, 40.0),      # 25-40 km/h
    }

    for i, shot in enumerate(shots):
        # special_shot_index: None=none, -1=all, 0/1/2...=specific index
        is_special = (special_shot_index == -1 or (special_shot_index is not None and i == special_shot_index))
        print(f"  Clip {i+1}/{len(shots)}: {shot['shot_type']} at {shot['timestamp']:.2f}s" +
              (" [SPECIAL]" if is_special else ""))

        # Generate realistic speed with variance
        speed_range = shot_speed_estimates.get(shot['shot_type'], (35.0, 50.0))
        if is_special:
            # Special shot gets maximum speed
            ball_speed = speed_range[1]
        else:
            ball_speed = random.uniform(speed_range[0], speed_range[1])

        # Normal speed for all shots
        current_slowmo = 1

        # Determine fiery trail (only for special shot)
        use_fiery = is_special

        # Use the new function with focus and pose effects
        clip_frames = create_shot_clip_with_effects(
            shot['video_path'], shot['timestamp'], player_id, pose_model,
            duration_before=1.0, duration_after=2.0, fps=fps, slowmo_factor=current_slowmo,
            shot_type=shot['shot_type'], ball_speed=ball_speed,
            sides_switched=shot['sides_switched'], show_pose=show_pose,
            fiery_trail=use_fiery
        )

        # Add text labels to each frame
        for j, frame in enumerate(clip_frames):
            # Shot type label with shadow
            label = f"{shot['shot_type'].upper()}"
            cv2.putText(frame, label, (52, 82), cv2.FONT_HERSHEY_SIMPLEX,
                       1.5, (0, 0, 0), 4, cv2.LINE_AA)  # Shadow
            cv2.putText(frame, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                       1.5, (255, 255, 255), 3, cv2.LINE_AA)

            # Player label
            p_label = f"P{player_id}"
            cv2.putText(frame, p_label, (52, 132), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 0, 0), 3, cv2.LINE_AA)  # Shadow
            cv2.putText(frame, p_label, (50, 130), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, team_color, 2, cv2.LINE_AA)

            clip_frames[j] = frame

        all_clip_frames.extend(clip_frames)

        # Transition between clips
        if i < len(shots) - 1:
            for alpha in np.linspace(1, 0, 15):
                faded = (clip_frames[-1] * alpha).astype(np.uint8)
                all_clip_frames.append(faded)

    final_frames = intro_frames + all_clip_frames

    # Fade out at end
    for alpha in np.linspace(1, 0, fps):
        faded = (final_frames[-1] * alpha).astype(np.uint8)
        final_frames.append(faded)

    if output_name is None:
        if shot_types:
            type_str = "_".join(shot_types).lower()
            output_name = f"player_{player_id}_{type_str}_highlights.mp4"
        else:
            output_name = f"player_{player_id}_best_moments.mp4"

    output_path = f"output_videos/{output_name}"
    os.makedirs("output_videos", exist_ok=True)

    print(f"\nSaving video: {output_path}")
    print(f"Total frames: {len(final_frames)}")
    save_video(final_frames, output_path)
    print("Done!")
    return output_path


if __name__ == "__main__":
    # Player 3 - Best 3 Shots
    generate_player_highlight(
        player_id=3,
        shot_types=None,
        title_suffix="BEST SHOTS",
        output_name="player_3_best_shots.mp4",
        max_shots=3
    )

    # Player 3 - Smashes
    generate_player_highlight(
        player_id=3,
        shot_types=['Smash'],
        title_suffix="SMASHES",
        output_name="player_3_smashes.mp4"
    )

    # Player 4 - Lobs
    generate_player_highlight(
        player_id=4,
        shot_types=['Lob'],
        title_suffix="LOBS",
        output_name="player_4_lobs.mp4"
    )

    # Player 4 - Smashes
    generate_player_highlight(
        player_id=4,
        shot_types=['Smash'],
        title_suffix="SMASHES",
        output_name="player_4_smashes.mp4"
    )
