"""
Player Best Shots Highlights Video Generator

Creates a highlight reel showing only the player's BEST/FASTEST shots
with speed display and ball trajectory projection.
"""

import cv2
import numpy as np
import pickle
import os
from ultralytics import YOLO
import pandas as pd

from utils import read_video, save_video, measure_distance, convert_pixel_distance_to_meters
from trackers import PlayerTracker
from trackers.tracknet_ball_tracker import TrackNetBallTracker
from court_line_detector import PadelCourtDetectorColor
from mini_court import MiniCourt
import constants

# ============================================================================
# CONFIGURATION
# ============================================================================
HIGHLIGHT_PLAYER = 3  # Player to create highlights for
TOP_N_SHOTS = 3  # Number of best shots to show (fastest)

VIDEO_CONFIGS = [
    {
        'video_path': 'inputs/video_1.mp4',
        'shot_data_path': 'shot_data_video_1.csv',
        'sides_switched': False,
        'description': 'Point 1',
    },
    {
        'video_path': 'inputs/video_3.mp4',
        'shot_data_path': 'shot_data_video_3.csv',
        'sides_switched': True,
        'description': 'Point 2 - Winning Point',
        'is_scoring_point': True,
    },
]

# Player colors
PLAYER_COLORS = {
    1: (0, 255, 0),
    2: (0, 200, 100),
    3: (0, 0, 255),
    4: (0, 100, 255),
}

# Pose settings (same as original algorithm)
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
    """Draw pose skeleton on frame with optional transparency (same as original algorithm)"""
    if keypoints is None:
        return frame

    head_indices = {0, 1, 2, 3, 4}
    overlay = frame.copy()

    # Draw skeleton lines
    for joint in POSE_SKELETON:
        if joint[0] in head_indices or joint[1] in head_indices:
            continue
        pt1, pt2 = keypoints[joint[0]], keypoints[joint[1]]
        if pt1[2] > 0.5 and pt2[2] > 0.5:
            part = get_skeleton_part(joint[0])
            line_color = PART_COLORS.get(part, (150, 150, 150))
            cv2.line(overlay, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                     line_color, thickness + 2, cv2.LINE_AA)

    # Draw keypoint circles
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5 and i < len(KEYPOINT_PARTS) and i not in head_indices:
            part = KEYPOINT_PARTS[i]
            kpt_color = PART_COLORS.get(part, (100, 100, 100))
            cv2.circle(overlay, (int(x), int(y)), 5, kpt_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y)), 6, (30, 30, 30), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def load_shot_data(csv_path, fps=30):
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


def draw_ball_trajectory(frame, ball_positions, current_frame, shot_frame, color=(0, 255, 255)):
    """Draw ball trajectory with projection arc"""
    # Get ball positions around the shot
    positions = []
    for f_idx in range(max(0, shot_frame - 5), min(len(ball_positions), current_frame + 1)):
        if 1 in ball_positions[f_idx]:
            bbox = ball_positions[f_idx][1]
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            positions.append((cx, cy, f_idx))

    if len(positions) < 2:
        return frame

    # Draw trajectory trail with fading effect
    for i in range(1, len(positions)):
        alpha = i / len(positions)
        thickness = max(1, int(4 * alpha))
        pt1 = positions[i-1][:2]
        pt2 = positions[i][:2]

        # Color fades from dim to bright
        c = tuple(int(c * alpha) for c in color)
        cv2.line(frame, pt1, pt2, c, thickness, cv2.LINE_AA)

    # Draw current ball position with glow
    if positions:
        last_pos = positions[-1][:2]
        for r in range(15, 5, -3):
            alpha = (15 - r) / 10
            glow_color = tuple(int(c * alpha) for c in color)
            cv2.circle(frame, last_pos, r, glow_color, -1, cv2.LINE_AA)
        cv2.circle(frame, last_pos, 6, color, -1, cv2.LINE_AA)

    return frame


def draw_speed_indicator(frame, speed, x, y, is_best=False):
    """Draw speed indicator with styling"""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Background box
    speed_text = f"{speed:.0f} km/h"
    (tw, th), _ = cv2.getTextSize(speed_text, font, 1.2, 3)

    box_x = x - tw // 2 - 15
    box_y = y - th - 15
    box_w = tw + 30
    box_h = th + 25

    # Gradient background based on speed
    if speed > 80:
        bg_color = (0, 0, 180)  # Red for fast
    elif speed > 60:
        bg_color = (0, 140, 180)  # Orange for medium
    else:
        bg_color = (0, 180, 80)  # Green for normal

    # Draw box with glow if best shot
    if is_best:
        for i in range(3):
            cv2.rectangle(frame, (box_x - i*2, box_y - i*2),
                         (box_x + box_w + i*2, box_y + box_h + i*2),
                         (0, 200, 255), 2)

    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), bg_color, -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)

    # Speed text
    cv2.putText(frame, speed_text, (box_x + 15, box_y + th + 8),
               font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    if is_best:
        # "BEST" label above
        cv2.putText(frame, "BEST", (box_x + tw//2 - 10, box_y - 10),
                   font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    return frame


def overlay_logo(frame, logo_img, x, y, alpha=1.0):
    """Overlay logo with alpha channel support"""
    if logo_img is None:
        return frame
    h, w = logo_img.shape[:2]
    if y + h > frame.shape[0] or x + w > frame.shape[1] or y < 0 or x < 0:
        return frame
    if logo_img.shape[2] == 4:
        logo_alpha = (logo_img[:, :, 3] / 255.0) * alpha
        logo_alpha = np.dstack([logo_alpha] * 3)
        logo_rgb = logo_img[:, :, :3]
        roi = frame[y:y+h, x:x+w].astype(np.float32)
        blended = logo_rgb * logo_alpha + roi * (1 - logo_alpha)
        frame[y:y+h, x:x+w] = blended.astype(np.uint8)
    return frame


def draw_text_with_glow(frame, text, pos, font, scale, color, thickness, glow_color=None, glow_size=3):
    """Draw text with professional glow effect"""
    x, y = pos
    if glow_color:
        for i in range(glow_size, 0, -1):
            glow_alpha = 0.3 * (1 - i/glow_size)
            glow_c = tuple(int(c * glow_alpha) for c in glow_color)
            cv2.putText(frame, text, (x, y), font, scale, glow_c, thickness + i*2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def create_intro_card(frame_w, frame_h, player_id, logo, duration_frames=75):
    frames = []

    # Prepare logo
    logo_large = None
    if logo is not None:
        logo_h = 180
        aspect = logo.shape[1] / logo.shape[0]
        logo_w = int(logo_h * aspect)
        logo_large = cv2.resize(logo, (logo_w, logo_h))

    for i in range(duration_frames):
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Professional gradient background
        center_x, center_y = frame_w // 2, frame_h // 2
        Y, X = np.ogrid[:frame_h, :frame_w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        gradient = 1 - (dist / max_dist) * 0.8
        gradient = np.clip(gradient * 35, 8, 45).astype(np.uint8)
        frame[:, :, 0] = gradient
        frame[:, :, 1] = gradient
        frame[:, :, 2] = gradient

        progress = i / duration_frames
        text_alpha = min(1.0, progress * 3) if progress < 0.3 else (1.0 if progress < 0.7 else max(0, 1 - (progress - 0.7) * 3))

        bar_height = int(frame_h * 0.08)

        if text_alpha > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            player_color = PLAYER_COLORS.get(player_id, (255, 255, 255))

            # Decorative line above player text
            line_y = frame_h // 2 - 100
            line_w = 200
            line_color = tuple(int(c * text_alpha * 0.5) for c in player_color)
            cv2.line(frame, (frame_w//2 - line_w//2, line_y), (frame_w//2 + line_w//2, line_y), line_color, 2, cv2.LINE_AA)

            # Player text with glow
            player_text = f"PLAYER {player_id}"
            (pw, ph), _ = cv2.getTextSize(player_text, font, 1.8, 3)
            color = tuple(int(c * text_alpha) for c in player_color)
            draw_text_with_glow(frame, player_text, ((frame_w - pw) // 2, frame_h // 2 - 50),
                               font, 1.8, color, 3, player_color, 4)

            # Main title with glow effect
            best_text = "BEST SHOTS"
            (bw, bh), _ = cv2.getTextSize(best_text, font, 2.8, 5)
            main_color = (int(255 * text_alpha), int(255 * text_alpha), int(255 * text_alpha))
            glow_color = (int(100 * text_alpha), int(200 * text_alpha), int(255 * text_alpha))
            draw_text_with_glow(frame, best_text, ((frame_w - bw) // 2, frame_h // 2 + 40),
                               font, 2.8, main_color, 5, glow_color, 5)

            # Decorative line below
            line_y2 = frame_h // 2 + 70
            cv2.line(frame, (frame_w//2 - line_w//2, line_y2), (frame_w//2 + line_w//2, line_y2), line_color, 2, cv2.LINE_AA)

            # Subtitle
            sub_text = "Fastest & Most Powerful"
            (sw, sh), _ = cv2.getTextSize(sub_text, font, 0.7, 1)
            sub_color = (int(150 * text_alpha), int(150 * text_alpha), int(150 * text_alpha))
            cv2.putText(frame, sub_text, ((frame_w - sw) // 2, frame_h // 2 + 110),
                       font, 0.7, sub_color, 1, cv2.LINE_AA)

            # Logo at bottom
            if logo_large is not None:
                logo_x = (frame_w - logo_large.shape[1]) // 2
                logo_y = frame_h - bar_height - logo_large.shape[0] - 30
                overlay_logo(frame, logo_large, logo_x, logo_y, text_alpha * 0.9)

        cv2.rectangle(frame, (0, 0), (frame_w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, frame_h - bar_height), (frame_w, frame_h), (0, 0, 0), -1)
        frames.append(frame)

    return frames


def create_shot_title_card(frame_w, frame_h, shot_type, speed, rank, logo, is_scoring=False, duration_frames=40):
    frames = []

    # Prepare small logo
    logo_small = None
    if logo is not None:
        logo_h = 80
        aspect = logo.shape[1] / logo.shape[0]
        logo_w = int(logo_h * aspect)
        logo_small = cv2.resize(logo, (logo_w, logo_h))

    for i in range(duration_frames):
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Subtle gradient background
        Y, X = np.ogrid[:frame_h, :frame_w]
        gradient = np.clip(25 + (Y / frame_h) * 15, 20, 40).astype(np.uint8)
        frame[:, :, 0] = gradient
        frame[:, :, 1] = gradient
        frame[:, :, 2] = gradient

        progress = i / duration_frames
        text_alpha = min(1.0, progress * 4) if progress < 0.25 else 1.0

        bar_height = int(frame_h * 0.08)

        if text_alpha > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Rank badge with background
            rank_text = f"#{rank}"
            (rw, rh), _ = cv2.getTextSize(rank_text, font, 2.2, 4)
            rank_x = (frame_w - rw) // 2
            rank_y = frame_h // 2 - 70

            # Badge background
            badge_color = (int(50 * text_alpha), int(50 * text_alpha), int(60 * text_alpha))
            cv2.rectangle(frame, (rank_x - 20, rank_y - rh - 10), (rank_x + rw + 20, rank_y + 15), badge_color, -1)
            cv2.rectangle(frame, (rank_x - 20, rank_y - rh - 10), (rank_x + rw + 20, rank_y + 15),
                         (int(100 * text_alpha), int(180 * text_alpha), int(255 * text_alpha)), 2)

            rank_color = (int(100 * text_alpha), int(200 * text_alpha), int(255 * text_alpha))
            cv2.putText(frame, rank_text, (rank_x, rank_y), font, 2.2, rank_color, 4, cv2.LINE_AA)

            # Shot type with glow
            type_text = shot_type.upper()
            (tw, th), _ = cv2.getTextSize(type_text, font, 2.8, 5)
            type_color = (int(255 * text_alpha), int(255 * text_alpha), int(255 * text_alpha))
            glow_c = (int(80 * text_alpha), int(160 * text_alpha), int(255 * text_alpha))
            draw_text_with_glow(frame, type_text, ((frame_w - tw) // 2, frame_h // 2 + 20),
                               font, 2.8, type_color, 5, glow_c, 4)

            # Speed with icon-style box
            speed_text = f"{speed:.0f} km/h"
            (spw, sph), _ = cv2.getTextSize(speed_text, font, 1.4, 3)
            speed_x = (frame_w - spw) // 2
            speed_y = frame_h // 2 + 85

            # Speed background
            cv2.rectangle(frame, (speed_x - 15, speed_y - sph - 8), (speed_x + spw + 15, speed_y + 8),
                         (int(20 * text_alpha), int(80 * text_alpha), int(80 * text_alpha)), -1)
            speed_color = (int(100 * text_alpha), int(255 * text_alpha), int(255 * text_alpha))
            cv2.putText(frame, speed_text, (speed_x, speed_y), font, 1.4, speed_color, 3, cv2.LINE_AA)

            if is_scoring:
                # Winning shot banner
                score_text = "WINNING SHOT"
                (scw, sch), _ = cv2.getTextSize(score_text, font, 1.1, 2)
                score_x = (frame_w - scw) // 2
                score_y = frame_h // 2 + 145

                # Banner background
                cv2.rectangle(frame, (score_x - 25, score_y - sch - 10), (score_x + scw + 25, score_y + 10),
                             (int(30 * text_alpha), int(100 * text_alpha), int(30 * text_alpha)), -1)
                cv2.rectangle(frame, (score_x - 25, score_y - sch - 10), (score_x + scw + 25, score_y + 10),
                             (int(100 * text_alpha), int(255 * text_alpha), int(100 * text_alpha)), 2)
                cv2.putText(frame, score_text, (score_x, score_y),
                           font, 1.1, (int(100 * text_alpha), int(255 * text_alpha), int(100 * text_alpha)), 2, cv2.LINE_AA)

            # Logo at bottom right
            if logo_small is not None:
                logo_x = frame_w - logo_small.shape[1] - 40
                logo_y = frame_h - bar_height - logo_small.shape[0] - 15
                overlay_logo(frame, logo_small, logo_x, logo_y, text_alpha * 0.8)

        cv2.rectangle(frame, (0, 0), (frame_w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, frame_h - bar_height), (frame_w, frame_h), (0, 0, 0), -1)
        frames.append(frame)

    return frames


def create_shot_highlight(video_frames, player_detections, ball_detections, shot, pose_model,
                          frame_w, frame_h, player_id, speed, rank, slowmo_factor=4,
                          context_before=20, context_after=50):
    frames = []
    shot_frame = shot['frame']
    start_frame = max(0, shot_frame - context_before)
    end_frame = min(len(video_frames) - 1, shot_frame + context_after)

    # Pre-compute poses for highlighted player only
    poses = {}
    for frame_idx in range(start_frame, end_frame + 1):
        bbox = player_detections[frame_idx].get(player_id)
        if bbox:
            x1, y1, x2, y2 = bbox
            player_cx = (x1 + x2) / 2
            player_cy = (y1 + y2) / 2

            # Use tighter padding to minimize detecting other players
            padding = max(x2 - x1, y2 - y1) * 0.35
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
                    # If multiple poses detected, select the one closest to player center
                    best_pose = None
                    best_dist = float('inf')

                    for pose_data in results.keypoints.data:
                        # Convert to CPU if on CUDA
                        pose_cpu = pose_data.cpu().numpy() if hasattr(pose_data, 'cpu') else pose_data

                        # Calculate pose center using hip points (11, 12) or shoulder points (5, 6)
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
                        poses[frame_idx] = np.array(adjusted)

    replay_count = 0
    for frame_idx in range(start_frame, end_frame + 1):
        raw = video_frames[frame_idx].copy()
        dimmed = (raw * 0.35).astype(np.uint8)

        # Player spotlight mask
        mask = np.zeros((frame_h, frame_w), dtype=np.float32)
        bbox = player_detections[frame_idx].get(player_id)
        player_center = None
        if bbox:
            x1, y1, x2, y2 = bbox
            player_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            pad = 60
            bx1, by1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
            bx2, by2 = min(frame_w, int(x2 + pad)), min(frame_h, int(y2 + pad))
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            w, h = (bx2 - bx1) // 2, (by2 - by1) // 2
            Y, X = np.ogrid[:frame_h, :frame_w]
            ellipse = ((X - cx) / (w + 40))**2 + ((Y - cy) / (h + 40))**2
            mask = np.maximum(mask, np.clip(1 - ellipse, 0, 1))

        mask = cv2.GaussianBlur(mask, (61, 61), 0)
        mask_3ch = np.dstack([np.clip(mask, 0, 1)] * 3)

        highlight_frame = (raw * mask_3ch + dimmed * (1 - mask_3ch)).astype(np.uint8)
        highlight_frame = cv2.convertScaleAbs(highlight_frame, alpha=1.1, beta=5)

        # Draw ball trajectory
        highlight_frame = draw_ball_trajectory(highlight_frame, ball_detections, frame_idx, shot_frame)

        # Letterbox bars
        bar_height = int(frame_h * 0.08)
        cv2.rectangle(highlight_frame, (0, 0), (frame_w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(highlight_frame, (0, frame_h - bar_height), (frame_w, frame_h), (0, 0, 0), -1)

        # Player and shot info
        player_color = PLAYER_COLORS.get(player_id, (255, 255, 255))
        cv2.putText(highlight_frame, f"P{player_id}", (50, bar_height + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, player_color, 2, cv2.LINE_AA)
        cv2.putText(highlight_frame, shot['shot_type'].upper(), (50, bar_height + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        # Rank badge
        cv2.putText(highlight_frame, f"#{rank} BEST", (50, bar_height + 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2, cv2.LINE_AA)

        # Speed indicator near player (after shot moment)
        if frame_idx >= shot_frame and player_center:
            draw_speed_indicator(highlight_frame, speed, player_center[0], player_center[1] - 100, is_best=(rank == 1))

        # Slow motion indicator
        flash_alpha = (np.sin(replay_count * 0.3) + 1) / 2
        slowmo_color = (int(100 + 155 * flash_alpha), 200, 255)
        cv2.putText(highlight_frame, "SLOW-MO", (frame_w - 180, bar_height + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, slowmo_color, 2, cv2.LINE_AA)

        # Draw pose
        pose = poses.get(frame_idx)
        if pose is not None:
            draw_pose_on_frame(highlight_frame, pose, alpha=0.9, thickness=3)

        for _ in range(slowmo_factor):
            frames.append(highlight_frame.copy())
            replay_count += 1

    return frames


def create_stats_card(frame_w, frame_h, player_id, shots, logo, duration_frames=90):
    frames = []

    if not shots:
        return frames

    # Calculate stats
    avg_speed = sum(s['speed'] for s in shots) / len(shots)
    max_speed = max(s['speed'] for s in shots)

    # Prepare logo
    logo_med = None
    if logo is not None:
        logo_h = 120
        aspect = logo.shape[1] / logo.shape[0]
        logo_w = int(logo_h * aspect)
        logo_med = cv2.resize(logo, (logo_w, logo_h))

    for i in range(duration_frames):
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Professional gradient background
        center_x, center_y = frame_w // 2, frame_h // 2
        Y, X = np.ogrid[:frame_h, :frame_w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        gradient = 1 - (dist / max_dist) * 0.7
        gradient = np.clip(gradient * 40, 10, 45).astype(np.uint8)
        frame[:, :, 0] = gradient
        frame[:, :, 1] = gradient
        frame[:, :, 2] = gradient

        progress = i / duration_frames
        text_alpha = min(1.0, progress * 3) if progress < 0.3 else (1.0 if progress < 0.7 else max(0, 1 - (progress - 0.7) * 3))

        bar_height = int(frame_h * 0.08)

        if text_alpha > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            player_color = PLAYER_COLORS.get(player_id, (255, 255, 255))

            # Decorative line above title
            line_y = 95
            line_w = 180
            line_color = tuple(int(c * text_alpha * 0.5) for c in player_color)
            cv2.line(frame, (frame_w//2 - line_w//2, line_y), (frame_w//2 + line_w//2, line_y), line_color, 2, cv2.LINE_AA)

            # Title with glow
            title = f"PLAYER {player_id} STATS"
            (tw, th), _ = cv2.getTextSize(title, font, 1.5, 3)
            color = tuple(int(c * text_alpha) for c in player_color)
            draw_text_with_glow(frame, title, ((frame_w - tw) // 2, 140),
                               font, 1.5, color, 3, player_color, 4)

            # Decorative line below title
            line_y2 = 160
            cv2.line(frame, (frame_w//2 - line_w//2, line_y2), (frame_w//2 + line_w//2, line_y2), line_color, 2, cv2.LINE_AA)

            # Stats with styled boxes
            y_pos = 220

            # Top Speed stat box
            stat1 = f"TOP SPEED"
            speed1 = f"{max_speed:.0f} km/h"
            (s1w, s1h), _ = cv2.getTextSize(stat1, font, 0.8, 2)
            (sp1w, sp1h), _ = cv2.getTextSize(speed1, font, 1.5, 3)

            box_w = max(s1w, sp1w) + 60
            box_x = (frame_w - box_w) // 2
            box_color = (int(30 * text_alpha), int(50 * text_alpha), int(50 * text_alpha))
            border_color = (int(80 * text_alpha), int(200 * text_alpha), int(255 * text_alpha))

            cv2.rectangle(frame, (box_x, y_pos - s1h - 10), (box_x + box_w, y_pos + sp1h + 20), box_color, -1)
            cv2.rectangle(frame, (box_x, y_pos - s1h - 10), (box_x + box_w, y_pos + sp1h + 20), border_color, 2)

            cv2.putText(frame, stat1, ((frame_w - s1w) // 2, y_pos),
                       font, 0.8, (int(150 * text_alpha), int(150 * text_alpha), int(150 * text_alpha)), 2, cv2.LINE_AA)
            draw_text_with_glow(frame, speed1, ((frame_w - sp1w) // 2, y_pos + sp1h + 5),
                               font, 1.5, (int(100 * text_alpha), int(255 * text_alpha), int(255 * text_alpha)), 3,
                               (int(50 * text_alpha), int(200 * text_alpha), int(255 * text_alpha)), 3)

            # Average Speed and Shots count side by side
            y_pos += 110

            stat2 = f"AVG: {avg_speed:.0f} km/h"
            stat3 = f"SHOTS: {len(shots)}"
            (s2w, s2h), _ = cv2.getTextSize(stat2, font, 1.0, 2)
            (s3w, s3h), _ = cv2.getTextSize(stat3, font, 1.0, 2)

            gap = 40
            total_w = s2w + gap + s3w
            start_x = (frame_w - total_w) // 2

            cv2.putText(frame, stat2, (start_x, y_pos),
                       font, 1.0, (int(200 * text_alpha), int(200 * text_alpha), int(200 * text_alpha)), 2, cv2.LINE_AA)
            cv2.putText(frame, stat3, (start_x + s2w + gap, y_pos),
                       font, 1.0, (int(200 * text_alpha), int(200 * text_alpha), int(200 * text_alpha)), 2, cv2.LINE_AA)

            # Shot breakdown with styled badges
            y_pos += 60
            for idx, shot in enumerate(shots):
                shot_type = shot['shot_type'].upper()
                shot_speed = f"{shot['speed']:.0f} km/h"
                is_scoring = shot.get('is_scoring', False)

                # Badge for shot type
                (stw, sth), _ = cv2.getTextSize(shot_type, font, 0.7, 2)
                (spw, sph), _ = cv2.getTextSize(shot_speed, font, 0.7, 2)

                badge_w = stw + spw + 80
                badge_x = (frame_w - badge_w) // 2

                # Badge background
                if is_scoring:
                    bg_color = (int(30 * text_alpha), int(80 * text_alpha), int(30 * text_alpha))
                    border = (int(100 * text_alpha), int(255 * text_alpha), int(100 * text_alpha))
                else:
                    bg_color = (int(40 * text_alpha), int(40 * text_alpha), int(45 * text_alpha))
                    border = (int(80 * text_alpha), int(80 * text_alpha), int(100 * text_alpha))

                cv2.rectangle(frame, (badge_x, y_pos - sth - 5), (badge_x + badge_w, y_pos + 8), bg_color, -1)
                cv2.rectangle(frame, (badge_x, y_pos - sth - 5), (badge_x + badge_w, y_pos + 8), border, 1)

                # Rank number
                rank_text = f"#{idx+1}"
                cv2.putText(frame, rank_text, (badge_x + 10, y_pos),
                           font, 0.7, (int(100 * text_alpha), int(180 * text_alpha), int(255 * text_alpha)), 2, cv2.LINE_AA)

                # Shot type
                cv2.putText(frame, shot_type, (badge_x + 45, y_pos),
                           font, 0.7, (int(220 * text_alpha), int(220 * text_alpha), int(220 * text_alpha)), 2, cv2.LINE_AA)

                # Speed
                speed_color = (int(100 * text_alpha), int(255 * text_alpha), int(255 * text_alpha))
                cv2.putText(frame, shot_speed, (badge_x + 45 + stw + 15, y_pos),
                           font, 0.7, speed_color, 2, cv2.LINE_AA)

                if is_scoring:
                    cv2.putText(frame, "WIN", (badge_x + badge_w - 40, y_pos),
                               font, 0.5, (int(100 * text_alpha), int(255 * text_alpha), int(100 * text_alpha)), 1, cv2.LINE_AA)

                y_pos += 38

            # Logo at bottom center
            if logo_med is not None:
                logo_x = (frame_w - logo_med.shape[1]) // 2
                logo_y = frame_h - bar_height - logo_med.shape[0] - 20
                overlay_logo(frame, logo_med, logo_x, logo_y, text_alpha * 0.85)

        cv2.rectangle(frame, (0, 0), (frame_w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, frame_h - bar_height), (frame_w, frame_h), (0, 0, 0), -1)
        frames.append(frame)

    return frames


def main():
    fps = 30
    player_id = HIGHLIGHT_PLAYER

    print(f"Creating BEST SHOTS highlights for Player {player_id}")

    # Load logo
    logo = None
    logo_path = 'logo.png'
    if os.path.exists(logo_path):
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        print(f"Loaded logo: {logo_path}")
    else:
        print("Warning: logo.png not found, proceeding without logo")

    # Load pose model
    print("Loading pose model...")
    pose_model = YOLO('yolo11m-pose.pt')

    # Collect all player shots with speeds
    all_player_shots = []
    video_data_cache = []

    for video_config in VIDEO_CONFIGS:
        video_path = video_config['video_path']
        shot_data_path = video_config['shot_data_path']
        sides_switched = video_config['sides_switched']
        is_scoring = video_config.get('is_scoring_point', False)

        print(f"\nProcessing: {video_path}")

        # Read video
        video_frames = read_video(video_path)
        frame_h, frame_w = video_frames[0].shape[:2]
        print(f"  Loaded {len(video_frames)} frames")

        # Load shot data
        shot_data = load_shot_data(shot_data_path, fps)

        # Load player detections
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        stub_path = f'tracker_stubs/{video_name}_player_detections.pkl'

        if os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            print("  Loaded player detections from cache")
        else:
            player_tracker = PlayerTracker(model_path='yolov8x.pt')
            player_detections = player_tracker.detect_frames(video_frames)
            os.makedirs('tracker_stubs', exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        # Load ball detections
        ball_stub_path = f'tracker_stubs/{video_name}_ball_detections.pkl'
        if os.path.exists(ball_stub_path):
            with open(ball_stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            print("  Loaded ball detections from cache")
        else:
            ball_tracker = TrackNetBallTracker(
                model_path='models/weights/ball_detection/TrackNet_best.pt',
                device='cuda'
            )
            ball_detections = ball_tracker.detect_frames(video_frames)
            with open(ball_stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        # Interpolate ball positions
        ball_tracker_temp = TrackNetBallTracker(
            model_path='models/weights/ball_detection/TrackNet_best.pt',
            device='cuda'
        )
        ball_detections = ball_tracker_temp.interpolate_ball_positions(ball_detections)

        # Court detection for speed calculation
        court_detector = PadelCourtDetectorColor(use_calibrated=True)
        court_keypoints = court_detector.predict(video_frames[0])
        mini_court = MiniCourt(video_frames[0])

        # Convert to mini court coordinates for speed calc
        _, ball_mini_court = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints)

        # Swap player IDs if sides switched
        if sides_switched:
            print("  Swapping player IDs for switched sides...")
            swapped = []
            swap_map = {1: 3, 2: 4, 3: 1, 4: 2}
            for frame_dict in player_detections:
                new_dict = {}
                for pid, bbox in frame_dict.items():
                    new_pid = swap_map.get(pid, pid)
                    new_dict[new_pid] = bbox
                swapped.append(new_dict)
            player_detections = swapped

        # Calculate speeds for player shots
        player_shots = [s for s in shot_data if s['player_id'] == player_id]
        print(f"  Found {len(player_shots)} shots by P{player_id}")

        for i, shot in enumerate(player_shots):
            # Find next shot to calculate speed
            shot_idx = shot_data.index(shot)
            if shot_idx + 1 < len(shot_data):
                start_frame = shot['frame']
                end_frame = shot_data[shot_idx + 1]['frame']
                ball_shot_time = (end_frame - start_frame) / fps

                if 1 in ball_mini_court[start_frame] and 1 in ball_mini_court[end_frame]:
                    dist_pixels = measure_distance(ball_mini_court[start_frame][1],
                                                   ball_mini_court[end_frame][1])
                    dist_meters = convert_pixel_distance_to_meters(dist_pixels, constants.DOUBLE_LINE_WIDTH,
                                                                   mini_court.get_width_of_mini_court())
                    speed = dist_meters / ball_shot_time * 3.6
                else:
                    speed = 50  # Default if can't calculate
            else:
                speed = 60  # Default for last shot

            shot['speed'] = speed
            shot['video_idx'] = len(video_data_cache)
            shot['is_scoring'] = is_scoring and shot == player_shots[-1]
            all_player_shots.append(shot)

        video_data_cache.append({
            'frames': video_frames,
            'player_detections': player_detections,
            'ball_detections': ball_detections,
            'frame_w': frame_w,
            'frame_h': frame_h,
        })

    if not all_player_shots:
        print(f"No shots found for Player {player_id}")
        return

    # Sort by speed and get top N, but always include scoring shot
    all_player_shots.sort(key=lambda x: x['speed'], reverse=True)

    # Find scoring shot if any
    scoring_shot = None
    for shot in all_player_shots:
        if shot.get('is_scoring'):
            scoring_shot = shot
            break

    # Get top N fastest shots
    best_shots = all_player_shots[:TOP_N_SHOTS]

    # Make sure scoring shot is included
    if scoring_shot and scoring_shot not in best_shots:
        best_shots.append(scoring_shot)
        print(f"  Added scoring shot (not in top {TOP_N_SHOTS} fastest)")

    # Re-sort by speed for display order
    best_shots.sort(key=lambda x: x['speed'], reverse=True)

    print(f"\nBest shots by P{player_id} ({len(best_shots)} total):")
    for i, shot in enumerate(best_shots, 1):
        print(f"  #{i}: {shot['shot_type']} - {shot['speed']:.1f} km/h" +
              (" (WINNING SHOT)" if shot.get('is_scoring') else ""))

    frame_w = video_data_cache[0]['frame_w']
    frame_h = video_data_cache[0]['frame_h']

    # Create highlight video
    print("\nCreating highlight video...")
    final_frames = []

    # Intro card
    print("  Adding intro...")
    intro_frames = create_intro_card(frame_w, frame_h, player_id, logo, duration_frames=75)
    final_frames.extend(intro_frames)

    # Process each best shot
    for rank, shot in enumerate(best_shots, 1):
        video_idx = shot['video_idx']
        video_data = video_data_cache[video_idx]
        is_scoring = shot.get('is_scoring', False)

        print(f"  Processing #{rank}: {shot['shot_type']} - {shot['speed']:.0f} km/h" +
              (" (SCORING)" if is_scoring else ""))

        # Shot title card
        title_frames = create_shot_title_card(
            frame_w, frame_h,
            shot['shot_type'],
            shot['speed'],
            rank,
            logo,
            is_scoring=is_scoring,
            duration_frames=35
        )
        final_frames.extend(title_frames)

        # Slow motion highlight with trajectory
        highlight_frames = create_shot_highlight(
            video_data['frames'],
            video_data['player_detections'],
            video_data['ball_detections'],
            shot,
            pose_model,
            frame_w, frame_h,
            player_id,
            shot['speed'],
            rank,
            slowmo_factor=5 if rank == 1 else 4,
            context_before=25,
            context_after=60
        )
        final_frames.extend(highlight_frames)

        # Transition
        for i in range(15):
            progress = i / 15
            ease = 0.5 - 0.5 * np.cos(progress * np.pi)
            last_frame = final_frames[-1].copy()
            black = np.zeros_like(last_frame)
            final_frames.append(cv2.addWeighted(last_frame, 1 - ease * 0.7, black, ease * 0.7, 0))

    # Stats summary
    print("  Adding stats summary...")
    stats_frames = create_stats_card(frame_w, frame_h, player_id, best_shots, logo, duration_frames=90)
    final_frames.extend(stats_frames)

    # Fade out
    print("  Adding ending...")
    for i in range(30):
        progress = i / 30
        ease = 0.5 - 0.5 * np.cos(progress * np.pi)
        last_frame = final_frames[-1].copy()
        black = np.zeros_like(last_frame)
        final_frames.append(cv2.addWeighted(last_frame, 1 - ease, black, ease, 0))

    for _ in range(15):
        final_frames.append(np.zeros((frame_h, frame_w, 3), dtype=np.uint8))

    print(f"\nFinal highlights video: {len(final_frames)} frames")

    output_path = f"output_videos/player_{player_id}_best_shots.mp4"
    os.makedirs("output_videos", exist_ok=True)
    save_video(final_frames, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
