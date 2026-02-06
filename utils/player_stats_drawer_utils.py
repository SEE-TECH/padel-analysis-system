import numpy as np
import cv2
import math
import os

# Load logo once at module level
_logo_cache = None

# Track shot changes for fade effect
_prev_shot_counts = {}  # {(player_id, shot_type): count}
_shot_change_frame = {}  # {(player_id, shot_type): frame_index when count changed}
FADE_DURATION = 30  # frames for fade effect (~1 sec at 30fps)

def get_logo():
    global _logo_cache
    if _logo_cache is None:
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logo.png')
        if os.path.exists(logo_path):
            _logo_cache = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    return _logo_cache


def format_val(val, decimals=1, is_int=False):
    """Format a value, returning '-' for NaN/None values"""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "-"
    try:
        if is_int:
            return str(int(val))
        return f"{val:.{decimals}f}"
    except (ValueError, TypeError):
        return "-"


def draw_player_stats(output_video_frames, player_stats):
    """
    Draw team-based stats for padel (2 teams of 2 players each)
    Team 1 (bottom): Players 1, 2 - Green
    Team 2 (top): Players 3, 4 - Red
    """
    # Load logo - smaller size
    logo = get_logo()
    logo_height = 0
    logo_width = 0
    logo_resized = None

    if logo is not None:
        target_logo_width = 150  # Compact logo
        scale = target_logo_width / logo.shape[1]
        logo_width = target_logo_width
        logo_height = int(logo.shape[0] * scale)
        logo_resized = cv2.resize(logo, (logo_width, logo_height))

    for index, row in player_stats.iterrows():
        # Team 1 stats (Players 1 & 2)
        team1_p1_shots = int(row.get('player_1_number_of_shots', 0) or 0)
        team1_p2_shots = int(row.get('player_2_number_of_shots', 0) or 0)
        team1_total_shots = team1_p1_shots + team1_p2_shots

        team1_p1_speed = row.get('player_1_last_player_speed', 0) or 0
        team1_p2_speed = row.get('player_2_last_player_speed', 0) or 0

        team1_p1_distance = row.get('player_1_distance_meters', 0) or 0
        team1_p2_distance = row.get('player_2_distance_meters', 0) or 0
        team1_total_distance = team1_p1_distance + team1_p2_distance

        # Team 2 stats (Players 3 & 4)
        team2_p3_shots = int(row.get('player_3_number_of_shots', 0) or 0)
        team2_p4_shots = int(row.get('player_4_number_of_shots', 0) or 0)
        team2_total_shots = team2_p3_shots + team2_p4_shots

        team2_p3_speed = row.get('player_3_last_player_speed', 0) or 0
        team2_p4_speed = row.get('player_4_last_player_speed', 0) or 0

        team2_p3_distance = row.get('player_3_distance_meters', 0) or 0
        team2_p4_distance = row.get('player_4_distance_meters', 0) or 0
        team2_total_distance = team2_p3_distance + team2_p4_distance

        # Shot speeds (ball speed)
        team1_p1_shot_speed = row.get('player_1_last_shot_speed', 0) or 0
        team1_p2_shot_speed = row.get('player_2_last_shot_speed', 0) or 0
        team2_p3_shot_speed = row.get('player_3_last_shot_speed', 0) or 0
        team2_p4_shot_speed = row.get('player_4_last_shot_speed', 0) or 0

        # Average shot speeds
        team1_p1_avg_shot = row.get('player_1_average_shot_speed', 0) or 0
        team1_p2_avg_shot = row.get('player_2_average_shot_speed', 0) or 0
        team2_p3_avg_shot = row.get('player_3_average_shot_speed', 0) or 0
        team2_p4_avg_shot = row.get('player_4_average_shot_speed', 0) or 0

        # Average player speeds
        team1_p1_avg_speed = row.get('player_1_average_player_speed', 0) or 0
        team1_p2_avg_speed = row.get('player_2_average_player_speed', 0) or 0
        team2_p3_avg_speed = row.get('player_3_average_player_speed', 0) or 0
        team2_p4_avg_speed = row.get('player_4_average_player_speed', 0) or 0

        # Top speeds
        team1_p1_top_speed = row.get('player_1_top_speed', 0) or 0
        team1_p2_top_speed = row.get('player_2_top_speed', 0) or 0
        team2_p3_top_speed = row.get('player_3_top_speed', 0) or 0
        team2_p4_top_speed = row.get('player_4_top_speed', 0) or 0

        # Court coverage percentages
        team1_p1_net = row.get('player_1_net_pct', 0) or 0
        team1_p2_net = row.get('player_2_net_pct', 0) or 0
        team2_p3_net = row.get('player_3_net_pct', 0) or 0
        team2_p4_net = row.get('player_4_net_pct', 0) or 0

        team1_p1_baseline = row.get('player_1_baseline_pct', 0) or 0
        team1_p2_baseline = row.get('player_2_baseline_pct', 0) or 0
        team2_p3_baseline = row.get('player_3_baseline_pct', 0) or 0
        team2_p4_baseline = row.get('player_4_baseline_pct', 0) or 0

        team1_p1_midcourt = row.get('player_1_midcourt_pct', 0) or 0
        team1_p2_midcourt = row.get('player_2_midcourt_pct', 0) or 0
        team2_p3_midcourt = row.get('player_3_midcourt_pct', 0) or 0
        team2_p4_midcourt = row.get('player_4_midcourt_pct', 0) or 0

        # Rally stats
        rally_length = row.get('current_rally_length', 0) or 0

        # Shot types per player
        shot_types_data = {}
        for pid in [1, 2, 3, 4]:
            shot_types_data[pid] = {
                'F': int(row.get(f'player_{pid}_forehand', 0) or 0),
                'B': int(row.get(f'player_{pid}_backhand', 0) or 0),
                'L': int(row.get(f'player_{pid}_lob', 0) or 0),
                'S': int(row.get(f'player_{pid}_smash', 0) or 0),
            }

        # Best shots (fastest per type)
        best_shots = {}
        for st in ['forehand', 'backhand', 'lob', 'smash']:
            speed = row.get(f'best_{st}_speed', 0) or 0
            player = int(row.get(f'best_{st}_player', 0) or 0)
            best_shots[st] = {'speed': speed, 'player': player}

        frame = output_video_frames[index]

        width = 440
        height = 970  # With larger fonts

        start_x = 40
        logo_area_height = logo_height + 10 if logo_resized is not None else 0
        start_y = 20 + logo_area_height
        end_x = start_x + width
        end_y = start_y + height

        # Draw logo with semi-transparent gray background above the table
        if logo_resized is not None:
            logo_bg_y1 = 20
            logo_bg_y2 = 20 + logo_height + 10
            logo_bg_x1 = start_x
            logo_bg_x2 = end_x

            overlay = frame.copy()
            cv2.rectangle(overlay, (logo_bg_x1, logo_bg_y1), (logo_bg_x2, logo_bg_y2), (60, 60, 60), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (logo_bg_x1, logo_bg_y1), (logo_bg_x2, logo_bg_y2), (80, 80, 80), 2)

            logo_x = start_x + (width - logo_width) // 2
            logo_y = 20 + 5

            if logo_resized.shape[2] == 4:
                for c in range(3):
                    alpha = logo_resized[:, :, 3] / 255.0
                    frame[logo_y:logo_y+logo_height, logo_x:logo_x+logo_width, c] = \
                        alpha * logo_resized[:, :, c] + (1 - alpha) * frame[logo_y:logo_y+logo_height, logo_x:logo_x+logo_width, c]
            else:
                frame[logo_y:logo_y+logo_height, logo_x:logo_x+logo_width] = logo_resized

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (25, 25, 25), -1)
        alpha = 0.9
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (80, 80, 80), 2)

        # Title header
        cv2.rectangle(frame, (start_x, start_y), (end_x, start_y + 45), (45, 45, 45), -1)
        cv2.line(frame, (start_x, start_y + 45), (end_x, start_y + 45), (0, 200, 255), 3)

        output_video_frames[index] = frame

        cv2.putText(output_video_frames[index], "PADEL STATS", (start_x + 140, start_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        y_offset = start_y + 55
        row_height = 22  # Balanced row height
        label_x = start_x + 15

        # Player colors
        player_colors = {
            1: (0, 200, 0),
            2: (0, 180, 120),
            3: (0, 0, 200),
            4: (0, 100, 200),
        }

        # Calculate team totals for shot types
        team1_shots = {
            'Forehand': shot_types_data[1]['F'] + shot_types_data[2]['F'],
            'Backhand': shot_types_data[1]['B'] + shot_types_data[2]['B'],
            'Lob': shot_types_data[1]['L'] + shot_types_data[2]['L'],
            'Smash': shot_types_data[1]['S'] + shot_types_data[2]['S'],
        }
        team2_shots = {
            'Forehand': shot_types_data[3]['F'] + shot_types_data[4]['F'],
            'Backhand': shot_types_data[3]['B'] + shot_types_data[4]['B'],
            'Lob': shot_types_data[3]['L'] + shot_types_data[4]['L'],
            'Smash': shot_types_data[3]['S'] + shot_types_data[4]['S'],
        }

        # Calculate max for heatmap (across all players and teams)
        all_counts = []
        for pid in [1, 2, 3, 4]:
            all_counts.extend([shot_types_data[pid]['F'], shot_types_data[pid]['B'],
                              shot_types_data[pid]['L'], shot_types_data[pid]['S']])
        all_counts.extend(team1_shots.values())
        all_counts.extend(team2_shots.values())
        max_count = max(all_counts) if all_counts else 1

        # Shot type display names and widths
        shot_info = [
            ('Forehand', 'F', 85),
            ('Backhand', 'B', 85),
            ('Lob', 'L', 52),
            ('Smash', 'S', 68),
        ]

        # Track shot changes for fade effect
        global _prev_shot_counts, _shot_change_frame
        if index == 0:
            _prev_shot_counts = {}
            _shot_change_frame = {}

        # Update shot change tracking and calculate fade intensities
        fade_intensities = {}  # {(entity, shot_key): intensity}

        # Track individual player shots
        for pid in [1, 2, 3, 4]:
            for shot_key in ['F', 'B', 'L', 'S']:
                key = (f'P{pid}', shot_key)
                current = shot_types_data[pid][shot_key]
                prev = _prev_shot_counts.get(key, 0)

                if current > prev:
                    _shot_change_frame[key] = index
                _prev_shot_counts[key] = current

                # Calculate fade intensity
                if key in _shot_change_frame:
                    frames_since = index - _shot_change_frame[key]
                    if frames_since < FADE_DURATION:
                        fade_intensities[key] = 1.0 - (frames_since / FADE_DURATION)
                    else:
                        fade_intensities[key] = 0.0
                else:
                    fade_intensities[key] = 0.0

        # Track team shots
        for team_id, team_data in [('T1', team1_shots), ('T2', team2_shots)]:
            for shot_label in ['Forehand', 'Backhand', 'Lob', 'Smash']:
                key = (team_id, shot_label)
                current = team_data[shot_label]
                prev = _prev_shot_counts.get(key, 0)

                if current > prev:
                    _shot_change_frame[key] = index
                _prev_shot_counts[key] = current

                if key in _shot_change_frame:
                    frames_since = index - _shot_change_frame[key]
                    if frames_since < FADE_DURATION:
                        fade_intensities[key] = 1.0 - (frames_since / FADE_DURATION)
                    else:
                        fade_intensities[key] = 0.0
                else:
                    fade_intensities[key] = 0.0

        def draw_shot_badge(frame, x, y, label, count, fade_intensity=0.0):
            """Draw a shot type badge - darker orange base, flashes bright yellow on new shot"""
            if count == 0:
                bg_color = (40, 40, 40)  # Dark gray for zero
                text_color = (80, 80, 80)
            else:
                # Darker orange base (BGR): (20, 100, 180) = dark/burnt orange
                # Bright flash: (120, 255, 255) = bright yellow/white flash
                base_b, base_g, base_r = 20, 100, 180
                flash_b, flash_g, flash_r = 150, 255, 255

                # Interpolate based on fade intensity (1.0 = bright flash, 0.0 = dark orange)
                b = int(base_b + (flash_b - base_b) * fade_intensity)
                g = int(base_g + (flash_g - base_g) * fade_intensity)
                r = int(base_r + (flash_r - base_r) * fade_intensity)
                bg_color = (b, g, r)
                text_color = (255, 255, 255) if fade_intensity < 0.5 else (0, 0, 0)  # Dark text on bright flash

            # Get width based on label
            badge_width = 85 if label in ['Forehand', 'Backhand'] else (52 if label == 'Lob' else 68)

            cv2.rectangle(frame, (x, y - 17), (x + badge_width, y + 7), bg_color, -1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y - 17), (x + badge_width, y + 7), (80, 80, 80), 1, cv2.LINE_AA)
            text = f"{label}:{count}"
            cv2.putText(frame, text, (x + 3, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        player_row_start = start_x + 75

        # ==================== TEAM TOTALS ====================
        y_offset += 8  # Space after header
        cv2.putText(output_video_frames[index], "TEAM TOTALS", (start_x + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2, cv2.LINE_AA)
        y_offset += row_height + 3

        team1_val_x = start_x + 150
        team2_val_x = start_x + 340

        # Team headers
        cv2.putText(output_video_frames[index], "T1", (team1_val_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "T2", (team2_val_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 2, cv2.LINE_AA)
        y_offset += row_height

        def draw_team_row(label, val1, val2, y_pos):
            cv2.putText(output_video_frames[index], label, (label_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(output_video_frames[index], str(val1), (team1_val_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 255, 150), 1, cv2.LINE_AA)
            cv2.putText(output_video_frames[index], str(val2), (team2_val_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 255), 1, cv2.LINE_AA)

        draw_team_row("Shots", format_val(team1_total_shots, is_int=True), format_val(team2_total_shots, is_int=True), y_offset)
        y_offset += row_height

        draw_team_row("Distance (m)", format_val(team1_total_distance), format_val(team2_total_distance), y_offset)
        y_offset += row_height

        draw_team_row("Rally", f"{format_val(rally_length, is_int=True)} shots", "", y_offset)
        y_offset += row_height + 8

        # Team shot type badges
        cv2.putText(output_video_frames[index], "T1", (label_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 2, cv2.LINE_AA)
        x_pos = player_row_start
        for label, key, w in shot_info:
            count = team1_shots[label]
            fade = fade_intensities.get(('T1', label), 0.0)
            draw_shot_badge(output_video_frames[index], x_pos, y_offset, label, count, fade)
            x_pos += w + 4
        y_offset += row_height + 4

        cv2.putText(output_video_frames[index], "T2", (label_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 2, cv2.LINE_AA)
        x_pos = player_row_start
        for label, key, w in shot_info:
            count = team2_shots[label]
            fade = fade_intensities.get(('T2', label), 0.0)
            draw_shot_badge(output_video_frames[index], x_pos, y_offset, label, count, fade)
            x_pos += w + 4
        y_offset += row_height + 6

        # Separator
        cv2.line(output_video_frames[index], (start_x + 15, y_offset - 3), (end_x - 15, y_offset - 3), (70, 70, 70), 1)
        y_offset += 16  # Space before title

        # ==================== INDIVIDUAL STATS ====================
        cv2.putText(output_video_frames[index], "INDIVIDUAL", (start_x + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2, cv2.LINE_AA)
        y_offset += row_height + 3

        # Player column positions (shifted right to avoid label overlap)
        p1_x = start_x + 120
        p2_x = start_x + 190
        p3_x = start_x + 310
        p4_x = start_x + 380

        # Player labels
        cv2.putText(output_video_frames[index], "P1", (p1_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P2", (p2_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 120), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P3", (p3_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P4", (p4_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 200), 2, cv2.LINE_AA)
        y_offset += row_height

        def draw_4player_row(label, v1, v2, v3, v4, y_pos):
            cv2.putText(output_video_frames[index], label, (label_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(output_video_frames[index], str(v1), (p1_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(output_video_frames[index], str(v2), (p2_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(output_video_frames[index], str(v3), (p3_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(output_video_frames[index], str(v4), (p4_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        draw_4player_row("Shots", format_val(team1_p1_shots, is_int=True), format_val(team1_p2_shots, is_int=True),
                         format_val(team2_p3_shots, is_int=True), format_val(team2_p4_shots, is_int=True), y_offset)
        y_offset += row_height

        draw_4player_row("Distance (m)", format_val(team1_p1_distance), format_val(team1_p2_distance),
                         format_val(team2_p3_distance), format_val(team2_p4_distance), y_offset)
        y_offset += row_height + 8

        # Individual player shot type badges
        for pid in [1, 2, 3, 4]:
            cv2.putText(output_video_frames[index], f"P{pid}", (label_x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, player_colors[pid], 2, cv2.LINE_AA)

            x_pos = player_row_start
            for lbl, shot_key, w in shot_info:
                count = shot_types_data[pid][shot_key]
                fade = fade_intensities.get((f'P{pid}', shot_key), 0.0)
                draw_shot_badge(output_video_frames[index], x_pos, y_offset, lbl, count, fade)
                x_pos += w + 4

            y_offset += row_height + 2

        # Best shots (in individual section) - highlighted
        y_offset += 6
        best_display = [('Smash', 'smash'), ('Forehand', 'forehand')]
        for display_name, key in best_display:
            if best_shots[key]['player'] > 0 and best_shots[key]['speed'] > 0:
                pid = best_shots[key]['player']
                spd = best_shots[key]['speed']
                # Draw subtle background for best shot
                cv2.rectangle(output_video_frames[index], (label_x - 5, y_offset - 14),
                             (label_x + 220, y_offset + 4), (50, 50, 50), -1, cv2.LINE_AA)
                # Star icon effect
                cv2.putText(output_video_frames[index], "*", (label_x, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(output_video_frames[index], f"Best {display_name}:", (label_x + 15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(output_video_frames[index], f"P{pid}", (label_x + 130, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, player_colors[pid], 2, cv2.LINE_AA)
                cv2.putText(output_video_frames[index], f"{spd:.0f}km/h", (label_x + 165, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
                y_offset += row_height + 2

        y_offset += 2

        # Separator
        cv2.line(output_video_frames[index], (start_x + 15, y_offset - 3), (end_x - 15, y_offset - 3), (70, 70, 70), 1)
        y_offset += 16  # Space before title

        # ==================== MOVEMENT (km/h) ====================
        cv2.putText(output_video_frames[index], "MOVEMENT (km/h)", (start_x + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2, cv2.LINE_AA)
        y_offset += row_height + 3

        cv2.putText(output_video_frames[index], "P1", (p1_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P2", (p2_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 120), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P3", (p3_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P4", (p4_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 200), 2, cv2.LINE_AA)
        y_offset += row_height

        draw_4player_row("Current", format_val(team1_p1_speed), format_val(team1_p2_speed),
                         format_val(team2_p3_speed), format_val(team2_p4_speed), y_offset)
        y_offset += row_height

        draw_4player_row("Average", format_val(team1_p1_avg_speed), format_val(team1_p2_avg_speed),
                         format_val(team2_p3_avg_speed), format_val(team2_p4_avg_speed), y_offset)
        y_offset += row_height

        draw_4player_row("Top Speed", format_val(team1_p1_top_speed), format_val(team1_p2_top_speed),
                         format_val(team2_p3_top_speed), format_val(team2_p4_top_speed), y_offset)
        y_offset += row_height + 6

        # Separator
        cv2.line(output_video_frames[index], (start_x + 15, y_offset - 3), (end_x - 15, y_offset - 3), (70, 70, 70), 1)
        y_offset += 16  # Space before title

        # ==================== SHOT SPEED (Ball) ====================
        cv2.putText(output_video_frames[index], "SHOT SPEED (km/h)", (start_x + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 2, cv2.LINE_AA)
        y_offset += row_height + 3

        cv2.putText(output_video_frames[index], "P1", (p1_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P2", (p2_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 120), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P3", (p3_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P4", (p4_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 200), 2, cv2.LINE_AA)
        y_offset += row_height

        draw_4player_row("Last Shot", format_val(team1_p1_shot_speed), format_val(team1_p2_shot_speed),
                         format_val(team2_p3_shot_speed), format_val(team2_p4_shot_speed), y_offset)
        y_offset += row_height

        draw_4player_row("Avg Shot", format_val(team1_p1_avg_shot), format_val(team1_p2_avg_shot),
                         format_val(team2_p3_avg_shot), format_val(team2_p4_avg_shot), y_offset)
        y_offset += row_height + 6

        # Separator
        cv2.line(output_video_frames[index], (start_x + 15, y_offset - 3), (end_x - 15, y_offset - 3), (70, 70, 70), 1)
        y_offset += 16  # Space before title

        # ==================== COURT POSITION ====================
        cv2.putText(output_video_frames[index], "COURT POSITION (%)", (start_x + 15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 255), 2, cv2.LINE_AA)
        y_offset += row_height + 3

        cv2.putText(output_video_frames[index], "P1", (p1_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P2", (p2_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 120), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P3", (p3_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 2, cv2.LINE_AA)
        cv2.putText(output_video_frames[index], "P4", (p4_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 200), 2, cv2.LINE_AA)
        y_offset += row_height

        draw_4player_row("Net", format_val(team1_p1_net, decimals=0), format_val(team1_p2_net, decimals=0),
                         format_val(team2_p3_net, decimals=0), format_val(team2_p4_net, decimals=0), y_offset)
        y_offset += row_height

        draw_4player_row("Midcourt", format_val(team1_p1_midcourt, decimals=0), format_val(team1_p2_midcourt, decimals=0),
                         format_val(team2_p3_midcourt, decimals=0), format_val(team2_p4_midcourt, decimals=0), y_offset)
        y_offset += row_height

        draw_4player_row("Baseline", format_val(team1_p1_baseline, decimals=0), format_val(team1_p2_baseline, decimals=0),
                         format_val(team2_p3_baseline, decimals=0), format_val(team2_p4_baseline, decimals=0), y_offset)

    return output_video_frames
