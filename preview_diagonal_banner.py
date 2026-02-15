"""
Preview the diagonal banner animation
"""
import cv2
import numpy as np
import sys
sys.path.append('.')
from player_highlights import draw_diagonal_banner
from utils import save_video


def create_preview():
    """Create a preview video of the diagonal banner animation"""
    frame_w, frame_h = 1920, 1080
    fps = 30

    # Create a sample background (dark with some texture)
    base_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    for y in range(frame_h):
        progress = y / frame_h
        base_frame[y, :] = [int(40 - progress * 20), int(35 - progress * 15), int(30 - progress * 10)]

    # Add some noise for texture
    noise = np.random.randint(0, 15, (frame_h, frame_w, 3), dtype=np.uint8)
    base_frame = cv2.add(base_frame, noise)

    frames = []

    # Animation phases (slower transitions):
    # enter (0.8s) → stay (0.6s) → ball_transition (1.0s) → stay_fire (0.6s) → exit (0.5s)
    enter_frames = int(fps * 0.8)
    stay_frames = int(fps * 0.6)
    ball_frames = int(fps * 1.0)
    stay_fire_frames = int(fps * 0.6)
    exit_frames = int(fps * 0.5)

    phase_boundaries = [
        enter_frames,
        enter_frames + stay_frames,
        enter_frames + stay_frames + ball_frames,
        enter_frames + stay_frames + ball_frames + stay_fire_frames,
        enter_frames + stay_frames + ball_frames + stay_fire_frames + exit_frames
    ]
    total_frames = phase_boundaries[-1]

    for i in range(total_frames):
        frame = base_frame.copy()

        # Determine phase and progress
        if i < phase_boundaries[0]:
            phase = 'enter'
            progress = i / enter_frames
        elif i < phase_boundaries[1]:
            phase = 'stay'
            progress = (i - phase_boundaries[0]) / stay_frames
        elif i < phase_boundaries[2]:
            phase = 'ball_transition'
            progress = (i - phase_boundaries[1]) / ball_frames
        elif i < phase_boundaries[3]:
            phase = 'stay_fire'
            progress = (i - phase_boundaries[2]) / stay_fire_frames
        else:
            phase = 'exit'
            progress = (i - phase_boundaries[3]) / exit_frames

        # Draw the diagonal banner
        draw_diagonal_banner(frame, "SMASH", progress, phase=phase)

        # Add speed display
        show_speed = phase in ['stay', 'ball_transition', 'stay_fire'] or (phase == 'exit' and progress < 0.3)
        if show_speed:
            font = cv2.FONT_HERSHEY_SIMPLEX
            speed_text = "82.3"
            speed_unit = "Km/h"

            speed_x = frame_w // 2 - 80
            speed_y = frame_h // 3 + 150
            cv2.putText(frame, speed_text, (speed_x + 3, speed_y + 3),
                       font, 2.5, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(frame, speed_text, (speed_x, speed_y),
                       font, 2.5, (255, 255, 255), 6, cv2.LINE_AA)
            cv2.putText(frame, speed_unit, (speed_x + 40, speed_y + 45),
                       font, 1.0, (200, 200, 200), 2, cv2.LINE_AA)

        frames.append(frame)

    # Hold last frame briefly
    for _ in range(int(fps * 0.5)):
        frames.append(base_frame.copy())

    # Save preview
    output_path = "output/diagonal_banner_preview.mp4"
    import os
    os.makedirs("output", exist_ok=True)
    save_video(frames, output_path)
    print(f"Preview saved to: {output_path}")

    # Save key frames for visualization
    cv2.imwrite("output/banner_enter.png", frames[enter_frames // 2])
    cv2.imwrite("output/banner_stay.png", frames[phase_boundaries[0] + stay_frames // 2])
    cv2.imwrite("output/banner_ball.png", frames[phase_boundaries[1] + ball_frames // 2])
    cv2.imwrite("output/banner_fire.png", frames[phase_boundaries[2] + stay_fire_frames // 2])
    cv2.imwrite("output/banner_exit.png", frames[phase_boundaries[3] + exit_frames // 2])

    print("Key frames saved to output/banner_*.png")


if __name__ == "__main__":
    create_preview()
