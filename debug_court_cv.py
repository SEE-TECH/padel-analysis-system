"""Debug script for CV-based court detection (gchlebus algorithm)"""
import cv2
import numpy as np
from utils import read_video
from court_line_detector import CourtLineDetectorCV, CourtLinePixelDetector, CourtLineCandidateDetector

def main():
    # Read first frame
    video_frames = read_video("inputs/input_video.mp4")
    if not video_frames:
        print("Could not read video")
        return

    frame = video_frames[0]
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    # Step 1: Pixel detection
    print("\n=== Step 1: Court Line Pixel Detection ===")
    pixel_detector = CourtLinePixelDetector()
    binary_mask = pixel_detector.detect(frame)
    cv2.imwrite("debug_binary_mask.jpg", binary_mask)
    print(f"Saved debug_binary_mask.jpg (white pixels: {np.sum(binary_mask > 0)})")

    # Step 2: Line candidate detection
    print("\n=== Step 2: Line Candidate Detection ===")
    candidate_detector = CourtLineCandidateDetector()
    lines = candidate_detector.detect(binary_mask)
    print(f"Detected {len(lines)} line candidates")

    # Draw lines
    lines_frame = frame.copy()
    h_lines = [l for l in lines if not l.is_vertical()]
    v_lines = [l for l in lines if l.is_vertical()]
    print(f"  Horizontal: {len(h_lines)}, Vertical: {len(v_lines)}")

    for line in h_lines:
        # Draw line extending across image
        p1 = line.point - 1000 * line.direction
        p2 = line.point + 1000 * line.direction
        cv2.line(lines_frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 2)

    for line in v_lines:
        p1 = line.point - 1000 * line.direction
        p2 = line.point + 1000 * line.direction
        cv2.line(lines_frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

    cv2.imwrite("debug_lines.jpg", lines_frame)
    print("Saved debug_lines.jpg (blue=horizontal, green=vertical)")

    # Step 3: Full detection
    print("\n=== Step 3: Court Model Fitting ===")
    detector = CourtLineDetectorCV()
    keypoints = detector.predict(frame)
    print(f"Got {len(keypoints)//2} keypoints")

    # Draw keypoints
    keypoints_frame = frame.copy()
    for i in range(0, len(keypoints), 2):
        x = int(keypoints[i])
        y = int(keypoints[i + 1])
        cv2.putText(keypoints_frame, str(i // 2), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(keypoints_frame, (x, y), 5, (0, 255, 0), -1)

    # Draw court outline using first 4 corners
    corners = [
        (int(keypoints[0]), int(keypoints[1])),   # 0: top-left
        (int(keypoints[2]), int(keypoints[3])),   # 1: top-right
        (int(keypoints[6]), int(keypoints[7])),   # 3: bottom-right
        (int(keypoints[4]), int(keypoints[5])),   # 2: bottom-left
    ]
    for i in range(4):
        cv2.line(keypoints_frame, corners[i], corners[(i+1)%4], (0, 255, 255), 2)

    cv2.imwrite("debug_keypoints.jpg", keypoints_frame)
    print("Saved debug_keypoints.jpg")

    print("\nKeypoint positions (court corners):")
    print(f"  0 (top-left):     ({keypoints[0]:.1f}, {keypoints[1]:.1f})")
    print(f"  1 (top-right):    ({keypoints[2]:.1f}, {keypoints[3]:.1f})")
    print(f"  2 (bottom-left):  ({keypoints[4]:.1f}, {keypoints[5]:.1f})")
    print(f"  3 (bottom-right): ({keypoints[6]:.1f}, {keypoints[7]:.1f})")

if __name__ == "__main__":
    main()
