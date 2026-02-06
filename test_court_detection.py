"""
Test script for padel court detection with manual refinement option.
"""
import cv2
import numpy as np
import sys
sys.path.append('.')

from court_line_detector.court_line_detector import PadelCourtDetectorColor


class ManualCourtSelector:
    """
    Manual court point selection tool.
    Allows user to click on all 14 court keypoints.
    """

    def __init__(self, image, initial_keypoints=None):
        self.image = image.copy()
        self.display_image = image.copy()
        self.points = []
        self.point_names = [
            "0: Far-Left (top-left corner)",
            "1: Far-Right (top-right corner)",
            "2: Near-Left (bottom-left corner)",
            "3: Near-Right (bottom-right corner)",
            "4: Service-Near-Left",
            "5: Service-Near-Right",
            "6: Service-Far-Left",
            "7: Service-Far-Right",
            "8: Center-Near (service line)",
            "9: Center-Far (service line)",
            "10: Net-Left",
            "11: Net-Right",
            "12: Center-Top (far baseline)",
            "13: Center-Bottom (near baseline)"
        ]
        self.num_points = 14
        self.current_point = 0
        self.window_name = "Manual Court Selection (R=reset, Enter=confirm, S=skip)"
        self.initial_keypoints = initial_keypoints

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_point < self.num_points:
            self.points.append((x, y))
            print(f"  Point {self.current_point}: ({x}, {y})")
            self.current_point += 1
            self._update_display()

    def _update_display(self):
        self.display_image = self.image.copy()

        # Draw initial detection as reference (faded gray)
        if self.initial_keypoints is not None:
            # Draw all lines
            lines = [
                (0, 1), (2, 3),   # baselines
                (0, 2), (1, 3),   # sidelines
                (4, 5), (6, 7),   # service lines
                (8, 9),           # center service
                (10, 11),         # net
            ]
            for i1, i2 in lines:
                x1 = int(self.initial_keypoints[i1*2])
                y1 = int(self.initial_keypoints[i1*2+1])
                x2 = int(self.initial_keypoints[i2*2])
                y2 = int(self.initial_keypoints[i2*2+1])
                cv2.line(self.display_image, (x1, y1), (x2, y2), (100, 100, 100), 1)

            # Draw all points with numbers
            for i in range(14):
                x = int(self.initial_keypoints[i*2])
                y = int(self.initial_keypoints[i*2+1])
                cv2.circle(self.display_image, (x, y), 4, (100, 100, 100), -1)
                cv2.putText(self.display_image, str(i), (x+5, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Draw manually selected points
        colors = [
            (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),  # corners - green
            (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0),  # service - cyan
            (255, 0, 255), (255, 0, 255),  # center service - magenta
            (0, 165, 255), (0, 165, 255),  # net - orange
            (255, 0, 0), (255, 0, 0),  # center baseline - blue
        ]

        for i, (x, y) in enumerate(self.points):
            color = colors[i] if i < len(colors) else (255, 255, 255)
            cv2.circle(self.display_image, (x, y), 8, color, -1)
            cv2.circle(self.display_image, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(self.display_image, str(i), (x + 12, y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw lines between selected points
        if len(self.points) >= 2:
            cv2.line(self.display_image, self.points[0], self.points[1], (0, 255, 0), 2)
        if len(self.points) >= 4:
            cv2.line(self.display_image, self.points[2], self.points[3], (0, 255, 0), 2)
            cv2.line(self.display_image, self.points[0], self.points[2], (0, 255, 0), 2)
            cv2.line(self.display_image, self.points[1], self.points[3], (0, 255, 0), 2)
        if len(self.points) >= 6:
            cv2.line(self.display_image, self.points[4], self.points[5], (255, 255, 0), 2)
        if len(self.points) >= 8:
            cv2.line(self.display_image, self.points[6], self.points[7], (255, 255, 0), 2)
        if len(self.points) >= 10:
            cv2.line(self.display_image, self.points[8], self.points[9], (255, 0, 255), 2)
        if len(self.points) >= 12:
            cv2.line(self.display_image, self.points[10], self.points[11], (0, 165, 255), 2)

        # Show instructions
        if self.current_point < self.num_points:
            text = f"Click: {self.point_names[self.current_point]}"
            cv2.putText(self.display_image, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(self.display_image, "S=skip this point, R=reset all", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(self.display_image, "All points selected! Press ENTER to confirm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def select(self):
        """Run the manual selection interface"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\nManual Court Selection - Click all 14 points:")
        print("Corners (green):")
        print("  0: Far-Left, 1: Far-Right, 2: Near-Left, 3: Near-Right")
        print("Service lines (cyan):")
        print("  4: Service-Near-Left, 5: Service-Near-Right")
        print("  6: Service-Far-Left, 7: Service-Far-Right")
        print("Center service (magenta):")
        print("  8: Center-Near, 9: Center-Far")
        print("Net (orange):")
        print("  10: Net-Left, 11: Net-Right")
        print("Center baseline (blue):")
        print("  12: Center-Top, 13: Center-Bottom")
        print("\nKeys: S=skip point, R=reset, ENTER=confirm, ESC=cancel\n")

        self._update_display()

        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and len(self.points) == self.num_points:  # Enter
                break
            elif key == ord('s') and self.current_point < self.num_points:  # Skip
                # Use auto-detected point if available
                if self.initial_keypoints is not None:
                    x = self.initial_keypoints[self.current_point * 2]
                    y = self.initial_keypoints[self.current_point * 2 + 1]
                    self.points.append((x, y))
                    print(f"  Point {self.current_point}: SKIPPED (using auto: {x:.0f}, {y:.0f})")
                else:
                    self.points.append((0, 0))
                    print(f"  Point {self.current_point}: SKIPPED")
                self.current_point += 1
                self._update_display()
            elif key == ord('r'):  # Reset
                self.points = []
                self.current_point = 0
                print("\nReset - click points again:")
                self._update_display()
            elif key == 27:  # Escape
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()

        # Convert to keypoints array (28 values)
        keypoints = np.zeros(28, dtype=np.float32)
        for i, (x, y) in enumerate(self.points):
            keypoints[i*2] = x
            keypoints[i*2+1] = y
        return keypoints


def test_court_detection(image_path, use_manual=False):
    """Test court detection on a single image"""

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Loaded image: {w}x{h}")

    # Create detector and get automatic detection first
    detector = PadelCourtDetectorColor()

    print("\n=== Automatic Detection ===")
    auto_keypoints = detector.predict(image)

    print("Automatic keypoints:")
    point_names = ['Far-Left', 'Far-Right', 'Near-Left', 'Near-Right',
        'Svc-Near-L', 'Svc-Near-R', 'Svc-Far-L', 'Svc-Far-R',
        'Center-Near', 'Center-Far', 'Net-Left', 'Net-Right',
        'Center-Top', 'Center-Bot']
    for i in range(14):
        x, y = auto_keypoints[i*2], auto_keypoints[i*2+1]
        print(f"  {i:2d}: {point_names[i]:12s} = ({x:7.1f}, {y:7.1f})")

    if use_manual:
        print("\n=== Manual Selection ===")
        selector = ManualCourtSelector(image, auto_keypoints)
        keypoints = selector.select()

        if keypoints is None:
            print("Selection cancelled")
            return

        print("\n" + "="*60)
        print("MANUAL KEYPOINTS (copy these to update the code):")
        print("="*60)
        for i in range(14):
            x, y = keypoints[i*2], keypoints[i*2+1]
            print(f"  {i:2d}: {point_names[i]:12s} = ({x:7.1f}, {y:7.1f})")

        print(f"\nKey measurements (image: {w}x{h}):")
        print(f"  Far baseline y:  {keypoints[1]:.0f}px = {keypoints[1]/h*100:.1f}%")
        print(f"  Near baseline y: {keypoints[5]:.0f}px = {keypoints[5]/h*100:.1f}%")
        print(f"  Net y:           {keypoints[21]:.0f}px = {keypoints[21]/h*100:.1f}%")
    else:
        keypoints = auto_keypoints

    # Draw and save result
    result = detector.draw_keypoints(image.copy(), keypoints)
    cv2.imwrite("court_detection_result.png", result)
    print("\nResult saved to court_detection_result.png")

    # Show result
    cv2.namedWindow("Court Detection Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Court Detection Result", 1280, 720)
    cv2.imshow("Court Detection Result", result)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return keypoints


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test padel court detection")
    parser.add_argument("--image", "-i", default="test_frame.png", help="Path to test image")
    parser.add_argument("--manual", "-m", action="store_true", help="Use manual point selection")

    args = parser.parse_args()

    test_court_detection(args.image, use_manual=args.manual)
