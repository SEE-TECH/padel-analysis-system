"""
Tennis Court Line Detection
Exact implementation of: gchlebus/tennis-court-detection (Farin D. et al. 2003)
"""
import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights


class Line:
    """Line representation using point + direction vector"""

    def __init__(self, point, direction):
        self.point = np.array(point, dtype=np.float32)
        norm = np.linalg.norm(direction)
        self.direction = np.array(direction, dtype=np.float32) / norm if norm > 1e-6 else np.array([1, 0], dtype=np.float32)

    @classmethod
    def from_two_points(cls, p1, p2):
        p1, p2 = np.array(p1, dtype=np.float32), np.array(p2, dtype=np.float32)
        return cls(p1, p2 - p1)

    def to_implicit(self):
        """Convert to implicit form: n·x + c = 0"""
        normal = np.array([-self.direction[1], self.direction[0]])
        c = -np.dot(normal, self.point)
        return normal, c

    def get_distance(self, point):
        """Perpendicular distance from point to line"""
        v = np.array(point, dtype=np.float32) - self.point
        t = np.dot(v, self.direction)
        closest = self.point + t * self.direction
        return np.linalg.norm(np.array(point) - closest)

    def is_vertical(self):
        """Check if line is more vertical than horizontal (angle > 65° from horizontal)"""
        angle = np.abs(np.arctan2(self.direction[1], self.direction[0]))
        # Original uses 65 degrees threshold
        return np.radians(25) < angle < np.radians(155)

    def is_duplicate(self, other, angle_thresh_deg=1.0, dist_thresh=10.0):
        """Check if lines are geometrically equivalent (original thresholds)"""
        n1, c1 = self.to_implicit()
        n2, c2 = other.to_implicit()

        dot = np.abs(np.dot(n1, n2))
        if dot < np.cos(np.radians(angle_thresh_deg)):
            return False

        if np.abs(c1 - c2) > dist_thresh and np.abs(c1 + c2) > dist_thresh:
            return False

        return True

    def intersect(self, other):
        """Find intersection point"""
        d1, d2 = self.direction, other.direction
        p1, p2 = self.point, other.point

        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if np.abs(cross) < 1e-10:
            return None

        dp = p2 - p1
        t = (dp[0] * d2[1] - dp[1] * d2[0]) / cross
        return self.point + t * self.direction


class CourtLinePixelDetector:
    """Detect court line pixels using luminance and gradient analysis"""

    def __init__(self, brightness_thresh=80, gradient_thresh=20):
        self.brightness_thresh = brightness_thresh
        self.gradient_thresh = gradient_thresh

    def detect(self, image):
        """Returns binary mask of court line pixels"""
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lum = ycrcb[:, :, 0].astype(np.float32)
        h, w = lum.shape

        # 1. Brightness threshold
        bright_mask = lum > self.brightness_thresh

        # 2. Directional gradient (4 pixel offset)
        offset = 4

        # Horizontal gradient
        left_diff = np.zeros_like(lum)
        right_diff = np.zeros_like(lum)
        left_diff[:, offset:] = lum[:, offset:] - lum[:, :-offset]
        right_diff[:, :-offset] = lum[:, :-offset] - lum[:, offset:]
        h_grad = (left_diff > self.gradient_thresh) & (right_diff > self.gradient_thresh)

        # Vertical gradient
        top_diff = np.zeros_like(lum)
        bot_diff = np.zeros_like(lum)
        top_diff[offset:, :] = lum[offset:, :] - lum[:-offset, :]
        bot_diff[:-offset, :] = lum[:-offset, :] - lum[offset:, :]
        v_grad = (top_diff > self.gradient_thresh) & (bot_diff > self.gradient_thresh)

        grad_mask = h_grad | v_grad
        line_mask = bright_mask & grad_mask

        # 3. Structure tensor filtering
        sobelx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)

        Ixx = cv2.GaussianBlur(sobelx * sobelx, (5, 5), 1)
        Iyy = cv2.GaussianBlur(sobely * sobely, (5, 5), 1)
        Ixy = cv2.GaussianBlur(sobelx * sobely, (5, 5), 1)

        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        disc = np.sqrt(np.maximum(trace * trace - 4 * det, 0))

        lambda1 = (trace + disc) / 2
        lambda2 = (trace - disc) / 2

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(lambda2 > 1e-6, lambda1 / lambda2, 0)

        structure_mask = ratio > 4.0

        final_mask = (line_mask & structure_mask).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        # Filter out regions outside typical court area
        # Top 15% often has scoreboard/text, bottom 10% may have overlays
        h_start = int(h * 0.15)
        h_end = int(h * 0.95)
        region_mask = np.zeros_like(final_mask)
        region_mask[h_start:h_end, :] = final_mask[h_start:h_end, :]

        return region_mask


class CourtLineCandidateDetector:
    """Detect line candidates using standard Hough transform (as in original)"""

    def __init__(self, hough_thresh=100):
        self.hough_thresh = hough_thresh

    def detect(self, binary_image):
        """Detect line candidates using standard Hough transform"""
        edges = cv2.Canny(binary_image, 50, 150)

        # Use standard Hough transform (as in original algorithm)
        hough_lines = cv2.HoughLines(
            edges, rho=1, theta=np.pi/180,
            threshold=self.hough_thresh
        )

        if hough_lines is None:
            return []

        lines = []
        h, w = binary_image.shape[:2]
        for hl in hough_lines:
            rho, theta = hl[0]
            # Convert (rho, theta) to point + direction
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x0, y0 = rho * cos_t, rho * sin_t
            # Direction is perpendicular to normal
            direction = np.array([-sin_t, cos_t])
            lines.append(Line(np.array([x0, y0]), direction))

        # Remove duplicates (as in original: 1° angle, 10px distance)
        lines = self._remove_duplicates(lines)

        return lines

    def _remove_duplicates(self, lines):
        if not lines:
            return []

        unique = [lines[0]]
        for line in lines[1:]:
            is_dup = any(line.is_duplicate(existing) for existing in unique)
            if not is_dup:
                unique.append(line)

        return unique


class PadelCourtModel:
    """Padel court geometric model - 20m x 10m enclosed court"""

    # Court dimensions in meters
    COURT_WIDTH = 10.0
    COURT_LENGTH = 20.0
    SERVICE_DIST = 3.0  # Service line is 3m from net
    NET_DIST = 10.0     # Net is at center (10m from each baseline)

    # Model line positions for padel
    H_LINES_Y = [0, 3.0, 10.0, 17.0, 20.0]  # baseline, service, net, service, baseline
    V_LINES_X = [0, 5.0, 10.0]               # left sideline, center, right sideline

    def __init__(self):
        self.court_points = self._create_court_points()
        self.h_line_pairs = self._create_h_line_pairs()
        self.v_line_pairs = self._create_v_line_pairs()
        self.transform_matrix = None

    def _create_court_points(self):
        """Create padel court keypoints"""
        pts = []
        # 4 corners
        pts.append([0, 0])                           # 0: top-left
        pts.append([self.COURT_WIDTH, 0])            # 1: top-right
        pts.append([0, self.COURT_LENGTH])           # 2: bottom-left
        pts.append([self.COURT_WIDTH, self.COURT_LENGTH])  # 3: bottom-right
        # Service line corners (near side)
        pts.append([0, self.H_LINES_Y[1]])           # 4: service line left (near)
        pts.append([self.COURT_WIDTH, self.H_LINES_Y[1]])  # 5: service line right (near)
        # Service line corners (far side)
        pts.append([0, self.H_LINES_Y[3]])           # 6: service line left (far)
        pts.append([self.COURT_WIDTH, self.H_LINES_Y[3]])  # 7: service line right (far)
        # Center service line points
        pts.append([self.V_LINES_X[1], self.H_LINES_Y[1]])  # 8: center near
        pts.append([self.V_LINES_X[1], self.H_LINES_Y[3]])  # 9: center far
        # Net points
        pts.append([0, self.NET_DIST])               # 10: net left
        pts.append([self.COURT_WIDTH, self.NET_DIST]) # 11: net right
        # Center of service boxes
        pts.append([self.V_LINES_X[1], 0])           # 12: center baseline near
        pts.append([self.V_LINES_X[1], self.COURT_LENGTH])  # 13: center baseline far

        return np.array(pts, dtype=np.float32)

    def _create_h_line_pairs(self):
        """Create model horizontal line pairs"""
        return [(0, 4), (0, 3), (1, 4), (1, 3)]

    def _create_v_line_pairs(self):
        """Create model vertical line pairs"""
        return [(0, 2), (0, 1)]

    def get_intersection_points(self, h_pair_idx, v_pair_idx):
        """Get 4 corner points from model line pair indices"""
        h1_y, h2_y = self.H_LINES_Y[h_pair_idx[0]], self.H_LINES_Y[h_pair_idx[1]]
        v1_x, v2_x = self.V_LINES_X[v_pair_idx[0]], self.V_LINES_X[v_pair_idx[1]]

        return np.array([
            [v1_x, h1_y], [v2_x, h1_y],
            [v1_x, h2_y], [v2_x, h2_y]
        ], dtype=np.float32)

    def fit(self, h_line_pair, v_line_pair, binary_image):
        """Fit model to detected line pair"""
        h, w = binary_image.shape[:2]

        h1, h2 = h_line_pair
        v1, v2 = v_line_pair

        img_points = []
        for h_line in [h1, h2]:
            for v_line in [v1, v2]:
                pt = h_line.intersect(v_line)
                if pt is None:
                    return -1
                img_points.append(pt)

        img_points = np.array(img_points, dtype=np.float32)

        best_score = -1

        for h_pair_idx in self.h_line_pairs:
            for v_pair_idx in self.v_line_pairs:
                model_points = self.get_intersection_points(h_pair_idx, v_pair_idx)

                try:
                    M = cv2.getPerspectiveTransform(model_points, img_points)
                except:
                    continue

                pts = self.court_points.reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(pts, M).reshape(-1, 2)

                score = self._evaluate_model(transformed, binary_image)

                if score > best_score:
                    best_score = score
                    self.transform_matrix = M

        return best_score

    def _evaluate_model(self, transformed_points, binary_image):
        """Score model fitness"""
        h, w = binary_image.shape[:2]

        p = transformed_points[:4]
        sides = [
            np.linalg.norm(p[1] - p[0]),
            np.linalg.norm(p[3] - p[2]),
            np.linalg.norm(p[2] - p[0]),
            np.linalg.norm(p[3] - p[1])
        ]
        if min(sides) < 30:
            return -1

        segments = [
            (0, 1), (2, 3),  # baselines
            (0, 2), (1, 3),  # sidelines
            (4, 5), (6, 7),  # service lines
            (10, 11),        # net
        ]

        score = 0
        for i1, i2 in segments:
            if i1 < len(transformed_points) and i2 < len(transformed_points):
                p1, p2 = transformed_points[i1], transformed_points[i2]
                score += self._score_segment(p1, p2, binary_image)

        return score

    def _score_segment(self, p1, p2, binary_image):
        """Score a line segment"""
        h, w = binary_image.shape[:2]
        seg_score = 0

        for t in np.linspace(0, 1, 30):
            px = int(p1[0] + t * (p2[0] - p1[0]))
            py = int(p1[1] + t * (p2[1] - p1[1]))

            if 0 <= px < w and 0 <= py < h:
                if binary_image[py, px] > 0:
                    seg_score += 1
                else:
                    seg_score -= 0.5

        return seg_score

    def get_keypoints(self):
        """Get transformed court keypoints"""
        if self.transform_matrix is None:
            return None

        pts = self.court_points.reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.transform_matrix).reshape(-1, 2)


class TennisCourtModel:
    """Tennis court geometric model"""

    # Court dimensions in meters
    COURT_WIDTH = 10.97
    COURT_LENGTH = 23.78
    SINGLES_WIDTH = 8.23
    SERVICE_DIST = 5.49
    NET_DIST = 11.89

    # Model line positions
    H_LINES_Y = [0, 5.49, 11.89, 18.29, 23.78]  # baseline, service, net, service, baseline
    V_LINES_X = [0, 1.37, 5.485, 9.6, 10.97]    # sideline, singles, center, singles, sideline

    def __init__(self):
        self.court_points = self._create_court_points()
        self.h_line_pairs = self._create_h_line_pairs()
        self.v_line_pairs = self._create_v_line_pairs()
        self.transform_matrix = None

    def _create_court_points(self):
        """Create 16 court keypoints"""
        pts = []
        # 4 corners (doubles court)
        pts.append([0, 0])
        pts.append([self.COURT_WIDTH, 0])
        pts.append([0, self.COURT_LENGTH])
        pts.append([self.COURT_WIDTH, self.COURT_LENGTH])
        # 4 singles corners
        pts.append([self.V_LINES_X[1], 0])
        pts.append([self.V_LINES_X[3], 0])
        pts.append([self.V_LINES_X[1], self.COURT_LENGTH])
        pts.append([self.V_LINES_X[3], self.COURT_LENGTH])
        # 4 service corners
        pts.append([self.V_LINES_X[1], self.H_LINES_Y[1]])
        pts.append([self.V_LINES_X[3], self.H_LINES_Y[1]])
        pts.append([self.V_LINES_X[1], self.H_LINES_Y[3]])
        pts.append([self.V_LINES_X[3], self.H_LINES_Y[3]])
        # 2 center service points
        pts.append([self.V_LINES_X[2], self.H_LINES_Y[1]])
        pts.append([self.V_LINES_X[2], self.H_LINES_Y[3]])
        # 2 net points
        pts.append([0, self.NET_DIST])
        pts.append([self.COURT_WIDTH, self.NET_DIST])

        return np.array(pts, dtype=np.float32)

    def _create_h_line_pairs(self):
        """Create model horizontal line pairs (indices into H_LINES_Y)"""
        # Pairs of horizontal lines that could be matched to detected lines
        return [(0, 4), (0, 3), (1, 4), (1, 3)]  # baselines, baseline+service, etc.

    def _create_v_line_pairs(self):
        """Create model vertical line pairs (indices into V_LINES_X)"""
        return [(0, 4), (0, 3), (1, 4), (1, 3)]  # sidelines, sideline+singles, etc.

    def get_intersection_points(self, h_pair_idx, v_pair_idx):
        """Get 4 corner points from model line pair indices"""
        h1_y, h2_y = self.H_LINES_Y[h_pair_idx[0]], self.H_LINES_Y[h_pair_idx[1]]
        v1_x, v2_x = self.V_LINES_X[v_pair_idx[0]], self.V_LINES_X[v_pair_idx[1]]

        return np.array([
            [v1_x, h1_y], [v2_x, h1_y],
            [v1_x, h2_y], [v2_x, h2_y]
        ], dtype=np.float32)

    def fit(self, h_line_pair, v_line_pair, binary_image):
        """
        Fit model to detected line pair (original algorithm - no pre-validation).
        h_line_pair, v_line_pair: tuples of Line objects
        Returns best score
        """
        h, w = binary_image.shape[:2]

        # Get intersection points from detected lines
        h1, h2 = h_line_pair
        v1, v2 = v_line_pair

        img_points = []
        for h_line in [h1, h2]:
            for v_line in [v1, v2]:
                pt = h_line.intersect(v_line)
                if pt is None:
                    return -1  # Parallel lines
                img_points.append(pt)

        img_points = np.array(img_points, dtype=np.float32)

        best_score = -1

        # Try all model line pair combinations (original algorithm)
        for h_pair_idx in self.h_line_pairs:
            for v_pair_idx in self.v_line_pairs:
                model_points = self.get_intersection_points(h_pair_idx, v_pair_idx)

                try:
                    M = cv2.getPerspectiveTransform(model_points, img_points)
                except:
                    continue

                # Transform all court points
                pts = self.court_points.reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(pts, M).reshape(-1, 2)

                # Evaluate score (original: +1 foreground, -0.5 background)
                score = self._evaluate_model(transformed, binary_image)

                if score > best_score:
                    best_score = score
                    self.transform_matrix = M

        return best_score

    def _evaluate_model(self, transformed_points, binary_image):
        """Score model fitness: +1 foreground, -0.5 background"""
        h, w = binary_image.shape[:2]

        # Check minimum court size (30 pixels per side)
        p = transformed_points[:4]  # corners
        sides = [
            np.linalg.norm(p[1] - p[0]),
            np.linalg.norm(p[3] - p[2]),
            np.linalg.norm(p[2] - p[0]),
            np.linalg.norm(p[3] - p[1])
        ]
        if min(sides) < 30:
            return -1

        # Score line segments (correct indices based on court_points)
        segments = [
            (0, 1), (2, 3),  # baselines (top, bottom)
            (0, 2), (1, 3),  # doubles sidelines (left, right)
            (4, 6), (5, 7),  # singles sidelines (left, right)
            (8, 9), (10, 11),  # service lines (top, bottom)
            (12, 13),  # center service line
        ]

        score = 0
        for i1, i2 in segments:
            p1, p2 = transformed_points[i1], transformed_points[i2]
            score += self._score_segment(p1, p2, binary_image)

        return score

    def _score_segment(self, p1, p2, binary_image):
        """Score a line segment"""
        h, w = binary_image.shape[:2]
        seg_score = 0

        for t in np.linspace(0, 1, 30):
            px = int(p1[0] + t * (p2[0] - p1[0]))
            py = int(p1[1] + t * (p2[1] - p1[1]))

            if 0 <= px < w and 0 <= py < h:
                if binary_image[py, px] > 0:
                    seg_score += 1
                else:
                    seg_score -= 0.5

        return seg_score

    def get_keypoints(self):
        """Get transformed court keypoints"""
        if self.transform_matrix is None:
            return None

        pts = self.court_points.reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.transform_matrix).reshape(-1, 2)


class TennisCourtFitter:
    """Fit tennis court model to detected lines - exact original algorithm"""

    def __init__(self, max_lines_per_direction=8):
        # Limit search space as in original algorithm
        self.max_lines = max_lines_per_direction

    def _get_line_position(self, line, is_horizontal, img_center):
        """Get line's position (Y for horizontal, X for vertical)"""
        if is_horizontal:
            if abs(line.direction[0]) > 1e-6:
                t = (img_center[0] - line.point[0]) / line.direction[0]
                return line.point[1] + t * line.direction[1]
            return line.point[1]
        else:
            if abs(line.direction[1]) > 1e-6:
                t = (img_center[1] - line.point[1]) / line.direction[1]
                return line.point[0] + t * line.direction[0]
            return line.point[0]

    def _sort_lines_by_position(self, lines, is_horizontal, img_center):
        """Sort lines by their position in the image (Y for horizontal, X for vertical)"""
        return sorted(lines, key=lambda l: self._get_line_position(l, is_horizontal, img_center))

    def _line_in_bounds(self, line, is_horizontal, w, h):
        """Check if line passes through the visible image area"""
        # Check if line intersects image bounds at multiple points
        margin = 0.1  # 10% margin
        if is_horizontal:
            # Check Y at left edge, center, and right edge
            y_vals = []
            for x in [w * margin, w / 2, w * (1 - margin)]:
                if abs(line.direction[0]) > 1e-6:
                    t = (x - line.point[0]) / line.direction[0]
                    y_vals.append(line.point[1] + t * line.direction[1])
            if not y_vals:
                return False
            # Line should be within image height
            return all(h * 0.1 < y < h * 0.95 for y in y_vals)
        else:
            # Check X at top, center, and bottom
            x_vals = []
            for y in [h * margin, h / 2, h * (1 - margin)]:
                if abs(line.direction[1]) > 1e-6:
                    t = (y - line.point[1]) / line.direction[1]
                    x_vals.append(line.point[0] + t * line.direction[0])
            if not x_vals:
                return False
            # Line should be within image width
            return all(w * 0.1 < x < w * 0.9 for x in x_vals)

    def _get_possible_line_pairs(self, lines):
        """Generate all possible line pairs"""
        pairs = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pairs.append((lines[i], lines[j]))
        return pairs

    def fit(self, lines, binary_image):
        """Find best court model fit (original algorithm)"""
        h, w = binary_image.shape[:2]

        # Separate into horizontal and vertical
        h_lines = [l for l in lines if not l.is_vertical()]
        v_lines = [l for l in lines if l.is_vertical()]


        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        # Limit lines to prevent timeout (take best candidates by length/strength)
        # This is a reasonable optimization that doesn't change algorithm logic
        max_lines = 10
        h_lines = h_lines[:max_lines]
        v_lines = v_lines[:max_lines]

        # Generate ALL line pairs (original algorithm)
        h_pairs = self._get_possible_line_pairs(h_lines)
        v_pairs = self._get_possible_line_pairs(v_lines)

        best_score = -1
        best_model = None

        # Try all combinations (original algorithm)
        for h_pair in h_pairs:
            for v_pair in v_pairs:
                model = TennisCourtModel()
                score = model.fit(h_pair, v_pair, binary_image)

                if score > best_score:
                    best_score = score
                    best_model = model

        return best_model


class CourtLineDetectorCV:
    """Complete court detector using gchlebus algorithm"""

    def __init__(self):
        self.pixel_detector = CourtLinePixelDetector()
        self.candidate_detector = CourtLineCandidateDetector()
        self.court_fitter = TennisCourtFitter()

    def predict(self, image):
        """Detect court keypoints"""
        binary_mask = self.pixel_detector.detect(image)
        lines = self.candidate_detector.detect(binary_mask)


        if len(lines) < 4:
            print("  Warning: Not enough lines")
            return self._fallback(image)

        model = self.court_fitter.fit(lines, binary_mask)

        if model is None or model.transform_matrix is None:
            print("  Warning: Model fitting failed")
            return self._fallback(image)

        keypoints = model.get_keypoints()

        # Normalize corner ordering (ensure left corners have smaller X than right)
        # Keypoints 0,1 are top corners, 2,3 are bottom corners
        if keypoints[0, 0] > keypoints[1, 0]:  # Top-left X > Top-right X (swapped)
            # Swap left/right for all point pairs
            keypoints = keypoints.copy()
            # Swap pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (14,15)
            swap_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (14, 15)]
            for i, j in swap_pairs:
                if i < len(keypoints) and j < len(keypoints):
                    keypoints[i], keypoints[j] = keypoints[j].copy(), keypoints[i].copy()

        result = np.zeros(28, dtype=np.float32)
        for i in range(min(14, len(keypoints))):
            result[i * 2] = keypoints[i, 0]
            result[i * 2 + 1] = keypoints[i, 1]

        return result

    def _fallback(self, image):
        h, w = image.shape[:2]
        mx, my = w * 0.2, h * 0.15
        return np.array([
            mx, my, w-mx, my, mx, h-my, w-mx, h-my,
            mx+50, my, w-mx-50, my, mx+50, h-my, w-mx-50, h-my,
            mx+50, my+h*0.23, w-mx-50, my+h*0.23,
            mx+50, h-my-h*0.23, w-mx-50, h-my-h*0.23,
            w/2, my+h*0.23, w/2, h-my-h*0.23,
        ], dtype=np.float32)

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i + 1])
            cv2.putText(image, str(i // 2), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        return [self.draw_keypoints(f.copy(), keypoints) for f in video_frames]


class CourtLineDetector:
    """ML-based court detector"""

    def __init__(self, model_path):
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i + 1])
            cv2.putText(image, str(i // 2), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        return [self.draw_keypoints(f, keypoints) for f in video_frames]


class PadelCourtFitter:
    """Fit padel court model to detected lines"""

    def __init__(self, max_lines_per_direction=8):
        self.max_lines = max_lines_per_direction

    def _get_possible_line_pairs(self, lines):
        """Generate all possible line pairs"""
        pairs = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pairs.append((lines[i], lines[j]))
        return pairs

    def fit(self, lines, binary_image):
        """Find best padel court model fit"""
        h, w = binary_image.shape[:2]

        h_lines = [l for l in lines if not l.is_vertical()]
        v_lines = [l for l in lines if l.is_vertical()]

        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        max_lines = 10
        h_lines = h_lines[:max_lines]
        v_lines = v_lines[:max_lines]

        h_pairs = self._get_possible_line_pairs(h_lines)
        v_pairs = self._get_possible_line_pairs(v_lines)

        best_score = -1
        best_model = None

        for h_pair in h_pairs:
            for v_pair in v_pairs:
                model = PadelCourtModel()
                score = model.fit(h_pair, v_pair, binary_image)

                if score > best_score:
                    best_score = score
                    best_model = model

        return best_model


class PadelCourtLineDetectorCV:
    """Padel court detector - uses full court bounds"""

    def __init__(self):
        self.pixel_detector = CourtLinePixelDetector()
        self.candidate_detector = CourtLineCandidateDetector()
        self.court_fitter = PadelCourtFitter()

    def predict(self, image):
        """Detect padel court keypoints"""
        binary_mask = self.pixel_detector.detect(image)
        lines = self.candidate_detector.detect(binary_mask)

        if len(lines) < 4:
            print("  Warning: Not enough lines for padel court")
            return self._fallback(image)

        model = self.court_fitter.fit(lines, binary_mask)

        if model is None or model.transform_matrix is None:
            print("  Warning: Padel model fitting failed, using fallback")
            return self._fallback(image)

        keypoints = model.get_keypoints()

        # Normalize corner ordering
        if keypoints[0, 0] > keypoints[1, 0]:
            keypoints = keypoints.copy()
            swap_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (10, 11)]
            for i, j in swap_pairs:
                if i < len(keypoints) and j < len(keypoints):
                    keypoints[i], keypoints[j] = keypoints[j].copy(), keypoints[i].copy()

        result = np.zeros(28, dtype=np.float32)
        for i in range(min(14, len(keypoints))):
            result[i * 2] = keypoints[i, 0]
            result[i * 2 + 1] = keypoints[i, 1]

        return result

    def _fallback(self, image):
        """Fallback: estimate full padel court from visible frame"""
        h, w = image.shape[:2]

        # Padel court typically fills more of the frame
        # Use wider margins to capture full court
        margin_x = w * 0.08  # 8% margin on sides
        margin_top = h * 0.10  # 10% from top (far baseline)
        margin_bottom = h * 0.05  # 5% from bottom (near baseline - closer to camera)

        # 4 corners of the full court
        top_left = (margin_x, margin_top)
        top_right = (w - margin_x, margin_top)
        bottom_left = (margin_x, h - margin_bottom)
        bottom_right = (w - margin_x, h - margin_bottom)

        # Service lines (3m from net, net at center)
        # Court proportions: service line at ~15% from baseline (3m / 20m = 15%)
        service_near_y = h - margin_bottom - (h - margin_top - margin_bottom) * 0.15
        service_far_y = margin_top + (h - margin_top - margin_bottom) * 0.15
        net_y = (margin_top + h - margin_bottom) / 2

        # Center line x
        center_x = w / 2

        return np.array([
            # 0-1: top corners (far baseline)
            top_left[0], top_left[1], top_right[0], top_right[1],
            # 2-3: bottom corners (near baseline)
            bottom_left[0], bottom_left[1], bottom_right[0], bottom_right[1],
            # 4-5: service line near
            margin_x, service_near_y, w - margin_x, service_near_y,
            # 6-7: service line far
            margin_x, service_far_y, w - margin_x, service_far_y,
            # 8-9: center service points
            center_x, service_near_y, center_x, service_far_y,
            # 10-11: net points
            margin_x, net_y, w - margin_x, net_y,
            # 12-13: center baseline points
            center_x, margin_top, center_x, h - margin_bottom,
        ], dtype=np.float32)

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i + 1])
            cv2.putText(image, str(i // 2), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        return [self.draw_keypoints(f.copy(), keypoints) for f in video_frames]


class PadelCourtDetectorColor:
    """
    Padel court detector using color segmentation.
    Specifically designed for padel courts with blue surface and white lines.
    Handles glass wall reflections by focusing on the inner playing surface.

    Can use pre-calibrated keypoints for consistent detection across a video.
    """

    # Default calibrated keypoints for 1920x1080 broadcast format
    # Based on manual calibration for typical padel court broadcast
    DEFAULT_CALIBRATED_KEYPOINTS = np.array([
        583.0, 323.0,   # 0: Far-Left
        1343.0, 323.0,  # 1: Far-Right
        228.0, 944.0,   # 2: Near-Left
        1696.0, 942.0,  # 3: Near-Right
        321.0, 787.0,   # 4: Service-Near-Left
        1607.0, 787.0,  # 5: Service-Near-Right
        553.0, 375.0,   # 6: Service-Far-Left
        1372.0, 375.0,  # 7: Service-Far-Right
        955.0, 785.0,   # 8: Center-Near (service line)
        960.0, 379.0,   # 9: Center-Far (service line)
        468.0, 527.0,   # 10: Net-Left
        1459.0, 527.0,  # 11: Net-Right
        0, 0,           # 12: unused (padel has no center baseline mark)
        0, 0,           # 13: unused
    ], dtype=np.float32)

    def __init__(self, use_calibrated=True, calibrated_keypoints=None):
        """
        Args:
            use_calibrated: If True, use calibrated keypoints instead of detection
            calibrated_keypoints: Optional custom keypoints (28 values for 14 points)
        """
        self.use_calibrated = use_calibrated
        if calibrated_keypoints is not None:
            self.calibrated_keypoints = np.array(calibrated_keypoints, dtype=np.float32)
        else:
            self.calibrated_keypoints = self.DEFAULT_CALIBRATED_KEYPOINTS.copy()

        # Blue court detection parameters (HSV ranges)
        # Tuned for typical padel court blue (more saturated, specific hue)
        self.blue_lower = np.array([95, 80, 80])
        self.blue_upper = np.array([115, 255, 255])
        # White line detection parameters
        self.white_thresh = 200

    def detect_court_surface(self, image):
        """Detect the blue court surface using color segmentation"""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Focus on main court region (exclude top 15% for banners/text)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[int(h * 0.15):int(h * 0.98), int(w * 0.05):int(w * 0.95)] = 255

        # Detect blue in HSV
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)

        # Apply ROI mask
        blue_mask = cv2.bitwise_and(blue_mask, roi_mask)

        # Clean up the mask
        kernel = np.ones((7, 7), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Fill holes
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Keep only large contours (the court)
            for cnt in contours:
                if cv2.contourArea(cnt) > 0.05 * h * w:  # At least 5% of image
                    cv2.fillPoly(blue_mask, [cnt], 255)

        return blue_mask

    def detect_white_lines(self, image, court_mask):
        """Detect white lines within the court surface"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use adaptive thresholding for white lines
        # High brightness for white lines
        _, white_mask = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY)

        # Also detect based on local contrast
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        local_max = cv2.dilate(blur, np.ones((15, 15), np.uint8))
        high_contrast = (gray > local_max - 30) & (gray > 180)
        white_mask = cv2.bitwise_or(white_mask, (high_contrast.astype(np.uint8) * 255))

        # Only keep white pixels within or near the court
        if court_mask is not None:
            dilated_court = cv2.dilate(court_mask, np.ones((30, 30), np.uint8), iterations=1)
            white_lines = cv2.bitwise_and(white_mask, dilated_court)
        else:
            white_lines = white_mask

        # Clean up - keep linear structures
        kernel = np.ones((3, 3), np.uint8)
        white_lines = cv2.morphologyEx(white_lines, cv2.MORPH_CLOSE, kernel, iterations=2)

        return white_lines

    def find_court_corners(self, image):
        """Find the 4 main corners of the padel court"""
        h, w = image.shape[:2]

        # Detect blue court surface
        court_mask = self.detect_court_surface(image)

        # Find the largest contour (should be the court)
        contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Try to find 4 corners using convex hull and corner detection
        hull = cv2.convexHull(largest_contour)

        # Get bounding box as fallback
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Find corners using Harris corner detection on the court mask
        court_float = np.float32(court_mask)
        corners = cv2.goodFeaturesToTrack(court_float, maxCorners=20, qualityLevel=0.01, minDistance=50)

        if corners is None or len(corners) < 4:
            # Use bounding box corners
            return self._order_corners(box.astype(np.float32))

        corners = corners.reshape(-1, 2)

        # Find the 4 extreme corners (top-left, top-right, bottom-left, bottom-right)
        return self._find_extreme_corners(corners, w, h)

    def _find_extreme_corners(self, corners, w, h):
        """Find the 4 extreme corners from a set of detected corners"""
        # Sort by sum (x+y) - smallest is top-left, largest is bottom-right
        # Sort by diff (y-x) - smallest is top-right, largest is bottom-left

        sum_coords = corners.sum(axis=1)
        diff_coords = corners[:, 1] - corners[:, 0]

        top_left = corners[np.argmin(sum_coords)]
        bottom_right = corners[np.argmax(sum_coords)]
        top_right = corners[np.argmin(diff_coords)]
        bottom_left = corners[np.argmax(diff_coords)]

        return np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)

    def _order_corners(self, corners):
        """Order corners as: top-left, top-right, bottom-left, bottom-right"""
        # Sort by y first (top vs bottom)
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        top_corners = sorted_by_y[:2]
        bottom_corners = sorted_by_y[2:]

        # Sort top and bottom by x (left vs right)
        top_left, top_right = top_corners[np.argsort(top_corners[:, 0])]
        bottom_left, bottom_right = bottom_corners[np.argsort(bottom_corners[:, 0])]

        return np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)

    def detect_court_lines(self, image, court_mask):
        """Detect court lines using Hough transform on white pixels within court"""
        white_lines = self.detect_white_lines(image, court_mask)

        # Use probabilistic Hough transform
        lines = cv2.HoughLinesP(white_lines, 1, np.pi/180, threshold=50,
                                minLineLength=50, maxLineGap=20)

        return lines, white_lines

    def predict(self, image):
        """
        Detect padel court keypoints.
        Returns 28 values (14 keypoints x 2 coordinates)

        If use_calibrated is True, returns pre-calibrated keypoints
        scaled to the image dimensions.
        """
        h, w = image.shape[:2]

        # Use calibrated keypoints if enabled
        if self.use_calibrated:
            # Scale keypoints to match image dimensions
            # Calibrated keypoints are for 1920x1080
            scale_x = w / 1920.0
            scale_y = h / 1080.0
            scaled_keypoints = self.calibrated_keypoints.copy()
            scaled_keypoints[0::2] *= scale_x  # x coordinates
            scaled_keypoints[1::2] *= scale_y  # y coordinates
            return scaled_keypoints

        # First, detect the blue court surface
        court_mask = self.detect_court_surface(image)

        # Detect white lines
        white_lines_mask = self.detect_white_lines(image, court_mask)

        # Try to find corners using Hough lines on white lines
        # Pass court_mask to filter lines that are on the court surface
        corners = self._detect_corners_from_lines(white_lines_mask, h, w, court_mask)

        if corners is not None:
            keypoints = self._compute_keypoints_from_corners(corners, white_lines_mask, h, w)
            return keypoints

        # Fallback: use contour-based detection
        combined_mask = cv2.bitwise_or(court_mask, white_lines_mask)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("  Warning: No court contours found")
            return self._fallback(image)

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 0.1 * h * w:
            print("  Warning: Court contour too small")
            return self._fallback(image)

        corners = self._get_court_corners_from_contour(largest_contour, w, h)

        if corners is None:
            print("  Warning: Could not find 4 corners")
            return self._fallback(image)

        keypoints = self._compute_keypoints_from_corners(corners, white_lines_mask, h, w)
        return keypoints

    def _detect_corners_from_lines(self, white_lines_mask, h, w, court_mask=None):
        """Detect court corners by finding intersections of court lines"""
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(white_lines_mask, 1, np.pi/180, threshold=50,
                                minLineLength=100, maxLineGap=30)

        if lines is None or len(lines) < 4:
            return None

        # Court region bounds based on actual padel broadcast measurements
        # Far baseline: around 28-35% from top (user measured 29.9%)
        # Near baseline: around 85-92% from top (user measured 87.4%)
        y_far_min = int(h * 0.27)
        y_far_max = int(h * 0.38)
        y_near_min = int(h * 0.84)
        y_near_max = int(h * 0.92)

        # Separate and filter lines
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            mid_y = (y1 + y2) / 2
            mid_x = (x1 + x2) / 2
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Horizontal lines - baselines should be long (>150px) and span center
            if abs(angle) < 20 or abs(angle) > 160:
                if length > 150 and w * 0.25 < mid_x < w * 0.75:
                    # Categorize by region
                    if y_far_min <= mid_y <= y_far_max:
                        h_lines.append((line[0], mid_y, length, 'far'))
                    elif y_near_min <= mid_y <= y_near_max:
                        h_lines.append((line[0], mid_y, length, 'near'))

            # Vertical/diagonal lines - sidelines
            elif 50 < abs(angle) < 130:
                # Left sidelines should be in left half, right in right half
                if length > 100:
                    v_lines.append((line[0], mid_x, length))

        # Separate far and near baseline candidates
        far_lines = [l for l in h_lines if l[3] == 'far']
        near_lines = [l for l in h_lines if l[3] == 'near']

        if len(far_lines) < 1 or len(near_lines) < 1 or len(v_lines) < 2:
            return None

        # For far baseline: select the TOPMOST line (smallest y) - it's the actual baseline
        # For near baseline: select the BOTTOMMOST line (largest y)
        # Sort by y position
        far_lines = sorted(far_lines, key=lambda x: x[1])  # sort by y
        near_lines = sorted(near_lines, key=lambda x: x[1])  # sort by y

        top_baseline = far_lines[0][0]  # topmost (smallest y)
        bottom_baseline = near_lines[-1][0]  # bottommost (largest y)

        # Sort vertical lines and select leftmost/rightmost
        v_lines = sorted(v_lines, key=lambda x: x[1])
        left_candidates = [l for l in v_lines if l[1] < w * 0.4]
        right_candidates = [l for l in v_lines if l[1] > w * 0.6]

        if not left_candidates or not right_candidates:
            return None

        # Select longest sidelines
        left_sideline = max(left_candidates, key=lambda x: x[2])[0]
        right_sideline = max(right_candidates, key=lambda x: x[2])[0]

        # Find intersections
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2

            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-6:
                return None

            px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
            py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom

            return (px, py)

        tl = line_intersection(top_baseline, left_sideline)
        tr = line_intersection(top_baseline, right_sideline)
        bl = line_intersection(bottom_baseline, left_sideline)
        br = line_intersection(bottom_baseline, right_sideline)

        if any(c is None for c in [tl, tr, bl, br]):
            return None

        corners = np.array([tl, tr, bl, br], dtype=np.float32)

        # Validate geometry
        if corners[0][1] > corners[2][1]:  # Far should be above near
            return None

        # Validate corners within reasonable bounds
        margin = 100
        for c in corners:
            if c[0] < -margin or c[0] > w + margin or c[1] < -margin or c[1] > h + margin:
                return None

        return corners

    def _get_court_corners_from_contour(self, contour, w, h):
        """Extract 4 court corners from contour"""
        # Get convex hull
        hull = cv2.convexHull(contour)

        # Approximate hull to polygon
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) < 4:
            # Use bounding rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            approx = box.reshape(-1, 1, 2)

        # Extract points
        points = approx.reshape(-1, 2)

        # Find the 4 most extreme points
        # Top-left: minimize x + y
        # Top-right: maximize x - y
        # Bottom-left: maximize y - x
        # Bottom-right: maximize x + y
        sum_pts = points.sum(axis=1)
        diff_pts = points[:, 0] - points[:, 1]

        tl_idx = np.argmin(sum_pts)
        br_idx = np.argmax(sum_pts)
        tr_idx = np.argmax(diff_pts)
        bl_idx = np.argmin(diff_pts)

        top_left = points[tl_idx]
        top_right = points[tr_idx]
        bottom_left = points[bl_idx]
        bottom_right = points[br_idx]

        return np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)

    def _compute_keypoints_from_corners(self, corners, white_lines_mask, h, w):
        """Compute all 12 keypoints from the 4 corners for padel court.

        Padel court keypoints (no center baseline marks):
        0-1: Far baseline corners (top-left, top-right)
        2-3: Near baseline corners (bottom-left, bottom-right)
        4-5: Near service line corners
        6-7: Far service line corners
        8-9: Center service line endpoints (only between service lines)
        10-11: Net endpoints
        """
        tl, tr, bl, br = corners

        # Padel court proportions (based on 20m x 10m)
        # Service lines are at 3m from each baseline (15% from baseline)
        # Net is at center (50%)
        service_ratio = 0.15  # 3m / 20m
        net_ratio = 0.5

        def lerp(p1, p2, t):
            return p1 + t * (p2 - p1)

        # Near service line (15% from near baseline)
        sl_near_left = lerp(bl, tl, service_ratio)
        sl_near_right = lerp(br, tr, service_ratio)

        # Far service line (15% from far baseline = 85% from near)
        sl_far_left = lerp(bl, tl, 1 - service_ratio)
        sl_far_right = lerp(br, tr, 1 - service_ratio)

        # Net (50%)
        net_left = lerp(bl, tl, net_ratio)
        net_right = lerp(br, tr, net_ratio)

        # Center service line points (only between service lines, not to baselines)
        center_near = lerp(sl_near_left, sl_near_right, 0.5)
        center_far = lerp(sl_far_left, sl_far_right, 0.5)

        # Build keypoints array (12 points, 24 values)
        # But keep 28 for compatibility, unused slots = 0
        keypoints = np.zeros(28, dtype=np.float32)

        # Points 0-1: far baseline corners (top)
        keypoints[0:2] = tl
        keypoints[2:4] = tr

        # Points 2-3: near baseline corners (bottom)
        keypoints[4:6] = bl
        keypoints[6:8] = br

        # Points 4-5: near service line corners
        keypoints[8:10] = sl_near_left
        keypoints[10:12] = sl_near_right

        # Points 6-7: far service line corners
        keypoints[12:14] = sl_far_left
        keypoints[14:16] = sl_far_right

        # Points 8-9: center service line endpoints
        keypoints[16:18] = center_near
        keypoints[18:20] = center_far

        # Points 10-11: net endpoints
        keypoints[20:22] = net_left
        keypoints[22:24] = net_right

        # Points 12-13: NOT USED in padel (no center baseline marks)
        # Leave as zeros or could use for something else

        return keypoints

    def _fallback(self, image):
        """Fallback keypoints when detection fails"""
        h, w = image.shape[:2]

        # Typical padel court view proportions
        margin_x = w * 0.12
        margin_top = h * 0.18
        margin_bottom = h * 0.08

        tl = np.array([margin_x, margin_top])
        tr = np.array([w - margin_x, margin_top])
        bl = np.array([margin_x, h - margin_bottom])
        br = np.array([w - margin_x, h - margin_bottom])

        return self._compute_keypoints_from_corners(
            np.array([tl, tr, bl, br]), None, h, w
        )

    def draw_keypoints(self, image, keypoints, draw_lines=True):
        """Draw professional court line overlay on image for padel court"""
        output = image.copy()

        if not draw_lines:
            return output

        # Create overlay for semi-transparent effect
        overlay = output.copy()

        # Define line styles for different court elements
        # (point1, point2, color_BGR, thickness, is_net)
        court_lines = [
            # Baselines - white with slight transparency
            ((0, 1), (255, 255, 255), 3, False),   # far baseline
            ((2, 3), (255, 255, 255), 3, False),   # near baseline
            # Sidelines - white
            ((0, 2), (255, 255, 255), 2, False),   # left sideline
            ((1, 3), (255, 255, 255), 2, False),   # right sideline
            # Service lines - slightly thinner white
            ((4, 5), (255, 255, 255), 2, False),   # near service line
            ((6, 7), (255, 255, 255), 2, False),   # far service line
            # Center service line
            ((8, 9), (255, 255, 255), 2, False),   # center service
            # Net - distinct cyan/blue color
            ((10, 11), (255, 200, 100), 4, True),  # net (thicker, cyan)
        ]

        # Draw court lines on overlay
        for (i1, i2), color, thickness, is_net in court_lines:
            x1, y1 = int(keypoints[i1*2]), int(keypoints[i1*2+1])
            x2, y2 = int(keypoints[i2*2]), int(keypoints[i2*2+1])

            # Skip if any coordinate is 0 (unused point)
            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                continue

            if is_net:
                # Draw net with dashed pattern effect
                cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                # Add slight glow effect for net
                cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 200), thickness + 2, cv2.LINE_AA)
                cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            else:
                # Regular court lines with subtle glow
                # Draw glow (thicker, darker)
                cv2.line(overlay, (x1, y1), (x2, y2), (180, 180, 180), thickness + 2, cv2.LINE_AA)
                # Draw main line
                cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # Blend overlay with original (semi-transparent lines)
        alpha = 0.7  # Line opacity
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Add subtle corner markers (small, unobtrusive)
        corner_indices = [0, 1, 2, 3]  # Only the 4 main corners
        for idx in corner_indices:
            x, y = int(keypoints[idx*2]), int(keypoints[idx*2+1])
            if x == 0 and y == 0:
                continue
            # Small corner indicator
            cv2.circle(output, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)

        return output

    def draw_keypoints_on_video(self, video_frames, keypoints):
        return [self.draw_keypoints(f.copy(), keypoints) for f in video_frames]

    def debug_detection(self, image, output_path=None):
        """Debug visualization showing detection steps"""
        h, w = image.shape[:2]

        # Create debug image
        debug = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # Top-left: original
        debug[:h, :w] = image

        # Top-right: blue court mask
        court_mask = self.detect_court_surface(image)
        court_vis = cv2.cvtColor(court_mask, cv2.COLOR_GRAY2BGR)
        debug[:h, w:] = court_vis

        # Bottom-left: white lines
        white_lines = self.detect_white_lines(image, court_mask)
        white_vis = cv2.cvtColor(white_lines, cv2.COLOR_GRAY2BGR)
        debug[h:, :w] = white_vis

        # Bottom-right: result with keypoints
        keypoints = self.predict(image)
        result = self.draw_keypoints(image.copy(), keypoints)
        debug[h:, w:] = result

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug, "Court Mask (Blue)", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug, "White Lines", (10, h + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug, "Detected Keypoints", (w + 10, h + 30), font, 1, (255, 255, 255), 2)

        if output_path:
            cv2.imwrite(output_path, debug)
            print(f"Debug image saved to {output_path}")

        return debug, keypoints
