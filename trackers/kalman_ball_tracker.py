"""
Kalman Filter Ball Tracker for Padel

Implements the PDF recommendations:
1. Extended Kalman Filter for smoothing and occlusion prediction
2. Mahalanobis distance gating for outlier rejection
3. Cubic Spline interpolation for gap filling
4. Event-aware segmentation (don't interpolate across hits/bounces)

Based on: "Padel AI: Tracking, Bounces, and Shots" - Section 3
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2


@dataclass
class BallState:
    """Ball state vector: position, velocity, acceleration"""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 9.8 * 50  # Gravity in pixels/frame^2 (approximate)


class KalmanBallFilter:
    """
    Extended Kalman Filter for ball tracking.

    State vector: [x, y, vx, vy]
    Measurement: [x, y]

    Includes:
    - Constant acceleration motion model (gravity)
    - Mahalanobis gating for outlier rejection
    - Coasting mode during occlusions
    """

    def __init__(self,
                 process_noise: float = 5.0,
                 measurement_noise: float = 10.0,
                 gate_threshold: float = 9.21):  # Chi-squared 99% for 2 DOF
        """
        Args:
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
            gate_threshold: Mahalanobis distance threshold for gating
        """
        # State dimension (x, y, vx, vy)
        self.state_dim = 4
        self.meas_dim = 2

        # State vector [x, y, vx, vy]
        self.x = np.zeros(self.state_dim)

        # State covariance
        self.P = np.eye(self.state_dim) * 1000  # High initial uncertainty

        # Process noise covariance
        self.Q = np.eye(self.state_dim) * process_noise**2
        self.Q[2, 2] *= 2  # Higher uncertainty in velocity
        self.Q[3, 3] *= 2

        # Measurement noise covariance
        self.R = np.eye(self.meas_dim) * measurement_noise**2

        # Measurement matrix (we observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # Gate threshold for Mahalanobis distance
        self.gate_threshold = gate_threshold

        # Track state
        self.initialized = False
        self.frames_since_update = 0
        self.max_coast_frames = 15  # Max frames to coast without measurement

        # Gravity (approximate, in pixels/frame^2)
        # At 30fps, g ≈ 9.8 m/s^2 ≈ 0.01 m/frame^2
        # With ~50 pixels/meter, gravity ≈ 0.5 pixels/frame^2
        self.gravity = 0.5

    def _get_transition_matrix(self, dt: float = 1.0) -> np.ndarray:
        """State transition matrix F for constant velocity model"""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        return F

    def _get_control_vector(self, dt: float = 1.0) -> np.ndarray:
        """Control vector for gravity (affects y velocity)"""
        # Gravity acceleration in y direction
        B = np.array([0, 0.5 * self.gravity * dt**2, 0, self.gravity * dt])
        return B

    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Predict next state"""
        F = self._get_transition_matrix(dt)
        B = self._get_control_vector(dt)

        # State prediction with gravity
        self.x = F @ self.x + B

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

        self.frames_since_update += 1

        return self.x[:2].copy()  # Return predicted position

    def update(self, measurement: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Update state with measurement if it passes gating.

        Returns:
            (accepted, state) - whether measurement was accepted and current state
        """
        if not self.initialized:
            # Initialize with first measurement
            self.x[0] = measurement[0]
            self.x[1] = measurement[1]
            self.x[2] = 0  # Initial velocity unknown
            self.x[3] = 0
            self.initialized = True
            self.frames_since_update = 0
            return True, self.x[:2].copy()

        # Compute innovation (measurement residual)
        z = np.array(measurement)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Mahalanobis distance for gating
        try:
            S_inv = np.linalg.inv(S)
            mahal_dist = y.T @ S_inv @ y
        except np.linalg.LinAlgError:
            mahal_dist = float('inf')

        # Gate check
        if mahal_dist > self.gate_threshold:
            # Measurement rejected as outlier
            return False, self.x[:2].copy()

        # Kalman gain
        K = self.P @ self.H.T @ S_inv

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self.frames_since_update = 0

        return True, self.x[:2].copy()

    def is_coasting(self) -> bool:
        """Check if filter is in coasting mode (no recent measurements)"""
        return self.frames_since_update > 0

    def is_lost(self) -> bool:
        """Check if track should be terminated"""
        return self.frames_since_update > self.max_coast_frames

    def get_state(self) -> np.ndarray:
        """Get current state estimate"""
        return self.x.copy()

    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate"""
        return (self.x[2], self.x[3])

    def reset(self):
        """Reset filter state"""
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1000
        self.initialized = False
        self.frames_since_update = 0


class PhysicsAwareInterpolator:
    """
    Cubic Spline interpolation with physics awareness.

    - Uses cubic splines for smooth parabolic trajectories
    - Segments trajectory at detected events (hits/bounces)
    - Never interpolates across direction changes
    """

    def __init__(self, max_gap_frames: int = 30):
        """
        Args:
            max_gap_frames: Maximum gap to interpolate (larger gaps likely cross events)
        """
        self.max_gap_frames = max_gap_frames

    def detect_events(self,
                     positions: List[Tuple[Optional[float], Optional[float]]],
                     angle_threshold: float = 90.0) -> List[int]:
        """
        Detect trajectory events (direction changes) that shouldn't be interpolated across.

        Returns:
            List of frame indices where events occur
        """
        events = []

        # Compute velocities
        velocities = []
        for i in range(1, len(positions)):
            if positions[i][0] is not None and positions[i-1][0] is not None:
                vx = positions[i][0] - positions[i-1][0]
                vy = positions[i][1] - positions[i-1][1]
                velocities.append((i, vx, vy))
            else:
                velocities.append((i, None, None))

        # Find direction changes
        for i in range(1, len(velocities)):
            curr = velocities[i]
            prev = velocities[i-1]

            if curr[1] is None or prev[1] is None:
                continue

            # Compute angle change
            v1 = np.array([prev[1], prev[2]])
            v2 = np.array([curr[1], curr[2]])

            mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if mag1 < 1 or mag2 < 1:
                continue

            cos_angle = np.dot(v1, v2) / (mag1 * mag2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if angle > angle_threshold:
                events.append(curr[0])

        return events

    def interpolate(self,
                   positions: List[Tuple[Optional[float], Optional[float]]],
                   events: Optional[List[int]] = None) -> List[Tuple[float, float]]:
        """
        Interpolate missing positions using cubic splines.

        Args:
            positions: List of (x, y) or (None, None) for each frame
            events: Optional list of event frames (don't interpolate across these)

        Returns:
            List of interpolated (x, y) positions
        """
        n = len(positions)
        result = list(positions)

        # Detect events if not provided
        if events is None:
            events = self.detect_events(positions)

        # Add boundaries
        event_frames = [0] + sorted(events) + [n]

        # Interpolate within each segment
        for seg_start, seg_end in zip(event_frames[:-1], event_frames[1:]):
            self._interpolate_segment(result, seg_start, seg_end)

        return result

    def _interpolate_segment(self,
                            positions: List[Tuple[Optional[float], Optional[float]]],
                            start: int, end: int):
        """Interpolate within a single segment using cubic splines"""
        # Collect valid points in segment
        valid_frames = []
        valid_x = []
        valid_y = []

        for i in range(start, end):
            if positions[i][0] is not None:
                valid_frames.append(i)
                valid_x.append(positions[i][0])
                valid_y.append(positions[i][1])

        if len(valid_frames) < 2:
            # Not enough points for interpolation
            return

        # Find gaps
        gaps = []
        for i in range(len(valid_frames) - 1):
            gap_start = valid_frames[i]
            gap_end = valid_frames[i + 1]
            gap_size = gap_end - gap_start - 1

            if gap_size > 0 and gap_size <= self.max_gap_frames:
                gaps.append((gap_start, gap_end))

        if not gaps:
            return

        # Create cubic splines
        try:
            cs_x = CubicSpline(valid_frames, valid_x)
            cs_y = CubicSpline(valid_frames, valid_y)
        except Exception:
            # Fall back to linear if spline fails
            return

        # Interpolate gaps
        for gap_start, gap_end in gaps:
            for frame in range(gap_start + 1, gap_end):
                x = float(cs_x(frame))
                y = float(cs_y(frame))
                positions[frame] = (x, y)


class KalmanBallTracker:
    """
    Complete ball tracking pipeline with:
    1. Kalman filtering for smoothing
    2. Mahalanobis gating for outlier rejection
    3. Cubic spline interpolation for gap filling
    4. Event-aware segmentation
    5. Pre-serve noise filtering
    """

    def __init__(self,
                 process_noise: float = 5.0,
                 measurement_noise: float = 10.0,
                 gate_threshold: float = 25.0,  # Higher = accept more measurements
                 max_interpolate_gap: int = 20,
                 filter_pre_serve: bool = True):
        """
        Args:
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
            gate_threshold: Mahalanobis distance threshold
            max_interpolate_gap: Max frames to interpolate
            filter_pre_serve: Filter out noisy pre-serve detections
        """
        self.kalman = KalmanBallFilter(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            gate_threshold=gate_threshold
        )
        self.interpolator = PhysicsAwareInterpolator(
            max_gap_frames=max_interpolate_gap
        )
        self.filter_pre_serve = filter_pre_serve

    def process_detections(self,
                          raw_detections: List[Dict],
                          verbose: bool = True) -> List[Dict]:
        """
        Process raw ball detections through full pipeline.

        Args:
            raw_detections: List of {1: [x1,y1,x2,y2]} dicts from detector
            verbose: Print statistics

        Returns:
            Smoothed and interpolated detections
        """
        n = len(raw_detections)

        # Step 0: Filter pre-serve noise if enabled
        if self.filter_pre_serve:
            play_start = self._detect_play_start(raw_detections)
            if play_start > 0 and verbose:
                print(f"Play starts at frame {play_start}, filtering {play_start} pre-serve frames")
            # Clear detections before play starts
            for i in range(play_start):
                raw_detections[i] = {}

        # Step 1: Extract positions and apply Kalman filter
        filtered_positions = []
        accepted_count = 0
        rejected_count = 0

        self.kalman.reset()

        for i, det in enumerate(raw_detections):
            # Predict (updates Kalman state)
            predicted = self.kalman.predict()

            if 1 in det and det[1]:
                bbox = det[1]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                # Update with measurement - use gating to reject outliers
                accepted, state = self.kalman.update(np.array([cx, cy]))

                if accepted:
                    # Use actual measurement (slightly smoothed by Kalman)
                    filtered_positions.append((cx, cy))  # Use raw measurement, not Kalman state
                    accepted_count += 1
                else:
                    # Outlier rejected - mark as gap (will be interpolated later)
                    filtered_positions.append((None, None))
                    rejected_count += 1
            else:
                # No detection - mark as gap (DON'T use Kalman prediction)
                filtered_positions.append((None, None))

        if verbose:
            print(f"Kalman filter: {accepted_count} accepted, {rejected_count} rejected (gating)")

        # Step 2: Detect events (direction changes) for segmentation
        events = self.interpolator.detect_events(filtered_positions)
        if verbose:
            print(f"Detected {len(events)} trajectory events (hits/bounces)")

        # Step 3: Cubic spline interpolation within segments
        interpolated = self.interpolator.interpolate(filtered_positions, events)

        if verbose:
            interp_count = sum(1 for i, (orig, interp) in enumerate(zip(filtered_positions, interpolated))
                               if orig[0] is None and interp[0] is not None)
            print(f"Cubic spline interpolated {interp_count} missing frames")

        # Step 4: Convert back to detection format
        result = []
        ball_radius = 10

        for pos in interpolated:
            if pos[0] is not None:
                x, y = pos
                result.append({1: [x - ball_radius, y - ball_radius,
                                   x + ball_radius, y + ball_radius]})
            else:
                result.append({})

        # Step 5: Minimal post-processing (no heavy smoothing to avoid lag)
        if verbose:
            print("Applying minimal post-processing...")

        # Extract positions
        final_positions = []
        for d in result:
            if d and 1 in d:
                bbox = d[1]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                final_positions.append((cx, cy))
            else:
                final_positions.append((None, None))

        # Only fix aggressive jumps (no smoothing to preserve ball speed)
        fixed_positions = self._fix_aggressive_jumps(final_positions, verbose=verbose)

        # Fill any remaining gaps
        final_filled = self._fill_gaps_linear(fixed_positions)

        # Reconstruct result
        smoothed_result = []
        for pos in final_filled:
            if pos[0] is not None:
                smoothed_result.append({1: [pos[0] - ball_radius, pos[1] - ball_radius,
                                            pos[0] + ball_radius, pos[1] + ball_radius]})
            else:
                smoothed_result.append({})

        result = smoothed_result

        # Final coverage
        coverage = sum(1 for d in result if d) / n * 100
        if verbose:
            print(f"Final coverage: {sum(1 for d in result if d)}/{n} frames ({coverage:.1f}%)")

        return result

    def _fix_aggressive_jumps(self, positions: List[Tuple], verbose: bool = False) -> List[Tuple]:
        """
        Detect and fix aggressive jumps by replacing outlier positions
        with interpolated values.
        """
        result = list(positions)
        n = len(result)
        fixed_count = 0

        # Compute velocities
        velocities = []
        for i in range(1, n):
            if result[i][0] is not None and result[i-1][0] is not None:
                vx = result[i][0] - result[i-1][0]
                vy = result[i][1] - result[i-1][1]
                speed = np.sqrt(vx**2 + vy**2)
                velocities.append(speed)
            else:
                velocities.append(None)

        # Compute median velocity for threshold
        valid_velocities = [v for v in velocities if v is not None]
        if not valid_velocities:
            return result

        median_vel = np.median(valid_velocities)
        # Aggressive jump threshold: 5x median velocity
        jump_threshold = max(median_vel * 5, 50)  # At least 50 pixels

        # Find and fix jumps
        for i in range(1, n - 1):
            if velocities[i - 1] is None:
                continue

            # Check for aggressive jump (velocity spike)
            if velocities[i - 1] > jump_threshold:
                # Check if it's a single-frame spike (next velocity also high = direction change, not noise)
                if i < len(velocities) and velocities[i] is not None:
                    # If both before and after are high, it might be a real direction change
                    if velocities[i] > jump_threshold:
                        continue  # Real event, don't fix

                # Single-frame spike - interpolate this position
                prev_valid = None
                next_valid = None

                # Find previous valid position
                for j in range(i - 1, -1, -1):
                    if result[j][0] is not None:
                        prev_valid = (j, result[j])
                        break

                # Find next valid position (skip current)
                for j in range(i + 1, n):
                    if result[j][0] is not None:
                        next_valid = (j, result[j])
                        break

                if prev_valid and next_valid:
                    # Linear interpolate
                    t = (i - prev_valid[0]) / (next_valid[0] - prev_valid[0])
                    new_x = prev_valid[1][0] + t * (next_valid[1][0] - prev_valid[1][0])
                    new_y = prev_valid[1][1] + t * (next_valid[1][1] - prev_valid[1][1])
                    result[i] = (new_x, new_y)
                    fixed_count += 1

        if verbose and fixed_count > 0:
            print(f"  Fixed {fixed_count} aggressive jumps")

        return result

    def _exponential_smooth(self, positions: List[Tuple], alpha: float = 0.4) -> List[Tuple]:
        """
        Apply bi-directional exponential moving average smoothing.
        Forward and backward passes are averaged to eliminate lag.
        """
        n = len(positions)

        # Forward pass
        forward = []
        smoothed_x = None
        smoothed_y = None

        for pos in positions:
            if pos[0] is None:
                forward.append((None, None))
                smoothed_x = None
                smoothed_y = None
                continue

            if smoothed_x is None:
                smoothed_x = pos[0]
                smoothed_y = pos[1]
            else:
                smoothed_x = alpha * pos[0] + (1 - alpha) * smoothed_x
                smoothed_y = alpha * pos[1] + (1 - alpha) * smoothed_y

            forward.append((smoothed_x, smoothed_y))

        # Backward pass
        backward = [(None, None)] * n
        smoothed_x = None
        smoothed_y = None

        for i in range(n - 1, -1, -1):
            pos = positions[i]
            if pos[0] is None:
                smoothed_x = None
                smoothed_y = None
                continue

            if smoothed_x is None:
                smoothed_x = pos[0]
                smoothed_y = pos[1]
            else:
                smoothed_x = alpha * pos[0] + (1 - alpha) * smoothed_x
                smoothed_y = alpha * pos[1] + (1 - alpha) * smoothed_y

            backward[i] = (smoothed_x, smoothed_y)

        # Average forward and backward
        result = []
        for i in range(n):
            if forward[i][0] is None or backward[i][0] is None:
                result.append((None, None))
            else:
                avg_x = (forward[i][0] + backward[i][0]) / 2
                avg_y = (forward[i][1] + backward[i][1]) / 2
                result.append((avg_x, avg_y))

        return result

    def _median_smooth(self, positions: List[Tuple], window: int = 3) -> List[Tuple]:
        """
        Apply median filter smoothing.
        """
        result = []
        n = len(positions)

        for i in range(n):
            if positions[i][0] is None:
                result.append((None, None))
                continue

            x_vals = []
            y_vals = []
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if positions[j][0] is not None:
                    x_vals.append(positions[j][0])
                    y_vals.append(positions[j][1])

            if x_vals:
                result.append((np.median(x_vals), np.median(y_vals)))
            else:
                result.append((None, None))

        return result

    def _detect_play_start(self, detections: List[Dict],
                           window_size: int = 30,
                           min_density: float = 0.5,
                           min_velocity: float = 10.0,
                           min_sustained_frames: int = 5) -> int:
        """
        Detect when actual play starts (after serve).

        Uses two criteria:
        1. Detection density increases (ball is consistently visible)
        2. Sustained ball movement (ball is traveling, not oscillating)

        Args:
            detections: Raw ball detections
            window_size: Window size for density calculation (frames)
            min_density: Minimum detection density to consider as play (0-1)
            min_velocity: Minimum ball velocity (pixels/frame)
            min_sustained_frames: Minimum consecutive frames of movement

        Returns:
            Frame index where play starts (0 if can't detect)
        """
        n = len(detections)

        # Extract positions
        positions = []
        for d in detections:
            if d and 1 in d:
                bbox = d[1]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                positions.append((cx, cy))
            else:
                positions.append(None)

        # Method 1: Find where detection density increases
        density_start = 0
        for start in range(0, n - window_size, window_size // 2):
            end = start + window_size
            count = sum(1 for p in positions[start:end] if p is not None)
            density = count / window_size
            if density >= min_density:
                density_start = start
                break

        # Method 2: Find sustained movement within the high-density region
        velocities = []
        for i in range(1, n):
            if positions[i] is not None and positions[i-1] is not None:
                vx = positions[i][0] - positions[i-1][0]
                vy = positions[i][1] - positions[i-1][1]
                speed = np.sqrt(vx**2 + vy**2)
                velocities.append((i, speed))
            else:
                velocities.append((i, 0))

        # Find sustained movement starting from density_start
        consecutive = 0
        for frame_idx, speed in velocities:
            if frame_idx < density_start:
                continue
            if speed >= min_velocity:
                consecutive += 1
                if consecutive >= min_sustained_frames:
                    play_start = frame_idx - min_sustained_frames + 1
                    return max(0, play_start)
            else:
                consecutive = 0

        # Fall back to density-based detection
        if density_start > 0:
            return density_start

        return 0

    def _linear_interpolate_all(self, positions: List[Tuple]) -> List[Tuple]:
        """
        Simple linear interpolation for all gaps.
        Same as _fill_gaps_linear but used earlier in the pipeline.
        """
        return self._fill_gaps_linear(positions)

    def _fill_gaps_linear(self, positions: List[Tuple]) -> List[Tuple]:
        """
        Fill remaining gaps with linear interpolation for 100% coverage.
        """
        result = list(positions)
        n = len(result)

        # Find first and last valid positions
        first_valid = None
        last_valid = None
        for i in range(n):
            if result[i][0] is not None:
                if first_valid is None:
                    first_valid = i
                last_valid = i

        if first_valid is None:
            return result  # No valid positions

        # Forward fill start
        for i in range(first_valid):
            result[i] = result[first_valid]

        # Backward fill end
        for i in range(last_valid + 1, n):
            result[i] = result[last_valid]

        # Linear interpolate internal gaps
        i = first_valid
        while i < last_valid:
            if result[i][0] is None:
                # Find gap boundaries
                gap_start = i - 1
                gap_end = i
                while gap_end < n and result[gap_end][0] is None:
                    gap_end += 1

                # Interpolate
                if gap_end < n:
                    for j in range(gap_start + 1, gap_end):
                        t = (j - gap_start) / (gap_end - gap_start)
                        x = result[gap_start][0] + t * (result[gap_end][0] - result[gap_start][0])
                        y = result[gap_start][1] + t * (result[gap_end][1] - result[gap_start][1])
                        result[j] = (x, y)
                    i = gap_end
                else:
                    break
            else:
                i += 1

        return result


def upgrade_ball_tracking(raw_detections: List[Dict], verbose: bool = True) -> List[Dict]:
    """
    Convenience function to upgrade raw TrackNet detections with Kalman + Spline.

    Args:
        raw_detections: Raw detections from TrackNet
        verbose: Print statistics

    Returns:
        Improved detections
    """
    tracker = KalmanBallTracker()
    return tracker.process_detections(raw_detections, verbose=verbose)


def hybrid_ball_tracking(raw_detections: List[Dict], verbose: bool = True) -> List[Dict]:
    """
    Hybrid approach combining:
    1. Old outlier filtering (bounds, stuck detection)
    2. Kalman smoothing
    3. Cubic spline interpolation

    This is the recommended approach.
    """
    from trackers.tracknet_ball_tracker import TrackNetBallTracker

    n = len(raw_detections)

    # Step 1: Apply old-style outlier filtering (bounds + stuck detection)
    if verbose:
        print("Step 1: Applying outlier filtering...")
    old_tracker = TrackNetBallTracker.__new__(TrackNetBallTracker)
    old_tracker.frame_buffer = []
    filtered = old_tracker._filter_ball_outliers(raw_detections.copy())
    filtered = old_tracker._filter_stuck_detections(filtered)

    filtered_count = sum(1 for d in filtered if d)
    if verbose:
        print(f"  After filtering: {filtered_count}/{n} detections")

    # Step 2: Kalman smoothing (no interpolation yet)
    if verbose:
        print("Step 2: Kalman smoothing...")

    kalman = KalmanBallFilter(
        process_noise=10.0,
        measurement_noise=5.0,
        gate_threshold=100.0  # Very permissive - just smooth, don't reject
    )

    smoothed_positions = []

    for i, det in enumerate(filtered):
        kalman.predict()

        if 1 in det and det[1]:
            bbox = det[1]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            _, state = kalman.update(np.array([cx, cy]))
            smoothed_positions.append((state[0], state[1]))
        else:
            smoothed_positions.append((None, None))

    # Step 3: Cubic spline interpolation
    if verbose:
        print("Step 3: Cubic spline interpolation...")

    interpolator = PhysicsAwareInterpolator(max_gap_frames=200)  # Allow larger gaps

    # Don't detect events - interpolate all gaps
    # (events were already handled by the original filtering)
    interpolated = interpolator.interpolate(smoothed_positions, events=[])

    # Forward/backward fill for boundary frames
    # Find first and last valid positions
    first_valid = None
    last_valid = None
    for i, pos in enumerate(interpolated):
        if pos[0] is not None:
            if first_valid is None:
                first_valid = i
            last_valid = i

    if first_valid is not None:
        # Backward fill (fill start with first valid)
        for i in range(first_valid):
            interpolated[i] = interpolated[first_valid]

        # Forward fill (fill end with last valid)
        for i in range(last_valid + 1, n):
            interpolated[i] = interpolated[last_valid]

    interp_count = sum(1 for orig, interp in zip(smoothed_positions, interpolated)
                      if orig[0] is None and interp[0] is not None)
    if verbose:
        print(f"  Interpolated {interp_count} frames")

    # Convert back to detection format
    result = []
    ball_radius = 10

    for pos in interpolated:
        if pos[0] is not None:
            x, y = pos
            result.append({1: [x - ball_radius, y - ball_radius,
                               x + ball_radius, y + ball_radius]})
        else:
            result.append({})

    coverage = sum(1 for d in result if d) / n * 100
    if verbose:
        print(f"Final coverage: {sum(1 for d in result if d)}/{n} ({coverage:.1f}%)")

    return result
