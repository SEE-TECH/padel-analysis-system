"""
Padel MVP Analyzer - Lightweight analysis system for padel videos

Features:
- V-shape velocity-based shot detection
- Shot attribution (which player hit the ball)
- Shot speed estimation
- Hybrid shot classification (trajectory + pose)
- Basic statistics (shots per player, rally lengths, heatmaps)

Based on the analysis document: "Padel AI: Tracking, Bounces, and Shots"
"""

import numpy as np
import cv2
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json

from .shot_classifier import ShotClassifier


@dataclass
class ShotEvent:
    """Represents a detected shot event"""
    frame_idx: int
    player_id: int  # 1-4, or 0 if unknown
    ball_position: Tuple[float, float]  # (x, y) in pixels
    velocity_before: Tuple[float, float]  # (vx, vy) pixels/frame
    velocity_after: Tuple[float, float]  # (vx, vy) pixels/frame
    speed_pixels: float  # ball speed in pixels/frame after hit
    speed_kmh: Optional[float] = None  # estimated real speed if homography available
    shot_type: str = "unknown"  # basic classification: overhead, ground, volley
    angle_change: float = 0.0  # direction change in degrees


@dataclass
class RallyStats:
    """Statistics for a single rally"""
    start_frame: int
    end_frame: int
    shot_count: int
    shots: List[ShotEvent] = field(default_factory=list)
    winner: Optional[int] = None  # player_id who won the rally


@dataclass
class MatchStats:
    """Overall match statistics"""
    total_shots: int = 0
    shots_by_player: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    avg_speed_by_player: Dict[int, float] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    shot_speeds: Dict[int, List[float]] = field(default_factory=lambda: {1: [], 2: [], 3: [], 4: []})
    rallies: List[RallyStats] = field(default_factory=list)
    avg_rally_length: float = 0.0
    player_positions: Dict[int, List[Tuple[float, float]]] = field(default_factory=lambda: {1: [], 2: [], 3: [], 4: []})


class PadelMVPAnalyzer:
    """
    MVP Padel Analysis System

    Uses V-shape velocity analysis for shot detection without requiring
    complex 3D reconstruction or physics simulation.
    """

    # Court dimensions in meters (standard padel court)
    COURT_LENGTH_M = 20.0
    COURT_WIDTH_M = 10.0

    # Default court corners in pixels (for 1920x1080, can be calibrated)
    # Full court corners (far baseline to near baseline = 20m)
    DEFAULT_COURT_CORNERS = np.array([
        [583, 323],    # Far-Left (net line left)
        [1343, 323],   # Far-Right (net line right)
        [1696, 942],   # Near-Right (baseline right)
        [228, 944],    # Near-Left (baseline left)
    ], dtype=np.float32)

    def __init__(self,
                 court_corners: Optional[np.ndarray] = None,
                 fps: float = 30.0,
                 frame_size: Tuple[int, int] = (1920, 1080)):
        """
        Initialize the analyzer.

        Args:
            court_corners: 4 corners of court in image coords [[x,y], ...]
                          Order: far-left, far-right, near-right, near-left
            fps: Video frame rate
            frame_size: (width, height) of video frames
        """
        self.fps = fps
        self.frame_width, self.frame_height = frame_size

        # Setup homography for court-to-pixel mapping
        if court_corners is not None:
            self.court_corners = court_corners.astype(np.float32)
        else:
            # Scale default corners to frame size
            scale_x = frame_size[0] / 1920.0
            scale_y = frame_size[1] / 1080.0
            self.court_corners = self.DEFAULT_COURT_CORNERS.copy()
            self.court_corners[:, 0] *= scale_x
            self.court_corners[:, 1] *= scale_y

        # Real-world court corners (in meters, origin at far-left)
        self.court_corners_real = np.array([
            [0, 0],                          # Far-Left
            [self.COURT_WIDTH_M, 0],         # Far-Right
            [self.COURT_WIDTH_M, self.COURT_LENGTH_M],  # Near-Right
            [0, self.COURT_LENGTH_M],        # Near-Left
        ], dtype=np.float32)

        # Compute homography (pixels -> meters on ground plane)
        self.homography, _ = cv2.findHomography(
            self.court_corners,
            self.court_corners_real
        )

        # Compute inverse homography (meters -> pixels)
        self.homography_inv = np.linalg.inv(self.homography)

        # Approximate scale factor (pixels per meter at mid-court)
        # This is a simplification - actual scale varies with perspective
        mid_court_pixel = np.mean(self.court_corners, axis=0)
        self.pixels_per_meter = self._estimate_local_scale(mid_court_pixel)

        # Shot detection parameters
        self.velocity_angle_threshold = 30.0  # degrees - min angle change for shot (lowered for sensitivity)
        self.min_shot_gap_frames = 30  # ~1 sec at 30fps between shots (prevents consecutive false positives)
        self.player_ball_distance_threshold = 150  # pixels - max distance for attribution (reduced from 250)

        # Shot classifier
        self.shot_classifier = ShotClassifier(frame_height=frame_size[1])

        # Pose model (lazy loaded)
        self._pose_model = None

        # Statistics
        self.shots: List[ShotEvent] = []
        self.match_stats = MatchStats()

    def _get_pose_model(self):
        """Lazy load YOLO pose model"""
        if self._pose_model is None:
            try:
                from ultralytics import YOLO
                self._pose_model = YOLO('yolo11m-pose.pt')
                print("Loaded YOLO pose model")
            except Exception as e:
                print(f"Warning: Could not load pose model: {e}")
                self._pose_model = False  # Mark as unavailable
        return self._pose_model if self._pose_model else None

    def _estimate_local_scale(self, pixel_pos: np.ndarray) -> float:
        """Estimate pixels per meter at a given pixel position"""
        # Use homography to estimate local scale
        # Move 1 meter in real world, see how many pixels it corresponds to
        pt = np.array([[pixel_pos[0], pixel_pos[1]]], dtype=np.float32)
        pt_real = cv2.perspectiveTransform(pt.reshape(1, -1, 2), self.homography)[0, 0]

        # Point 1 meter to the right in real coords
        pt_real_offset = pt_real + np.array([1.0, 0.0])
        pt_pixel_offset = cv2.perspectiveTransform(
            pt_real_offset.reshape(1, -1, 2).astype(np.float32),
            self.homography_inv
        )[0, 0]

        # Distance in pixels for 1 meter
        scale = np.linalg.norm(pt_pixel_offset - pixel_pos)
        return scale if scale > 0 else 50.0  # fallback

    def pixel_to_meters(self, pixel_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Convert pixel coordinates to court meters"""
        pt = np.array([[pixel_pos[0], pixel_pos[1]]], dtype=np.float32)
        pt_real = cv2.perspectiveTransform(pt.reshape(1, -1, 2), self.homography)[0, 0]
        return (float(pt_real[0]), float(pt_real[1]))

    def pixel_speed_to_kmh(self, speed_pixels_per_frame: float,
                          ball_y: float, ball_x: float = None) -> float:
        """
        Convert pixel speed to km/h.

        Uses perspective-aware scaling based on ball's Y position.
        Note: Ball flies through the air, so homography (ground plane) isn't accurate.
        These empirical values are calibrated for typical padel broadcast views.
        """
        # Perspective-aware pixels per meter (empirically tuned for broadcast)
        y_ratio = ball_y / self.frame_height  # 0 = top, 1 = bottom

        # At y_ratio=0.3 (far): ~35 px/m, at y_ratio=0.9 (near): ~90 px/m
        pixels_per_meter = 35 + (90 - 35) * y_ratio

        # Convert: pixels/frame -> meters/second -> km/h
        meters_per_frame = speed_pixels_per_frame / pixels_per_meter
        meters_per_second = meters_per_frame * self.fps
        kmh = meters_per_second * 3.6

        # Sanity check: padel ball speeds typically 20-150 km/h
        kmh = min(max(kmh, 0), 200)

        return kmh

    def compute_velocity(self,
                        positions: List[Tuple[Optional[float], Optional[float]]],
                        frame_idx: int,
                        window: int = 3) -> Tuple[float, float]:
        """
        Compute velocity at a frame using surrounding positions.

        Returns (vx, vy) in pixels per frame.
        """
        # Collect valid positions in window
        valid_before = []
        valid_after = []

        for i in range(max(0, frame_idx - window), frame_idx):
            if positions[i][0] is not None:
                valid_before.append((i, positions[i]))

        for i in range(frame_idx + 1, min(len(positions), frame_idx + window + 1)):
            if positions[i][0] is not None:
                valid_after.append((i, positions[i]))

        if not valid_before or not valid_after:
            return (0.0, 0.0)

        # Use closest valid points
        before_idx, before_pos = valid_before[-1]
        after_idx, after_pos = valid_after[0]

        dt = after_idx - before_idx
        if dt == 0:
            return (0.0, 0.0)

        vx = (after_pos[0] - before_pos[0]) / dt
        vy = (after_pos[1] - before_pos[1]) / dt

        return (vx, vy)

    def detect_shots_vshape(self,
                           ball_positions: List[Dict],
                           player_positions: List[Dict],
                           frames: Optional[List[np.ndarray]] = None,
                           use_pose: bool = True,
                           verbose: bool = True) -> List[ShotEvent]:
        """
        Detect shots using V-shape velocity analysis.

        A shot is detected when:
        1. Ball velocity direction changes significantly (> threshold angle)
        2. The change happens near a player

        Args:
            ball_positions: List of {1: [x1,y1,x2,y2]} dicts from ball tracker
            player_positions: List of {player_id: [x1,y1,x2,y2]} dicts
            frames: Optional video frames for pose extraction
            use_pose: Whether to use pose model for classification

        Returns:
            List of ShotEvent objects
        """
        # Extract ball center positions
        positions = []
        for bp in ball_positions:
            if 1 in bp and bp[1]:
                bbox = bp[1]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                positions.append((cx, cy))
            else:
                positions.append((None, None))

        # Compute frame-to-frame velocities
        velocities = []
        for i in range(len(positions)):
            if i == 0:
                velocities.append((0.0, 0.0))
            elif positions[i][0] is not None and positions[i-1][0] is not None:
                vx = positions[i][0] - positions[i-1][0]
                vy = positions[i][1] - positions[i-1][1]
                velocities.append((vx, vy))
            else:
                velocities.append((None, None))

        # Smooth velocities with rolling window
        smoothed_vel = []
        window = 3
        for i in range(len(velocities)):
            vx_sum, vy_sum, count = 0, 0, 0
            for j in range(max(0, i - window), min(len(velocities), i + window + 1)):
                if velocities[j][0] is not None:
                    vx_sum += velocities[j][0]
                    vy_sum += velocities[j][1]
                    count += 1
            if count > 0:
                smoothed_vel.append((vx_sum / count, vy_sum / count))
            else:
                smoothed_vel.append((0.0, 0.0))

        # Detect V-shapes (sudden direction changes)
        # First pass: collect ALL candidates with their distances to players
        candidates = []

        for i in range(window, len(smoothed_vel) - window):
            if positions[i][0] is None:
                continue

            # Get velocity before and after this frame
            vel_before = self._avg_velocity(smoothed_vel, i - window, i)
            vel_after = self._avg_velocity(smoothed_vel, i, i + window)

            # Compute angle change
            angle_change = self._compute_angle_change(vel_before, vel_after)

            # Check if this is a significant direction change
            if angle_change >= self.velocity_angle_threshold:
                # Filter out bounces: ball near ground + downward to upward transition
                ball_y_ratio = positions[i][1] / self.frame_height
                ball_x_ratio = positions[i][0] / self.frame_width

                is_ground_bounce = (ball_y_ratio > 0.70 and  # Ball in lower 30% of frame
                                   vel_before[1] > 2 and     # Was moving downward
                                   vel_after[1] < -2)        # Now moving upward

                # Filter wall bounces: ball near side edges + horizontal direction reversal
                is_wall_bounce = ((ball_x_ratio < 0.15 or ball_x_ratio > 0.85) and  # Near side walls
                                 abs(vel_before[0]) > 3 and  # Moving horizontally
                                 vel_before[0] * vel_after[0] < 0)  # Direction reversed

                # Filter gravity arc: ball trajectory just curving more downward (not a true hit)
                # A real hit typically: changes horizontal direction OR sends ball upward
                # Gravity arc: ball keeps going same horizontal direction, just goes more downward
                is_gravity_arc = (
                    vel_before[1] > 0 and vel_after[1] > 0 and  # Both moving downward
                    vel_after[1] > vel_before[1] and  # Going MORE downward (accelerating down)
                    vel_before[0] * vel_after[0] > 0 and  # Same horizontal direction
                    abs(vel_after[0] - vel_before[0]) < 5 and  # Horizontal speed similar
                    angle_change < 50  # Not a very sharp turn
                )

                # Filter weak direction changes: tiny velocity reversal is not a real hit
                # A real hit causes significant velocity change, not just ball slowing down
                speed_before = np.sqrt(vel_before[0]**2 + vel_before[1]**2)
                speed_after = np.sqrt(vel_after[0]**2 + vel_after[1]**2)
                is_weak_change = (
                    angle_change < 50 and  # Not a sharp turn
                    (abs(vel_after[0]) < 3 or abs(vel_before[0]) < 3) and  # One side has weak horizontal motion
                    speed_after < 15  # Low outgoing speed in pixels/frame
                )

                # Filter gravity descent: ball was descending and continues to descend faster
                # A real hit would typically reverse direction or at least reduce downward speed
                is_gravity_descent = (
                    vel_before[1] > 3 and vel_after[1] > 3 and  # Both moving significantly downward
                    vel_after[1] > vel_before[1] * 1.5 and  # Vertical speed increased by >50% (gravity acceleration)
                    angle_change < 60  # Not a very sharp turn
                )

                if is_ground_bounce or is_wall_bounce or is_gravity_arc or is_weak_change or is_gravity_descent:
                    continue

                # Find which player hit the ball and get distance
                player_id, ball_player_dist = self._attribute_shot_to_player(
                    i, positions[i], player_positions, return_distance=True
                )

                # Skip if no player is close enough
                if player_id == 0:
                    continue

                candidates.append({
                    'frame': i,
                    'player_id': player_id,
                    'distance': ball_player_dist,
                    'angle_change': angle_change,
                    'vel_before': vel_before,
                    'vel_after': vel_after,
                    'position': positions[i]
                })

        # Second pass: for each player, pick best frame (closest distance) in consecutive sequences
        shots = []
        last_shot_by_player = {1: -999, 2: -999, 3: -999, 4: -999}
        last_global_shot_frame = -999  # Track ANY shot for post-shot bounce filtering
        last_global_shot_speed = 0  # Speed of last shot

        i = 0
        while i < len(candidates):
            c = candidates[i]
            player_id = c['player_id']

            # Check gap from last committed shot by this player
            if c['frame'] - last_shot_by_player[player_id] < self.min_shot_gap_frames:
                i += 1
                continue

            # Post-shot bounce filter: if there was a shot VERY recently (from ANY player),
            # and this candidate has very low speed, it's likely a bounce, not a new shot
            # Use shorter window (15 frames = 0.5 sec) to allow quick rally exchanges
            frames_since_shot = c['frame'] - last_global_shot_frame
            if frames_since_shot > 0 and frames_since_shot < 15:  # Within ~0.5 sec after a shot
                # Estimate speed of this candidate
                cand_speed = np.linalg.norm(c['vel_after'])
                cand_speed_kmh = self.pixel_speed_to_kmh(cand_speed, c['position'][1])
                # If very low speed (<15 km/h) immediately after a shot, likely bounce
                if cand_speed_kmh < 15.0:
                    if verbose:
                        print(f"  Filtered as bounce: frame {c['frame']} (speed={cand_speed_kmh:.1f} km/h, {frames_since_shot} frames after shot)")
                    i += 1
                    continue

            # Find all truly consecutive candidates (within 5 frames of each other)
            # Pick the FIRST frame (start of direction change = actual impact moment)
            # NOT the peak angle change (which comes later, after ball already changed direction)
            best_candidate = c  # First frame is the impact
            last_frame = c['frame']
            j = i + 1
            while j < len(candidates):
                next_c = candidates[j]
                # Check if this is truly consecutive (within 5 frames, same player)
                if (next_c['player_id'] == player_id and
                    next_c['frame'] - last_frame <= 5):
                    # Keep the FIRST candidate (actual impact), skip the rest
                    last_frame = next_c['frame']
                    j += 1
                else:
                    break

            # Commit the best candidate
            bc = best_candidate
            frame_idx = bc['frame']
            vel_before = bc['vel_before']
            vel_after = bc['vel_after']
            angle_change = bc['angle_change']

            # Compute ball speed after hit
            peak_speed = 0.0
            for k in range(frame_idx, min(len(velocities), frame_idx + 5)):
                if velocities[k][0] is not None:
                    v = np.sqrt(velocities[k][0]**2 + velocities[k][1]**2)
                    peak_speed = max(peak_speed, v)

            speed_after = max(np.linalg.norm(vel_after), peak_speed)
            speed_kmh = self.pixel_speed_to_kmh(speed_after, bc['position'][1], bc['position'][0])

            # Extract pose features if frames available
            pose_features = None
            if use_pose and frames is not None and player_id > 0:
                pose_features = self._extract_player_pose(
                    frames, frame_idx, player_id, player_positions, bc['position']
                )

            # Hybrid shot classification
            shot_type = self.shot_classifier.classify_shot(
                ball_speed_kmh=speed_kmh,
                ball_angle_change=angle_change,
                ball_position=bc['position'],
                velocity_before=vel_before,
                velocity_after=vel_after,
                pose_features=pose_features
            )

            shot = ShotEvent(
                frame_idx=frame_idx,
                player_id=player_id,
                ball_position=bc['position'],
                velocity_before=vel_before,
                velocity_after=vel_after,
                speed_pixels=speed_after,
                speed_kmh=speed_kmh,
                shot_type=shot_type,
                angle_change=angle_change
            )

            shots.append(shot)
            last_shot_by_player[player_id] = frame_idx
            last_global_shot_frame = frame_idx
            last_global_shot_speed = speed_kmh

            if verbose:
                print(f"  Shot: P{player_id} at frame {frame_idx} (dist={bc['distance']:.0f}px, speed={speed_kmh:.1f}km/h)")

            i = j  # Skip to after the consecutive sequence

        if verbose:
            print(f"Detected {len(shots)} shots using V-shape analysis")

        self.shots = shots
        return shots

    def _avg_velocity(self, velocities: List[Tuple[float, float]],
                     start: int, end: int) -> Tuple[float, float]:
        """Compute average velocity over a range"""
        vx_sum, vy_sum, count = 0, 0, 0
        for i in range(start, end):
            if 0 <= i < len(velocities):
                vx_sum += velocities[i][0]
                vy_sum += velocities[i][1]
                count += 1
        if count == 0:
            return (0.0, 0.0)
        return (vx_sum / count, vy_sum / count)

    def _compute_angle_change(self,
                             vel1: Tuple[float, float],
                             vel2: Tuple[float, float]) -> float:
        """Compute angle change between two velocity vectors in degrees"""
        mag1 = np.sqrt(vel1[0]**2 + vel1[1]**2)
        mag2 = np.sqrt(vel2[0]**2 + vel2[1]**2)

        if mag1 < 1 or mag2 < 1:  # Too slow to determine direction
            return 0.0

        # Normalize
        v1 = (vel1[0] / mag1, vel1[1] / mag1)
        v2 = (vel2[0] / mag2, vel2[1] / mag2)

        # Dot product = cos(angle)
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        dot = np.clip(dot, -1.0, 1.0)

        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def _attribute_shot_to_player(self,
                                  frame_idx: int,
                                  ball_pos: Tuple[float, float],
                                  player_positions: List[Dict],
                                  return_distance: bool = False):
        """
        Determine which player hit the ball.

        Uses proximity: the closest player to the ball is the hitter.

        Args:
            return_distance: If True, return (player_id, distance) tuple
        """
        if frame_idx >= len(player_positions):
            return (0, float('inf')) if return_distance else 0

        players = player_positions[frame_idx]
        if not players:
            return (0, float('inf')) if return_distance else 0

        best_player = 0
        best_distance = float('inf')

        for player_id, bbox in players.items():
            # Player center (using upper body - more relevant for hitting)
            px = (bbox[0] + bbox[2]) / 2
            # Use top third of player bbox (hitting position)
            py = bbox[1] + (bbox[3] - bbox[1]) * 0.3

            dist = np.sqrt((ball_pos[0] - px)**2 + (ball_pos[1] - py)**2)

            if dist < best_distance and dist < self.player_ball_distance_threshold:
                best_distance = dist
                best_player = player_id

        if return_distance:
            return (best_player, best_distance)
        return best_player

    def _extract_player_pose(self,
                            frames: List[np.ndarray],
                            frame_idx: int,
                            player_id: int,
                            player_positions: List[Dict],
                            ball_position: Tuple[float, float]) -> Optional[Dict]:
        """
        Extract pose features for the hitting player.

        Args:
            frames: Video frames
            frame_idx: Frame index of the shot
            player_id: ID of the hitting player
            player_positions: All player positions
            ball_position: Ball position at impact

        Returns:
            Dict of pose features or None
        """
        pose_model = self._get_pose_model()
        if pose_model is None:
            return None

        if frame_idx >= len(frames) or frame_idx >= len(player_positions):
            return None

        players = player_positions[frame_idx]
        if player_id not in players:
            return None

        frame = frames[frame_idx]
        bbox = players[player_id]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Add padding around player
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)

        # Crop player region
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        try:
            # Run pose model on cropped region
            results = pose_model(cropped, verbose=False)[0]

            if results.keypoints is None or len(results.keypoints.data) == 0:
                return None

            # Get keypoints and adjust to full frame coordinates
            kpts = results.keypoints.data[0].cpu().numpy()

            # Adjust coordinates from crop to full frame
            adjusted_kpts = []
            for kpt in kpts:
                adjusted_kpts.append([kpt[0] + x1, kpt[1] + y1, kpt[2]])
            keypoints = np.array(adjusted_kpts)

            # Extract pose features using classifier
            pose_features = self.shot_classifier.extract_pose_features(
                keypoints, ball_position, bbox
            )

            return pose_features if pose_features else None

        except Exception as e:
            # Silently fail - pose extraction is optional
            return None

    def compute_statistics(self,
                          player_positions: List[Dict],
                          verbose: bool = True) -> MatchStats:
        """
        Compute match statistics from detected shots.
        """
        stats = MatchStats()

        # Count shots by player
        for shot in self.shots:
            stats.total_shots += 1
            if shot.player_id in stats.shots_by_player:
                stats.shots_by_player[shot.player_id] += 1
                if shot.speed_kmh:
                    stats.shot_speeds[shot.player_id].append(shot.speed_kmh)

        # Compute average speeds
        for player_id in [1, 2, 3, 4]:
            speeds = stats.shot_speeds[player_id]
            if speeds:
                stats.avg_speed_by_player[player_id] = np.mean(speeds)

        # Detect rallies (sequences of shots)
        stats.rallies = self._detect_rallies()
        if stats.rallies:
            stats.avg_rally_length = np.mean([r.shot_count for r in stats.rallies])

        # Collect player positions for heatmap
        for frame_players in player_positions:
            for player_id, bbox in frame_players.items():
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                if player_id in stats.player_positions:
                    stats.player_positions[player_id].append((cx, cy))

        self.match_stats = stats

        if verbose:
            print("\n=== Match Statistics ===")
            print(f"Total shots detected: {stats.total_shots}")
            print(f"Shots by player:")
            for pid in [1, 2, 3, 4]:
                count = stats.shots_by_player[pid]
                avg_speed = stats.avg_speed_by_player[pid]
                print(f"  Player {pid}: {count} shots, avg speed: {avg_speed:.1f} km/h")
            print(f"Number of rallies: {len(stats.rallies)}")
            print(f"Average rally length: {stats.avg_rally_length:.1f} shots")

        return stats

    def _detect_rallies(self, rally_gap_frames: int = 90) -> List[RallyStats]:
        """
        Group shots into rallies.

        A rally ends when there's a gap > rally_gap_frames (~3 seconds) between shots.
        """
        if not self.shots:
            return []

        rallies = []
        current_rally_shots = [self.shots[0]]

        for i in range(1, len(self.shots)):
            gap = self.shots[i].frame_idx - self.shots[i-1].frame_idx

            if gap > rally_gap_frames:
                # End current rally, start new one
                rally = RallyStats(
                    start_frame=current_rally_shots[0].frame_idx,
                    end_frame=current_rally_shots[-1].frame_idx,
                    shot_count=len(current_rally_shots),
                    shots=current_rally_shots.copy()
                )
                rallies.append(rally)
                current_rally_shots = [self.shots[i]]
            else:
                current_rally_shots.append(self.shots[i])

        # Don't forget the last rally
        if current_rally_shots:
            rally = RallyStats(
                start_frame=current_rally_shots[0].frame_idx,
                end_frame=current_rally_shots[-1].frame_idx,
                shot_count=len(current_rally_shots),
                shots=current_rally_shots.copy()
            )
            rallies.append(rally)

        return rallies

    def generate_heatmaps(self,
                         frame_size: Optional[Tuple[int, int]] = None) -> Dict[int, np.ndarray]:
        """
        Generate position heatmaps for each player.

        Returns dict of {player_id: heatmap_image}
        """
        if frame_size is None:
            frame_size = (self.frame_width, self.frame_height)

        heatmaps = {}

        for player_id in [1, 2, 3, 4]:
            positions = self.match_stats.player_positions.get(player_id, [])

            if not positions:
                heatmaps[player_id] = np.zeros((frame_size[1], frame_size[0]), dtype=np.float32)
                continue

            # Create accumulator
            heatmap = np.zeros((frame_size[1], frame_size[0]), dtype=np.float32)

            for x, y in positions:
                ix, iy = int(x), int(y)
                if 0 <= ix < frame_size[0] and 0 <= iy < frame_size[1]:
                    # Add gaussian blob
                    cv2.circle(heatmap, (ix, iy), 20, 1.0, -1)

            # Normalize
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            # Apply colormap
            heatmap_color = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            heatmaps[player_id] = heatmap_color

        return heatmaps

    def draw_shot_indicators(self,
                            frames: List[np.ndarray],
                            show_speed: bool = True,
                            pause_on_shots: bool = False,
                            pause_frames: int = 45) -> List[np.ndarray]:
        """
        Draw shot indicators on video frames.

        Args:
            frames: List of video frames
            show_speed: Show speed label
            pause_on_shots: If True, freeze video at each shot for pause_frames
            pause_frames: Number of frames to pause (e.g., 45 = 1.5 sec at 30fps)
        """
        output_frames = []
        shot_frames = {s.frame_idx: s for s in self.shots}

        # Team colors
        colors = {
            1: (0, 255, 0),    # Green - Team 1
            2: (0, 200, 100),  # Teal - Team 1
            3: (0, 0, 255),    # Red - Team 2
            4: (0, 100, 255),  # Orange - Team 2
            0: (255, 255, 255) # White - Unknown
        }

        for i, frame in enumerate(frames):
            frame_out = frame.copy()

            # Check if this frame has a shot
            if i in shot_frames:
                shot = shot_frames[i]
                bx, by = int(shot.ball_position[0]), int(shot.ball_position[1])
                color = colors.get(shot.player_id, (255, 255, 255))

                # Draw shot indicator ring
                cv2.circle(frame_out, (bx, by), 30, color, 3, cv2.LINE_AA)
                cv2.circle(frame_out, (bx, by), 35, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw speed label
                if show_speed and shot.speed_kmh:
                    label = f"{shot.speed_kmh:.0f} km/h"
                    cv2.putText(frame_out, label, (bx - 40, by - 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame_out, label, (bx - 40, by - 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)

                # Draw player indicator
                player_label = f"P{shot.player_id}" if shot.player_id > 0 else "?"
                cv2.putText(frame_out, player_label, (bx - 10, by + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # If pause mode, create extended frame with shot info panel
                if pause_on_shots:
                    pause_frame = self._create_shot_info_frame(frame_out, shot, color)
                    # Add pause frames
                    for _ in range(pause_frames):
                        output_frames.append(pause_frame.copy())
                    continue  # Don't add the original frame again

            output_frames.append(frame_out)

        return output_frames

    def _create_shot_info_frame(self, frame: np.ndarray, shot: ShotEvent,
                                color: Tuple[int, int, int]) -> np.ndarray:
        """Create a frame with shot information panel overlay."""
        frame_out = frame.copy()
        h, w = frame_out.shape[:2]

        # Create semi-transparent info panel at bottom
        panel_h = 120
        overlay = frame_out.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame_out, 0.3, 0, frame_out)

        # Draw panel border
        cv2.line(frame_out, (0, h - panel_h), (w, h - panel_h), color, 3)

        # Shot info text
        y_base = h - panel_h + 35

        # Player info
        team = "Team 1" if shot.player_id in [1, 2] else "Team 2"
        player_text = f"PLAYER {shot.player_id} ({team})"
        cv2.putText(frame_out, player_text, (30, y_base),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        # Shot type
        shot_type_display = shot.shot_type.upper()
        cv2.putText(frame_out, f"Shot: {shot_type_display}", (30, y_base + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Speed
        if shot.speed_kmh:
            speed_text = f"Speed: {shot.speed_kmh:.0f} km/h"
            cv2.putText(frame_out, speed_text, (350, y_base + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Angle change
        angle_text = f"Direction change: {shot.angle_change:.0f}°"
        cv2.putText(frame_out, angle_text, (600, y_base + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        # Frame number
        frame_text = f"Frame: {shot.frame_idx}"
        cv2.putText(frame_out, frame_text, (w - 200, y_base),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

        # "SHOT DETECTED" banner
        banner_text = "SHOT DETECTED"
        text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame_out, banner_text, (text_x, y_base + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        return frame_out

    def export_stats_json(self, filepath: str):
        """Export statistics to JSON file"""
        data = {
            "total_shots": self.match_stats.total_shots,
            "shots_by_player": self.match_stats.shots_by_player,
            "avg_speed_by_player": {k: round(v, 1) for k, v in self.match_stats.avg_speed_by_player.items()},
            "num_rallies": len(self.match_stats.rallies),
            "avg_rally_length": round(self.match_stats.avg_rally_length, 1),
            "shots": [
                {
                    "frame": s.frame_idx,
                    "player": s.player_id,
                    "speed_kmh": round(s.speed_kmh, 1) if s.speed_kmh else None,
                    "type": s.shot_type,
                    "angle_change": round(s.angle_change, 1)
                }
                for s in self.shots
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Stats exported to {filepath}")
