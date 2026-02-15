"""
Hybrid Shot Classifier for Padel

Combines:
1. Ball trajectory features (speed, angle, height)
2. Player pose features (arm angle, arm height, body orientation)

Shot Types:
- Smash: High arm, fast downward ball, overhead contact
- Bandeja: Medium height, slice motion, controlled pace
- Vibora: Side arm motion, spin trajectory
- Lob: Low-to-high trajectory, defensive
- Volley: Near net, quick exchange
- Forehand: Standard ground stroke, arm on dominant side
- Backhand: Standard ground stroke, arm across body
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# COCO keypoint indices
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


@dataclass
class ShotFeatures:
    """Features extracted for shot classification"""
    # Trajectory features
    ball_speed_kmh: float
    ball_angle_change: float  # degrees
    ball_y_position: float    # normalized 0-1 (0=top, 1=bottom)
    ball_direction_y: float   # positive = downward, negative = upward

    # Pose features (if available)
    arm_angle: Optional[float] = None       # elbow angle in degrees
    arm_height: Optional[float] = None      # wrist height relative to shoulder (normalized)
    is_overhead: Optional[bool] = None      # wrist above shoulder
    dominant_side: Optional[str] = None     # 'left' or 'right' arm used
    body_rotation: Optional[float] = None   # torso rotation angle


class ShotClassifier:
    """
    Hybrid shot classifier using trajectory + pose features.
    """

    # Shot type thresholds (tuned for padel)
    SMASH_MIN_SPEED = 60.0          # km/h
    SMASH_MIN_ARM_HEIGHT = 0.3      # wrist 30% above shoulder
    LOB_MAX_SPEED = 50.0            # km/h
    LOB_MIN_UPWARD_ANGLE = 30.0     # degrees upward
    VOLLEY_MAX_CONTACT_Y = 0.5      # ball in upper half of frame (near net)
    BANDEJA_ARM_HEIGHT_RANGE = (0.0, 0.3)  # wrist near to slightly above shoulder

    def __init__(self, frame_height: int = 1080):
        self.frame_height = frame_height

    def extract_pose_features(self,
                             keypoints: np.ndarray,
                             ball_position: Tuple[float, float],
                             player_bbox: List[float]) -> Dict:
        """
        Extract pose features from YOLO keypoints.

        Args:
            keypoints: (17, 3) array of [x, y, confidence] for each keypoint
            ball_position: (x, y) of ball at impact
            player_bbox: [x1, y1, x2, y2] of player

        Returns:
            Dict of pose features
        """
        features = {}

        if keypoints is None or len(keypoints) < 17:
            return features

        # Get shoulder and wrist positions
        left_shoulder = keypoints[LEFT_SHOULDER]
        right_shoulder = keypoints[RIGHT_SHOULDER]
        left_wrist = keypoints[LEFT_WRIST]
        right_wrist = keypoints[RIGHT_WRIST]
        left_elbow = keypoints[LEFT_ELBOW]
        right_elbow = keypoints[RIGHT_ELBOW]

        # Determine which arm is dominant (closer to ball or higher)
        ball_x, ball_y = ball_position

        left_wrist_dist = np.sqrt((left_wrist[0] - ball_x)**2 + (left_wrist[1] - ball_y)**2)
        right_wrist_dist = np.sqrt((right_wrist[0] - ball_x)**2 + (right_wrist[1] - ball_y)**2)

        if left_wrist[2] > 0.5 and right_wrist[2] > 0.5:
            # Both wrists visible - use the one closer to ball
            if left_wrist_dist < right_wrist_dist:
                dominant = 'left'
                wrist = left_wrist
                elbow = left_elbow
                shoulder = left_shoulder
            else:
                dominant = 'right'
                wrist = right_wrist
                elbow = right_elbow
                shoulder = right_shoulder
        elif left_wrist[2] > 0.5:
            dominant = 'left'
            wrist = left_wrist
            elbow = left_elbow
            shoulder = left_shoulder
        elif right_wrist[2] > 0.5:
            dominant = 'right'
            wrist = right_wrist
            elbow = right_elbow
            shoulder = right_shoulder
        else:
            return features

        features['dominant_side'] = dominant

        # Arm height: wrist Y relative to shoulder Y (normalized by bbox height)
        # Negative = wrist above shoulder (overhead)
        bbox_height = player_bbox[3] - player_bbox[1]
        if bbox_height > 0 and shoulder[2] > 0.5:
            arm_height = (wrist[1] - shoulder[1]) / bbox_height
            features['arm_height'] = arm_height
            features['is_overhead'] = arm_height < -0.1  # wrist above shoulder

        # Arm angle (elbow angle)
        if elbow[2] > 0.5 and shoulder[2] > 0.5 and wrist[2] > 0.5:
            angle = self._compute_angle(
                (shoulder[0], shoulder[1]),
                (elbow[0], elbow[1]),
                (wrist[0], wrist[1])
            )
            features['arm_angle'] = angle

        # Body rotation (shoulder line angle)
        if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]
            rotation = np.degrees(np.arctan2(dy, dx))
            features['body_rotation'] = rotation

        return features

    def _compute_angle(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """Compute angle at p2 formed by p1-p2-p3"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def classify_shot(self,
                     ball_speed_kmh: float,
                     ball_angle_change: float,
                     ball_position: Tuple[float, float],
                     velocity_before: Tuple[float, float],
                     velocity_after: Tuple[float, float],
                     pose_features: Optional[Dict] = None) -> str:
        """
        Classify shot type using trajectory + pose features.

        Args:
            ball_speed_kmh: Ball speed after hit in km/h
            ball_angle_change: Direction change in degrees
            ball_position: (x, y) position at impact
            velocity_before: (vx, vy) before hit
            velocity_after: (vx, vy) after hit
            pose_features: Dict from extract_pose_features (optional)

        Returns:
            Shot type string
        """
        # Normalize ball Y position
        ball_y_norm = ball_position[1] / self.frame_height

        # Ball direction after hit (positive vy = moving down)
        vy_after = velocity_after[1] if velocity_after else 0
        vy_before = velocity_before[1] if velocity_before else 0

        # Check for upward trajectory (lob)
        is_upward = vy_after < -5  # ball moving up significantly

        # Extract pose info if available
        is_overhead = False
        arm_height = 0
        arm_angle = 150  # default extended

        if pose_features:
            is_overhead = pose_features.get('is_overhead', False)
            arm_height = pose_features.get('arm_height', 0)
            arm_angle = pose_features.get('arm_angle', 150)

        # Classification logic (ordered by specificity)

        # 1. SMASH: Fast, overhead, downward
        # For extremely high-speed shots (>80 km/h), classify as Smash regardless of direction
        # (this could be a smash return or power hit)
        if ball_speed_kmh >= 80.0:
            return "Smash"
        # For high-speed shots (70-80 km/h), require ball not moving strongly upward
        elif ball_speed_kmh >= 70.0:
            if vy_after >= 0:  # Not moving upward
                return "Smash"
        elif ball_speed_kmh >= self.SMASH_MIN_SPEED:
            # Moderate fast shot (60-70) - need some overhead indicator
            if ((is_overhead or arm_height < 0 or ball_y_norm < 0.5) and
                vy_after > 0):
                return "Smash"

        # 2. LOB: Slow, upward trajectory
        if (is_upward and
            ball_speed_kmh <= self.LOB_MAX_SPEED and
            ball_angle_change > 30):
            return "Lob"

        # 3. VOLLEY: Near net (upper half of frame), quick exchange
        if (ball_y_norm < self.VOLLEY_MAX_CONTACT_Y and
            ball_speed_kmh < 80 and
            not is_overhead):
            return "Volley"

        # 4. BANDEJA: Medium height overhead, controlled pace
        if (is_overhead or (arm_height is not None and -0.3 < arm_height < 0.1)):
            if 30 <= ball_speed_kmh < 70:
                return "Bandeja"

        # 5. VIBORA: Similar to bandeja but with more spin/side motion
        # Hard to detect without trajectory curve analysis
        # For now, classify fast bandejas as vibora
        if (is_overhead or (arm_height is not None and arm_height < 0)):
            if ball_speed_kmh >= 70:
                return "Vibora"

        # 6. Default to Forehand/Backhand based on arm side
        if pose_features and 'dominant_side' in pose_features:
            side = pose_features['dominant_side']
            # This is simplified - would need player facing direction
            # For now, just return the side
            if side == 'right':
                return "Forehand"
            else:
                return "Backhand"

        # 7. Fallback classification based on trajectory only
        if ball_y_norm > 0.6:  # Lower half = ground stroke
            if ball_speed_kmh > 50:
                return "Drive"
            else:
                return "Forehand"
        else:
            return "Volley"

    def classify_with_confidence(self,
                                ball_speed_kmh: float,
                                ball_angle_change: float,
                                ball_position: Tuple[float, float],
                                velocity_before: Tuple[float, float],
                                velocity_after: Tuple[float, float],
                                pose_features: Optional[Dict] = None) -> Tuple[str, float]:
        """
        Classify shot and return confidence score.

        Returns:
            (shot_type, confidence) where confidence is 0-1
        """
        shot_type = self.classify_shot(
            ball_speed_kmh, ball_angle_change, ball_position,
            velocity_before, velocity_after, pose_features
        )

        # Compute confidence based on how well features match
        confidence = 0.5  # Base confidence

        if shot_type == "Smash":
            if ball_speed_kmh >= 80:
                confidence += 0.2
            if pose_features and pose_features.get('is_overhead'):
                confidence += 0.2
            if pose_features and pose_features.get('arm_angle', 180) > 150:
                confidence += 0.1

        elif shot_type == "Lob":
            vy_after = velocity_after[1] if velocity_after else 0
            if vy_after < -10:
                confidence += 0.2
            if ball_speed_kmh < 40:
                confidence += 0.2

        elif shot_type == "Volley":
            ball_y_norm = ball_position[1] / self.frame_height
            if ball_y_norm < 0.4:
                confidence += 0.2
            if ball_speed_kmh < 60:
                confidence += 0.1

        elif shot_type in ["Bandeja", "Vibora"]:
            if pose_features and pose_features.get('is_overhead'):
                confidence += 0.2
            if 40 <= ball_speed_kmh <= 80:
                confidence += 0.1

        # Pose features boost confidence
        if pose_features and len(pose_features) >= 3:
            confidence += 0.1

        return shot_type, min(confidence, 1.0)
