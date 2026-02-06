"""
Professional Ball Tracking Visualization System

Comprehensive visualization with broadcast-quality effects:
- Glowing ball with speed-based coloring
- Motion blur and particle effects
- Trajectory prediction and bezier curves
- Shot impact animations
- HUD elements (speed, rally counter, graphs)
- Court zone overlays
"""

import cv2
import numpy as np
from collections import deque
import math


class ProfessionalBallVisualizer:
    """
    Professional-grade ball tracking visualization with multiple effects.
    """

    def __init__(self, frame_width, frame_height, fps=30):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

        # Trajectory history
        self.trajectory = deque(maxlen=30)
        self.velocity_history = deque(maxlen=30)
        self.speed_history = deque(maxlen=100)  # For graph

        # Stats tracking
        self.max_speed = 0
        self.rally_count = 0
        self.total_distance = 0
        self.last_position = None

        # Shot detection
        self.shot_frames = []  # Frames where shots occurred
        self.current_shot_frame = -100  # For ripple animation

        # Colors (BGR)
        self.COLOR_SLOW = (255, 200, 0)      # Cyan for slow
        self.COLOR_MEDIUM = (0, 255, 255)    # Yellow for medium
        self.COLOR_FAST = (0, 100, 255)      # Orange-red for fast
        self.COLOR_GLOW = (255, 255, 0)      # Cyan glow
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_ORANGE = (0, 165, 255)

    def _get_speed_color(self, speed_kmh):
        """Get color based on ball speed (blue->yellow->red gradient)"""
        # Normalize speed (0-150 km/h typical range for padel)
        t = min(1.0, speed_kmh / 120.0)

        if t < 0.5:
            # Blue to Yellow
            t2 = t * 2
            b = int(255 * (1 - t2))
            g = int(255 * t2)
            r = int(255 * t2)
        else:
            # Yellow to Red
            t2 = (t - 0.5) * 2
            b = 0
            g = int(255 * (1 - t2))
            r = 255

        return (b, g, r)

    def _calculate_velocity(self, current_pos):
        """Calculate velocity and speed from position history"""
        if len(self.trajectory) < 2:
            return (0, 0), 0

        prev_pos = self.trajectory[-1]
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]

        # Pixels per frame to km/h (approximate conversion)
        # Assuming court width ~10m maps to frame width
        pixels_per_meter = self.frame_width / 10.0
        distance_meters = math.sqrt(dx**2 + dy**2) / pixels_per_meter
        speed_ms = distance_meters * self.fps
        speed_kmh = speed_ms * 3.6

        return (dx, dy), speed_kmh

    def _draw_motion_blur(self, frame, pos, velocity, speed_kmh):
        """Draw motion blur effect based on velocity"""
        if speed_kmh < 20:
            return

        dx, dy = velocity
        length = min(50, int(speed_kmh / 3))

        # Normalize direction
        mag = math.sqrt(dx**2 + dy**2)
        if mag > 0:
            dx, dy = dx/mag, dy/mag

        # Draw elongated blur trail
        for i in range(length, 0, -2):
            alpha = (length - i) / length * 0.3
            blur_x = int(pos[0] - dx * i)
            blur_y = int(pos[1] - dy * i)
            radius = max(2, 7 - i // 5)

            overlay = frame.copy()
            color = self._get_speed_color(speed_kmh)
            cv2.circle(overlay, (blur_x, blur_y), radius, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_particles(self, frame, pos, velocity, speed_kmh):
        """Draw particle effects behind fast-moving ball"""
        if speed_kmh < 40:
            return

        dx, dy = velocity
        mag = math.sqrt(dx**2 + dy**2)
        if mag == 0:
            return
        dx, dy = dx/mag, dy/mag

        num_particles = min(15, int(speed_kmh / 10))
        for _ in range(num_particles):
            # Random offset behind ball
            offset = np.random.randint(10, 40)
            spread = np.random.randint(-15, 15)

            px = int(pos[0] - dx * offset + dy * spread)
            py = int(pos[1] - dy * offset - dx * spread)

            # Random size and alpha
            size = np.random.randint(1, 4)
            alpha = np.random.uniform(0.2, 0.5)

            overlay = frame.copy()
            cv2.circle(overlay, (px, py), size, self.COLOR_WHITE, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_bezier_trail(self, frame, speed_kmh):
        """Draw smooth bezier curve trail"""
        if len(self.trajectory) < 4:
            return

        points = list(self.trajectory)

        # Draw smooth trail with gradient
        for i in range(1, len(points)):
            alpha = i / len(points)
            color = self._get_speed_color(speed_kmh * alpha)
            thickness = max(1, int(4 * alpha))

            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))

            # Anti-aliased line
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    def _draw_ghost_balls(self, frame):
        """Draw fading ghost balls along trajectory"""
        if len(self.trajectory) < 3:
            return

        points = list(self.trajectory)
        for i, pos in enumerate(points[:-1]):
            if i % 3 != 0:  # Every 3rd position
                continue

            alpha = (i / len(points)) * 0.3
            overlay = frame.copy()
            cv2.circle(overlay, (int(pos[0]), int(pos[1])), 5, self.COLOR_GLOW, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_prediction_arc(self, frame, pos, velocity, speed_kmh):
        """Draw predicted trajectory arc"""
        if speed_kmh < 30 or len(self.trajectory) < 3:
            return

        dx, dy = velocity

        # Simple physics prediction (with gravity)
        gravity = 0.5  # pixels per frame^2
        points = []
        px, py = float(pos[0]), float(pos[1])
        vx, vy = dx * 0.8, dy * 0.8  # Dampen velocity

        for _ in range(20):
            px += vx
            py += vy
            vy += gravity  # Add gravity

            if 0 < px < self.frame_width and 0 < py < self.frame_height:
                points.append((int(px), int(py)))
            else:
                break

        # Draw dotted prediction line
        for i, pt in enumerate(points):
            if i % 2 == 0:
                alpha = 0.3 * (1 - i / len(points)) if points else 0.3
                overlay = frame.copy()
                cv2.circle(overlay, pt, 3, self.COLOR_ORANGE, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_impact_ripple(self, frame, pos, frames_since_shot):
        """Draw expanding ripple effect on shot"""
        if frames_since_shot > 20:
            return

        # Multiple expanding rings
        for ring in range(3):
            radius = 20 + frames_since_shot * 8 + ring * 15
            alpha = max(0, 0.5 - frames_since_shot * 0.025 - ring * 0.1)

            if alpha > 0:
                overlay = frame.copy()
                cv2.circle(overlay, pos, radius, self.COLOR_ORANGE, 2, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_speed_indicator(self, frame, pos, speed_kmh):
        """Draw speed text near ball"""
        text = f"{int(speed_kmh)} km/h"
        text_pos = (pos[0] + 20, pos[1] - 20)

        # Background
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (text_pos[0] - 2, text_pos[1] - h - 2),
                     (text_pos[0] + w + 2, text_pos[1] + 2), (0, 0, 0), -1)

        color = self._get_speed_color(speed_kmh)
        cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, 1, cv2.LINE_AA)

    def _draw_power_meter(self, frame, speed_kmh):
        """Draw shot power meter bar"""
        # Position in top-right
        x, y = self.frame_width - 150, 20
        bar_width = 120
        bar_height = 15

        # Background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (100, 100, 100), 1)

        # Fill based on speed (max 150 km/h)
        fill_width = int(min(1.0, speed_kmh / 150) * bar_width)
        color = self._get_speed_color(speed_kmh)
        cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), color, -1)

        # Label
        cv2.putText(frame, "POWER", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, self.COLOR_WHITE, 1, cv2.LINE_AA)

    def _draw_speed_graph(self, frame):
        """Draw mini speed history graph"""
        if len(self.speed_history) < 2:
            return

        # Position in bottom-right
        graph_x, graph_y = self.frame_width - 220, self.frame_height - 80
        graph_w, graph_h = 200, 60

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x - 5, graph_y - 5),
                     (graph_x + graph_w + 5, graph_y + graph_h + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Border
        cv2.rectangle(frame, (graph_x, graph_y),
                     (graph_x + graph_w, graph_y + graph_h), (100, 100, 100), 1)

        # Draw speed line
        speeds = list(self.speed_history)
        max_speed_graph = max(speeds) if speeds else 100
        max_speed_graph = max(100, max_speed_graph)

        points = []
        for i, speed in enumerate(speeds):
            px = graph_x + int(i * graph_w / len(speeds))
            py = graph_y + graph_h - int(speed / max_speed_graph * graph_h)
            points.append((px, py))

        if len(points) >= 2:
            for i in range(1, len(points)):
                color = self._get_speed_color(speeds[i])
                cv2.line(frame, points[i-1], points[i], color, 2, cv2.LINE_AA)

        # Label
        cv2.putText(frame, "BALL SPEED", (graph_x, graph_y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1, cv2.LINE_AA)

    def _draw_stats_hud(self, frame):
        """Draw stats HUD (rally counter, max speed, distance)"""
        # Position in top-left
        x, y = 20, 30

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 25), (x + 180, y + 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Stats text
        stats = [
            f"RALLY: {self.rally_count}",
            f"MAX SPEED: {int(self.max_speed)} km/h",
            f"DISTANCE: {self.total_distance:.1f} m"
        ]

        for i, text in enumerate(stats):
            cv2.putText(frame, text, (x, y + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_WHITE, 1, cv2.LINE_AA)

    def _draw_glowing_ball(self, frame, pos, speed_kmh):
        """Draw the main glowing ball"""
        color = self._get_speed_color(speed_kmh)

        # Outer glow layers
        for glow_radius, glow_alpha in [(20, 0.1), (15, 0.2), (11, 0.35)]:
            overlay = frame.copy()
            cv2.circle(overlay, pos, glow_radius, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, glow_alpha, frame, 1 - glow_alpha, 0, frame)

        # Core ball
        cv2.circle(frame, pos, 8, color, -1, cv2.LINE_AA)

        # Inner highlight
        highlight_pos = (pos[0] - 2, pos[1] - 2)
        cv2.circle(frame, highlight_pos, 3, self.COLOR_WHITE, -1, cv2.LINE_AA)

    def _draw_court_zones(self, frame, court_keypoints=None):
        """Draw semi-transparent court zone overlay"""
        # Simplified zones based on frame dimensions
        overlay = frame.copy()

        # Service boxes (approximate)
        h = self.frame_height
        w = self.frame_width

        # Near service box
        pts = np.array([[w//4, h//2], [3*w//4, h//2],
                       [3*w//4, 3*h//4], [w//4, 3*h//4]], np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

        # Far service box
        pts = np.array([[w//4, h//4], [3*w//4, h//4],
                       [3*w//4, h//2], [w//4, h//2]], np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)

        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def update(self, ball_pos, frame_idx, is_shot=False):
        """Update tracking state with new ball position"""
        if ball_pos is None:
            return

        pos = (int(ball_pos[0]), int(ball_pos[1]))

        # Calculate velocity and speed
        velocity, speed_kmh = self._calculate_velocity(pos)

        # Update trajectory
        self.trajectory.append(pos)
        self.velocity_history.append(velocity)
        self.speed_history.append(speed_kmh)

        # Update stats
        if speed_kmh > self.max_speed:
            self.max_speed = speed_kmh

        if self.last_position:
            dist_pixels = math.sqrt((pos[0] - self.last_position[0])**2 +
                                   (pos[1] - self.last_position[1])**2)
            dist_meters = dist_pixels / (self.frame_width / 10.0)
            self.total_distance += dist_meters

        self.last_position = pos

        # Track shots
        if is_shot:
            self.current_shot_frame = frame_idx
            self.rally_count += 1

    def draw(self, frame, frame_idx, show_hud=True, show_zones=False):
        """Draw all visualization effects on frame"""
        if len(self.trajectory) == 0:
            return frame

        pos = self.trajectory[-1]
        velocity = self.velocity_history[-1] if self.velocity_history else (0, 0)
        speed_kmh = self.speed_history[-1] if self.speed_history else 0

        # Court zones (optional, can be distracting)
        if show_zones:
            self._draw_court_zones(frame)

        # Trajectory effects
        self._draw_ghost_balls(frame)
        self._draw_bezier_trail(frame, speed_kmh)

        # Motion effects
        self._draw_motion_blur(frame, pos, velocity, speed_kmh)
        self._draw_particles(frame, pos, velocity, speed_kmh)

        # Prediction arc
        self._draw_prediction_arc(frame, pos, velocity, speed_kmh)

        # Shot impact ripple
        frames_since_shot = frame_idx - self.current_shot_frame
        if frames_since_shot <= 20:
            self._draw_impact_ripple(frame, pos, frames_since_shot)

        # Main ball
        self._draw_glowing_ball(frame, pos, speed_kmh)

        # Speed indicator near ball
        self._draw_speed_indicator(frame, pos, speed_kmh)

        # HUD elements
        if show_hud:
            self._draw_power_meter(frame, speed_kmh)
            self._draw_speed_graph(frame)
            self._draw_stats_hud(frame)

        return frame

    def reset(self):
        """Reset all tracking state"""
        self.trajectory.clear()
        self.velocity_history.clear()
        self.speed_history.clear()
        self.max_speed = 0
        self.rally_count = 0
        self.total_distance = 0
        self.last_position = None
        self.current_shot_frame = -100
