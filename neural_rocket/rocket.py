"""
Rocket agent with improved reward system for obstacle learning.
Includes danger calculation, stuck detection, progress tracking.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
from neural_network import NeuralNetwork
import config


class Rocket:
    """
    A rocket agent with obstacle-aware reward system.
    """

    def __init__(self, nn: Optional[NeuralNetwork] = None, 
                 start_x: float = 100, start_y: float = 350):
        # Position and movement
        self.x = start_x
        self.y = start_y
        self.start_x = start_x
        self.start_y = start_y
        self.vx = 0.0
        self.vy = 0.0
        self.rotation = -math.pi / 2
        
        # State
        self.alive = True
        self.reached_target = False
        self.crashed = False
        self.frame_reached = -1
        self.distance_to_target = 0.0
        self.initial_distance = 0.0
        
        # Neural network outputs (last decision)
        self.last_turn: float = 0.0
        self.last_thrust: float = 0.0
        
        # Path tracking
        self.total_distance_traveled: float = 0.0
        self.last_x: float = start_x
        self.last_y: float = start_y
        
        # Sensor readings
        self.sensor_readings: List[float] = [1.0] * 5
        
        # --- PROGRESS TRACKING ---
        self.position_history: deque = deque(maxlen=config.STUCK_WINDOW)
        self.distance_history: deque = deque(maxlen=config.PROGRESS_WINDOW)
        self.recent_progress: float = 0.0
        
        # --- STUCK DETECTION ---
        self.stuck_counter: int = 0
        self.stuck_level: float = 0.0
        self.was_stuck: bool = False
        self.recovery_cooldown: int = 0
        
        # --- DANGER TRACKING ---
        self.obstacle_danger: float = 0.0
        self.approach_velocity: float = 0.0
        
        # --- REWARD TRACKING ---
        self.total_reward: float = 0.0
        self.fitness: float = 0.0
        self.frame_count: int = 0
        self.progress_reward: float = 0.0
        self.danger_penalty: float = 0.0
        self.stuck_penalty: float = 0.0
        self.recovery_bonus: float = 0.0
        
        # Trail
        self.trail: List[Tuple[float, float]] = []
        self.max_trail_length = 40
        
        # Neural network
        if nn is not None:
            self.nn = nn
        else:
            self.nn = NeuralNetwork([
                config.INPUT_NEURONS,
                config.HIDDEN1_NEURONS,
                config.HIDDEN2_NEURONS,
                config.OUTPUT_NEURONS
            ])
    
    def reset(self, start_x: float, start_y: float) -> None:
        """Reset rocket state for new generation."""
        self.x = start_x
        self.y = start_y
        self.start_x = start_x
        self.start_y = start_y
        self.vx = 0.0
        self.vy = 0.0
        self.rotation = -math.pi / 2
        self.alive = True
        self.reached_target = False
        self.crashed = False
        self.frame_reached = -1
        self.distance_to_target = 0.0
        self.initial_distance = 0.0
        self.last_turn = 0.0
        self.last_thrust = 0.0
        self.total_distance_traveled = 0.0
        self.last_x = start_x
        self.last_y = start_y
        self.sensor_readings = [1.0] * 5
        self.position_history.clear()
        self.distance_history.clear()
        self.recent_progress = 0.0
        self.stuck_counter = 0
        self.stuck_level = 0.0
        self.was_stuck = False
        self.recovery_cooldown = 0
        self.obstacle_danger = 0.0
        self.approach_velocity = 0.0
        self.total_reward = 0.0
        self.fitness = 0.0
        self.frame_count = 0
        self.progress_reward = 0.0
        self.danger_penalty = 0.0
        self.stuck_penalty = 0.0
        self.recovery_bonus = 0.0
        self.trail = []
    
    def get_inputs(self, target_x: float, target_y: float,
                   screen_width: int, screen_height: int) -> np.ndarray:
        """Compute normalized inputs for the neural network."""
        nx = self.x / screen_width
        ny = self.y / screen_height
        
        nvx = self.vx / config.MAX_SPEED if config.MAX_SPEED > 0 else 0
        nvy = self.vy / config.MAX_SPEED if config.MAX_SPEED > 0 else 0
        
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        max_dist = math.sqrt(screen_width ** 2 + screen_height ** 2)
        
        ntx = dx / max_dist
        nty = dy / max_dist
        normalized_dist = min(dist / max_dist, 1.0)
        
        angle_to_target = math.atan2(dy, dx)
        angle_diff = angle_to_target - self.rotation
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        normalized_angle = angle_diff / math.pi
        
        normalized_rotation = self.rotation / math.pi
        
        inputs = np.array([
            nx, ny,
            nvx, nvy,
            ntx, nty,
            normalized_dist,
            normalized_angle,
            normalized_rotation,
        ], dtype=np.float64)
        
        inputs = np.concatenate([inputs, np.array(self.sensor_readings, dtype=np.float64)])
        inputs = np.concatenate([inputs, np.array([self.recent_progress], dtype=np.float64)])
        inputs = np.concatenate([inputs, np.array([self.obstacle_danger], dtype=np.float64)])
        
        return inputs
    
    def think(self, target_x: float, target_y: float,
              screen_width: int, screen_height: int) -> Tuple[float, float]:
        """Use neural network to decide action."""
        inputs = self.get_inputs(target_x, target_y, screen_width, screen_height)
        outputs = self.nn.forward(inputs)
        
        turn = float(outputs[0, 0])
        thrust = (float(outputs[0, 1]) + 1.0) / 2.0
        
        self.last_turn = turn
        self.last_thrust = thrust
        
        return turn, thrust
    
    def update(self, target_x: float, target_y: float,
               screen_width: int, screen_height: int,
               obstacles: Optional[List[dict]] = None,
               sensor_readings: Optional[List[float]] = None) -> None:
        """Update rocket physics and state for one frame."""
        if not self.alive:
            return
        
        self.frame_count += 1
        
        self.position_history.append((self.x, self.y))
        
        if sensor_readings is not None:
            self.sensor_readings = sensor_readings
        
        self._update_danger_and_approach(target_x, target_y)
        
        turn, thrust = self.think(target_x, target_y, screen_width, screen_height)
        
        self.rotation += turn * config.ROTATION_SPEED
        while self.rotation > math.pi:
            self.rotation -= 2 * math.pi
        while self.rotation < -math.pi:
            self.rotation += 2 * math.pi
        
        if thrust > 0.15:
            ax = math.cos(self.rotation) * thrust * config.THRUST_POWER
            ay = math.sin(self.rotation) * thrust * config.THRUST_POWER
            self.vx += ax
            self.vy += ay
        
        self.vx *= config.FRICTION
        self.vy *= config.FRICTION
        
        if abs(self.vx) < config.MIN_SPEED_THRESHOLD:
            self.vx = 0.0
        if abs(self.vy) < config.MIN_SPEED_THRESHOLD:
            self.vy = 0.0
        
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed > config.MAX_SPEED:
            scale = config.MAX_SPEED / speed
            self.vx *= scale
            self.vy *= scale
        
        self.x += self.vx
        self.y += self.vy
        
        # Track distance traveled
        dx_move = self.x - self.last_x
        dy_move = self.y - self.last_y
        self.total_distance_traveled += math.sqrt(dx_move * dx_move + dy_move * dy_move)
        self.last_x = self.x
        self.last_y = self.y
        
        # Soft boundary
        margin = config.ROCKET_SIZE
        if self.x < margin:
            self.x = margin
            self.vx = abs(self.vx) * 0.5
        elif self.x > screen_width - margin:
            self.x = screen_width - margin
            self.vx = -abs(self.vx) * 0.5
        if self.y < margin:
            self.y = margin
            self.vy = abs(self.vy) * 0.5
        elif self.y > screen_height - margin:
            self.y = screen_height - margin
            self.vy = -abs(self.vy) * 0.5
        
        # Trail
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed > 0.05:
            self.trail.append((self.x, self.y))
            if len(self.trail) > self.max_trail_length:
                self.trail.pop(0)
        
        # Calculate distance
        dx = target_x - self.x
        dy = target_y - self.y
        self.distance_to_target = math.sqrt(dx * dx + dy * dy)
        
        if self.frame_count == 1:
            self.initial_distance = self.distance_to_target
        
        self.distance_history.append(self.distance_to_target)
        
        self._update_progress()
        self._update_stuck_state()
        
        if self.distance_to_target < config.TARGET_RADIUS:
            self.reached_target = True
            self.frame_reached = self.frame_count
            self.alive = False
            return
        
        self._calculate_rewards(target_x, target_y)
        
        if obstacles:
            for obs in obstacles:
                if self._point_in_rect(self.x, self.y, obs):
                    self.crashed = True
                    self.alive = False
                    self.total_reward += config.OBSTACLE_CRASH_PENALTY
                    return
    
    def _update_danger_and_approach(self, target_x: float, target_y: float) -> None:
        """Calculate obstacle danger and approach velocity from sensors."""
        if not self.sensor_readings:
            self.obstacle_danger = 0.0
            self.approach_velocity = 0.0
            return
        
        min_sensor = min(self.sensor_readings)
        
        if min_sensor < 0.5:
            self.obstacle_danger = (0.5 - min_sensor) / 0.5
            self.obstacle_danger = self.obstacle_danger ** 1.5
        else:
            self.obstacle_danger = 0.0
        
        front_sensor = self.sensor_readings[2] if len(self.sensor_readings) > 2 else 1.0
        
        if front_sensor < 0.5:
            speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            if speed > 0.1:
                heading_vx = math.cos(self.rotation)
                heading_vy = math.sin(self.rotation)
                self.approach_velocity = (self.vx * heading_vx + self.vy * heading_vy) / speed
                self.approach_velocity = max(0, self.approach_velocity)
            else:
                self.approach_velocity = 0.0
        else:
            self.approach_velocity = 0.0
    
    def _update_progress(self) -> None:
        """Calculate recent progress over rolling window."""
        if len(self.distance_history) < 2:
            self.recent_progress = 0.0
            return
        
        old_dist = self.distance_history[0]
        new_dist = self.distance_history[-1]
        progress = old_dist - new_dist
        
        self.recent_progress = max(-1.0, min(1.0, progress / config.SENSOR_RANGE))
    
    def _update_stuck_state(self) -> None:
        """Detect if rocket is stuck based on position and progress."""
        if len(self.position_history) < config.STUCK_WINDOW:
            self.stuck_counter = max(0, self.stuck_counter - 1)
            self.stuck_level = 0.0
            return
        
        old_pos = self.position_history[0]
        new_pos = self.position_history[-1]
        displacement = math.sqrt(
            (new_pos[0] - old_pos[0]) ** 2 + 
            (new_pos[1] - old_pos[1]) ** 2
        )
        
        is_displacement_stuck = displacement < config.STUCK_DISPLACEMENT_THRESHOLD
        is_progress_stuck = self.recent_progress < config.STUCK_PROGRESS_THRESHOLD
        is_obstacle_near = self.obstacle_danger > 0.1
        
        if is_displacement_stuck and is_progress_stuck and is_obstacle_near:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        max_counter = int(config.STUCK_MAX_PENALTY / config.STUCK_PENALTY_PER_FRAME)
        self.stuck_counter = min(self.stuck_counter, max_counter)
        
        self.stuck_level = min(1.0, self.stuck_counter / (max_counter * 0.5))
        
        if self.was_stuck and self.stuck_counter == 0 and self.recovery_cooldown <= 0:
            self.recovery_bonus += config.RECOVERY_BONUS
            self.recovery_cooldown = config.RECOVERY_COOLDOWN
            self.was_stuck = False
        
        if self.stuck_counter > 10:
            self.was_stuck = True
        
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
    
    def _calculate_rewards(self, target_x: float, target_y: float) -> None:
        """Calculate all reward components."""
        self.progress_reward = 0.0
        self.danger_penalty = 0.0
        self.stuck_penalty = 0.0
        self.recovery_bonus = 0.0
        
        if self.recent_progress > 0:
            self.progress_reward = self.recent_progress * config.PROGRESS_REWARD_SCALE
        else:
            self.progress_reward = self.recent_progress * config.NO_PROGRESS_PENALTY
        
        if self.obstacle_danger > 0:
            base_penalty = self.obstacle_danger * config.DANGER_PENALTY_SCALE
            approach_penalty = self.approach_velocity * config.DANGER_APPROACH_PENALTY * self.obstacle_danger
            self.danger_penalty = -(base_penalty + approach_penalty)
        
        if self.stuck_counter > 0:
            self.stuck_penalty = -min(
                self.stuck_counter * config.STUCK_PENALTY_PER_FRAME,
                config.STUCK_MAX_PENALTY
            )
        
        self.total_reward += self.progress_reward
        self.total_reward += self.danger_penalty
        self.total_reward += self.stuck_penalty
        self.total_reward += self.recovery_bonus
    
    def _point_in_rect(self, px: float, py: float, rect: dict) -> bool:
        """Check if a point is inside a rectangle."""
        return (rect["x"] <= px <= rect["x"] + rect["width"] and
                rect["y"] <= py <= rect["y"] + rect["height"])
    
    def calculate_fitness(self) -> float:
        """Calculate final fitness score."""
        fitness = self.total_reward
        
        fitness -= self.frame_count * 0.02
        
        if self.reached_target and self.frame_reached > 0:
            speed_ratio = (config.BASE_GENERATION_LENGTH - self.frame_reached) / config.BASE_GENERATION_LENGTH
            speed_bonus = speed_ratio * config.SPEED_REWARD
            fitness += config.TARGET_REWARD + speed_bonus
        
        return fitness
    
    def get_path_efficiency(self) -> float:
        """Calculate path efficiency (0-1, 1 = straight line)."""
        if self.total_distance_traveled <= 0:
            return 0.0
        
        straight_distance = self.initial_distance
        if straight_distance <= 0:
            return 1.0
        
        efficiency = straight_distance / self.total_distance_traveled
        return min(1.0, efficiency)
    
    def get_speed(self) -> float:
        """Get current speed."""
        return math.sqrt(self.vx ** 2 + self.vy ** 2)
    
    def get_vertices(self) -> List[Tuple[float, float]]:
        """Get triangle vertices for rendering."""
        size = config.ROCKET_SIZE
        
        tip_x = self.x + math.cos(self.rotation) * size
        tip_y = self.y + math.sin(self.rotation) * size
        
        left_angle = self.rotation + math.pi * 0.8
        right_angle = self.rotation - math.pi * 0.8
        back_size = size * 0.5
        
        left_x = self.x + math.cos(left_angle) * back_size
        left_y = self.y + math.sin(left_angle) * back_size
        
        right_x = self.x + math.cos(right_angle) * back_size
        right_y = self.y + math.sin(right_angle) * back_size
        
        return [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)]
    
    def get_debug_info(self) -> Dict[str, float]:
        """Get debug information for visualization."""
        return {
            "progress": self.recent_progress,
            "danger": self.obstacle_danger,
            "stuck_level": self.stuck_level,
            "stuck_counter": self.stuck_counter,
            "approach_vel": self.approach_velocity,
            "total_reward": self.total_reward,
            "progress_reward": self.progress_reward,
            "danger_penalty": self.danger_penalty,
            "stuck_penalty": self.stuck_penalty,
            "recovery_bonus": self.recovery_bonus,
        }
    
    def get_network_inputs(self, target_x: float, target_y: float,
                          screen_width: int, screen_height: int) -> Dict[str, float]:
        """Get the actual normalized inputs sent to the neural network."""
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        max_dist = math.sqrt(screen_width ** 2 + screen_height ** 2)
        
        angle_to_target = math.atan2(dy, dx)
        angle_diff = angle_to_target - self.rotation
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        return {
            "Rocket X": self.x / screen_width,
            "Rocket Y": self.y / screen_height,
            "Velocity X": self.vx / config.MAX_SPEED,
            "Velocity Y": self.vy / config.MAX_SPEED,
            "Target Rel X": dx / max_dist,
            "Target Rel Y": dy / max_dist,
            "Target Dist": min(dist / max_dist, 1.0),
            "Target Angle": angle_diff / math.pi,
            "Orientation": self.rotation / math.pi,
        }
    
    def get_sensor_inputs(self) -> Dict[str, float]:
        """Get the actual sensor values sent to the neural network."""
        names = ["Front-Left", "Left", "Front", "Right", "Front-Right"]
        result = {}
        for i, name in enumerate(names):
            if i < len(self.sensor_readings):
                result[name] = self.sensor_readings[i]
            else:
                result[name] = 1.0
        return result
