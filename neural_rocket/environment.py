"""
Environment handles target position, obstacles, boundaries, and obstacle sensors.
"""

import random
import math
from typing import List, Optional, Tuple
import config


class Environment:
    """
    Manages the simulation environment including target, obstacles, and bounds.
    """

    def __init__(self, width: int, height: int):
        """
        Initialize environment.
        
        Args:
            width: simulation area width
            height: simulation area height
        """
        self.width = width
        self.height = height
        
        # Target position (center by default)
        self.target_x = width * 0.75
        self.target_y = height * 0.5
        self.target_radius = config.TARGET_RADIUS
        
        # Obstacles
        self.obstacles: List[dict] = []
        self.obstacle_mode = "OFF"
        
        # Training mode
        self.generalization_mode = config.GENERALIZATION_MODE
        self.generation_count = 0
        self.move_interval = config.GENERALIZATION_MOVE_INTERVAL
        
        # Curriculum mode
        self.curriculum_mode = config.CURRICULUM_MODE
        self.curriculum_success_streak = 0
        self.curriculum_difficulty_index = 0
    
    def set_target(self, x: float, y: float) -> None:
        """
        Set target position, clamped to valid bounds.
        
        Args:
            x: target X coordinate
            y: target Y coordinate
        """
        self.target_x = max(self.target_radius, min(x, self.width - self.target_radius))
        self.target_y = max(self.target_radius, min(y, self.height - self.target_radius))
    
    def set_random_target(self) -> None:
        """Place target at a random position in the simulation area."""
        margin = 80
        self.target_x = random.uniform(margin, self.width - margin)
        self.target_y = random.uniform(margin, self.height - margin)
    
    def cycle_obstacle_mode(self) -> str:
        """
        Cycle through obstacle modes.
        
        Returns:
            new obstacle mode name
        """
        modes = config.OBSTACLE_MODES
        current_idx = modes.index(self.obstacle_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.obstacle_mode = modes[next_idx]
        self._update_obstacles()
        return self.obstacle_mode
    
    def set_obstacle_mode(self, mode: str) -> None:
        """Set a specific obstacle mode."""
        if mode in config.OBSTACLE_MODES:
            self.obstacle_mode = mode
            self._update_obstacles()
    
    def _update_obstacles(self) -> None:
        """Update obstacles based on current mode."""
        self.obstacles.clear()
        
        if self.obstacle_mode == "OFF":
            pass
        elif self.obstacle_mode == "SIMPLE":
            self.obstacles = [dict(o) for o in config.SIMPLE_OBSTACLES]
        elif self.obstacle_mode == "MEDIUM":
            self.obstacles = [dict(o) for o in config.MEDIUM_OBSTACLES]
        elif self.obstacle_mode == "HARD":
            self.obstacles = [dict(o) for o in config.HARD_OBSTACLES]
        elif self.obstacle_mode == "MAZE":
            self.obstacles = [dict(o) for o in config.MAZE_OBSTACLES]
        elif self.obstacle_mode == "RANDOM":
            self.generate_random_obstacles()
    
    def generate_random_obstacles(self) -> None:
        """Generate random obstacle layout ensuring a valid path exists."""
        self.obstacles.clear()
        
        num_obstacles = random.randint(config.RANDOM_MIN_OBSTACLES, config.RANDOM_MAX_OBSTACLES)
        
        for _ in range(num_obstacles):
            attempts = 0
            while attempts < 50:
                width = random.randint(config.RANDOM_MIN_WIDTH, config.RANDOM_MAX_WIDTH)
                height = random.randint(config.RANDOM_MIN_HEIGHT, config.RANDOM_MAX_HEIGHT)
                
                # Random position, avoiding start area
                x = random.randint(150, self.width - 100)
                y = random.randint(0, self.height - height)
                
                obs = {"x": x, "y": y, "width": width, "height": height}
                
                # Check if obstacle blocks spawn or target too severely
                if not self._blocks_critical_area(obs):
                    self.obstacles.append(obs)
                    break
                attempts += 1
    
    def _blocks_critical_area(self, obs: dict) -> bool:
        """Check if obstacle blocks spawn or target areas too severely."""
        # Spawn area (left side)
        spawn_x_max = config.START_X_MAX + 30
        if obs["x"] < spawn_x_max and obs["x"] + obs["width"] > config.START_X_MIN - 30:
            if obs["y"] < config.START_Y_MAX and obs["y"] + obs["height"] > config.START_Y_MIN:
                return True
        
        # Target area
        target_margin = config.TARGET_RADIUS + 30
        if (abs(obs["x"] + obs["width"] / 2 - self.target_x) < target_margin and
            abs(obs["y"] + obs["height"] / 2 - self.target_y) < target_margin):
            return True
        
        return False
    
    def generate_new_random_layout(self) -> None:
        """Generate a new random layout (called with N key)."""
        if self.obstacle_mode == "RANDOM":
            self.generate_random_obstacles()
    
    def toggle_obstacles(self) -> str:
        """Toggle obstacles - returns new mode."""
        self.cycle_obstacle_mode()
        return self.obstacle_mode
    
    def get_start_positions(self, count: int) -> List[Tuple[float, float]]:
        """
        Generate starting positions for rockets.
        All rockets spawn from the left side of the screen.
        
        Args:
            count: number of positions to generate
        
        Returns:
            list of (x, y) tuples
        """
        positions = []
        for _ in range(count):
            x = random.uniform(config.START_X_MIN, config.START_X_MAX)
            y = random.uniform(config.START_Y_MIN, config.START_Y_MAX)
            positions.append((x, y))
        return positions
    
    def is_point_in_obstacle(self, x: float, y: float) -> bool:
        """Check if a point collides with any obstacle."""
        if not self.obstacles:
            return False
        
        for obs in self.obstacles:
            if (obs["x"] <= x <= obs["x"] + obs["width"] and
                obs["y"] <= y <= obs["y"] + obs["height"]):
                return True
        return False
    
    def cast_sensor_ray(self, x: float, y: float, angle: float, 
                        max_range: float = config.SENSOR_RANGE) -> float:
        """
        Cast a ray from a point and return normalized distance to obstacle.
        
        Args:
            x: starting X position
            y: starting Y position
            angle: ray angle in radians (relative to rocket orientation)
            max_range: maximum sensor range
        
        Returns:
            normalized distance (0.0 = very close, 1.0 = no obstacle)
        """
        if not self.obstacles:
            return 1.0
        
        # Step along ray
        step_size = 5.0
        steps = int(max_range / step_size)
        
        for i in range(1, steps + 1):
            check_x = x + math.cos(angle) * (i * step_size)
            check_y = y + math.sin(angle) * (i * step_size)
            
            # Check boundary
            if check_x < 0 or check_x > self.width or check_y < 0 or check_y > self.height:
                return i / steps
            
            # Check obstacles
            for obs in self.obstacles:
                if (obs["x"] <= check_x <= obs["x"] + obs["width"] and
                    obs["y"] <= check_y <= obs["y"] + obs["height"]):
                    return i / steps
        
        return 1.0
    
    def get_all_sensor_readings(self, x: float, y: float, 
                                 orientation: float) -> List[float]:
        """
        Get all sensor readings for a rocket.
        
        Args:
            x: rocket X position
            y: rocket Y position
            orientation: rocket facing direction in radians
        
        Returns:
            list of 5 normalized sensor values
        """
        readings = []
        for angle_offset in config.SENSOR_ANGLES:
            ray_angle = orientation + angle_offset
            distance = self.cast_sensor_ray(x, y, ray_angle)
            readings.append(distance)
        return readings
    
    def update_generation(self, generation: int) -> None:
        """
        Update environment based on generation (for generalization mode).
        
        Args:
            generation: current generation number
        """
        self.generation_count = generation
        
        if self.generalization_mode:
            if generation > 0 and generation % self.move_interval == 0:
                self.set_random_target()
    
    def update_curriculum(self, success_rate: float) -> bool:
        """
        Update curriculum difficulty based on success rate.
        
        Args:
            success_rate: percentage of rockets reaching target
        
        Returns:
            True if difficulty was increased
        """
        if not self.curriculum_mode:
            return False
        
        if success_rate >= config.CURRICULUM_SUCCESS_THRESHOLD * 100:
            self.curriculum_success_streak += 1
            if self.curriculum_success_streak >= config.CURRICULUM_REQUIRED_GENERATIONS:
                self.curriculum_success_streak = 0
                self.curriculum_difficulty_index += 1
                if self.curriculum_difficulty_index >= len(config.OBSTACLE_MODES):
                    self.curriculum_difficulty_index = 0
                self.obstacle_mode = config.OBSTACLE_MODES[self.curriculum_difficulty_index]
                self._update_obstacles()
                return True
        else:
            self.curriculum_success_streak = 0
        
        return False
    
    def reset_curriculum(self) -> None:
        """Reset curriculum state."""
        self.curriculum_success_streak = 0
        self.curriculum_difficulty_index = 0
        self.obstacle_mode = "OFF"
        self._update_obstacles()
