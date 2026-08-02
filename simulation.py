"""
Simulation coordinates rockets, environment, and evolution.
Tracks reward diagnostics for debugging.
"""

from typing import List, Optional, Tuple, Dict
from rocket import Rocket
from environment import Environment
from evolution import Evolution
from neural_network import NeuralNetwork
import config


class Simulation:
    """
    Main simulation controller with diagnostic tracking.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        # Core components
        self.environment = Environment(width, height)
        self.evolution = Evolution(config.POPULATION_SIZE)
        
        # State
        self.rockets: List[Rocket] = []
        self.frame = 0
        self.paused = False
        self.simulation_speed = config.SPEED_OPTIONS[config.DEFAULT_SPEED_INDEX]
        self.speed_index = config.DEFAULT_SPEED_INDEX
        
        # Statistics history
        self.fitness_history: List[dict] = []
        
        # Trail
        self.best_trail: List[Tuple[float, float]] = []
        self.max_trail_length = 60
        self.trail_update_counter = 0
        self.trail_update_interval = 2
        
        # Champion tracking
        self.champion_trail: List[Tuple[float, float]] = []
        self.fastest_ever = float('inf')
        self.fastest_ever_trail: List[Tuple[float, float]] = []
        
        # Debug state
        self.debug_rocket_index: int = 0
        self.show_debug: bool = False
        self.show_sensors: bool = False
        
        # --- NEW: Selected rocket for inspection ---
        self.selected_rocket_index: Optional[int] = None
        self.selected_rocket: Optional[Rocket] = None
        
        # --- NEW: Adaptive generation ---
        self.current_generation_length: int = config.BASE_GENERATION_LENGTH
        self.generation_extended: bool = False
        self.active_rockets_count: int = 0
        
        # --- NEW: Training phase ---
        self.training_phase: str = "early"
        
        # --- NEW: Plateau detection ---
        self.plateau_detected: bool = False
        self.plateau_start_gen: int = 0
        self.plateau_duration: int = 0
        self.recent_fitness_history: List[float] = []
        
        # --- NEW: Trend tracking ---
        self.success_rate_trend: float = 0.0
        self.fitness_trend: float = 0.0
        
        # --- NEW: All-time best ---
        self.all_time_best_success_rate: float = 0.0
        self.all_time_best_fitness: float = float('-inf')
        self.all_time_best_generation: int = 0
        self.all_time_best_trail: List[Tuple[float, float]] = []
        
        # --- NEW: Generation summary ---
        self.last_generation_summary: Optional[Dict] = None
        
        # --- NEW: Turbo mode ---
        self.turbo_mode: bool = False
        self.turbo_frames_per_step: int = 50
        
        # --- NEW: Training reset tracking ---
        self.total_generations_completed: int = 0
        
        self._init_population()
    
    def _init_population(self) -> None:
        """Create initial population of rockets."""
        networks = self.evolution.create_initial_population()
        start_positions = self.environment.get_start_positions(config.POPULATION_SIZE)
        
        self.rockets = []
        for i, nn in enumerate(networks):
            x, y = start_positions[i]
            rocket = Rocket(nn, x, y)
            self.rockets.append(rocket)
        
        self.frame = 0
    
    def update(self) -> None:
        """Update simulation by one visual frame."""
        if self.paused:
            return
        
        if self.turbo_mode:
            frames_to_run = self.turbo_frames_per_step
        else:
            frames_to_run = self.simulation_speed
        
        for _ in range(frames_to_run):
            if self.frame >= self.current_generation_length:
                self._next_generation()
                continue
            
            for rocket in self.rockets:
                if rocket.alive:
                    sensor_readings = self.environment.get_all_sensor_readings(
                        rocket.x, rocket.y, rocket.rotation
                    )
                    rocket.update(
                        self.environment.target_x,
                        self.environment.target_y,
                        self.width,
                        self.height,
                        self.environment.obstacles if self.environment.obstacles else None,
                        sensor_readings
                    )
            
            self.active_rockets_count = sum(1 for r in self.rockets if r.alive)
            
            if self.active_rockets_count == 0:
                self._next_generation()
                continue
            
            self._check_early_termination()
            
            self.frame += 1
        
        self._update_trails()
        self._update_selected_rocket()
    
    def _check_early_termination(self) -> None:
        """Check if generation should end early based on activity."""
        if self.frame < 30:
            return
        
        reached_count = sum(1 for r in self.rockets if r.reached_target)
        alive_count = sum(1 for r in self.rockets if r.alive)
        
        if reached_count > config.POPULATION_SIZE * 0.85:
            self._next_generation()
            return
        
        if alive_count < 3 and self.frame > 60:
            self._next_generation()
            return
    
    def _check_extension_qualification(self) -> bool:
        """Check if active rockets deserve extended time."""
        active_rockets = [r for r in self.rockets if r.alive]
        
        if len(active_rockets) < 3:
            return False
        
        for rocket in active_rockets:
            if len(rocket.distance_history) >= 10:
                recent_dists = list(rocket.distance_history)[-10:]
                improvement = recent_dists[0] - recent_dists[-1]
                if improvement > config.EXTENSION_PROGRESS_THRESHOLD:
                    return True
        
        return False
    
    def _update_trails(self) -> None:
        """Update visualization trails."""
        self.trail_update_counter += 1
        if self.trail_update_counter < self.trail_update_interval:
            return
        self.trail_update_counter = 0
        
        best_rocket = self._get_best_alive_rocket()
        if best_rocket:
            self.best_trail.append((best_rocket.x, best_rocket.y))
            if len(self.best_trail) > self.max_trail_length:
                self.best_trail.pop(0)
    
    def _update_selected_rocket(self) -> None:
        """Update the selected rocket reference."""
        if self.selected_rocket_index is not None:
            alive = [r for r in self.rockets if r.alive]
            if 0 <= self.selected_rocket_index < len(alive):
                self.selected_rocket = alive[self.selected_rocket_index]
            else:
                self.selected_rocket = None
        else:
            self.selected_rocket = None
    
    def select_rocket_at(self, click_x: float, click_y: float) -> bool:
        """Try to select a rocket near the click position."""
        alive = [r for r in self.rockets if r.alive]
        
        min_dist = 50
        best_idx = None
        
        for i, rocket in enumerate(alive):
            dx = rocket.x - click_x
            dy = rocket.y - click_y
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        if best_idx is not None:
            self.selected_rocket_index = best_idx
            self.selected_rocket = alive[best_idx]
            return True
        else:
            self.deselect_rocket()
            return False
    
    def deselect_rocket(self) -> None:
        """Deselect the currently selected rocket."""
        self.selected_rocket_index = None
        self.selected_rocket = None
    
    def _next_generation(self) -> None:
        """Evolve to the next generation."""
        fitnesses = []
        for rocket in self.rockets:
            rocket.calculate_fitness()
            fitnesses.append(rocket.fitness)
        
        num_reached = sum(1 for r in self.rockets if r.reached_target)
        num_crashed = sum(1 for r in self.rockets if r.crashed)
        num_stuck = sum(1 for r in self.rockets if r.stuck_counter > 20)
        
        completion_times = [r.frame_reached for r in self.rockets 
                          if r.reached_target and r.frame_reached > 0]
        
        best_fitness = max(fitnesses) if fitnesses else 0
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
        success_rate = num_reached / config.POPULATION_SIZE * 100
        
        if success_rate > self.all_time_best_success_rate:
            self.all_time_best_success_rate = success_rate
            self.all_time_best_fitness = best_fitness
            self.all_time_best_generation = self.evolution.generation
            best_rocket = max(
                [r for r in self.rockets if r.reached_target],
                key=lambda r: r.fitness,
                default=None
            )
            if best_rocket:
                self.all_time_best_trail = list(best_rocket.trail)
        
        if completion_times:
            fastest_this_gen = min(completion_times)
            best_rocket = min(
                [r for r in self.rockets if r.reached_target and r.frame_reached > 0],
                key=lambda r: r.frame_reached
            )
            self.champion_trail = list(best_rocket.trail)
            if fastest_this_gen < self.fastest_ever:
                self.fastest_ever = fastest_this_gen
                self.fastest_ever_trail = list(best_rocket.trail)
        
        stats = self.evolution.get_statistics(fitnesses, num_reached)
        stats["num_crashed"] = num_crashed
        stats["num_alive"] = sum(1 for r in self.rockets if r.alive)
        stats["num_stuck"] = num_stuck
        stats["generation_length"] = self.current_generation_length
        
        if completion_times:
            stats["best_completion"] = min(completion_times)
            stats["avg_completion"] = sum(completion_times) / len(completion_times)
            stats["fastest_ever"] = self.fastest_ever
        else:
            stats["best_completion"] = None
            stats["avg_completion"] = None
            stats["fastest_ever"] = self.fastest_ever if self.fastest_ever < float('inf') else None
        
        avg_path_efficiency = 0.0
        path_effs = [r.get_path_efficiency() for r in self.rockets if r.reached_target]
        if path_effs:
            avg_path_efficiency = sum(path_effs) / len(path_effs)
        stats["avg_path_efficiency"] = avg_path_efficiency
        
        self.fitness_history.append(stats)
        if len(self.fitness_history) > config.GRAPH_HISTORY_LENGTH:
            self.fitness_history.pop(0)
        
        self.recent_fitness_history.append(best_fitness)
        if len(self.recent_fitness_history) > config.PLATEAU_WINDOW:
            self.recent_fitness_history.pop(0)
        
        self._update_training_phase()
        self._update_trend()
        self._update_plateau_detection()
        
        self.last_generation_summary = {
            "generation": self.evolution.generation,
            "success_rate": success_rate,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "num_reached": num_reached,
            "num_crashed": num_crashed,
            "num_stuck": num_stuck,
            "fastest": min(completion_times) if completion_times else None,
            "path_efficiency": avg_path_efficiency,
        }
        
        networks = [r.nn for r in self.rockets]
        new_networks = self.evolution.evolve(networks, fitnesses)
        
        self.environment.update_generation(self.evolution.generation)
        
        self.environment.update_curriculum(success_rate)
        
        start_positions = self.environment.get_start_positions(config.POPULATION_SIZE)
        self.rockets = []
        for i, nn in enumerate(new_networks):
            x, y = start_positions[i]
            rocket = Rocket(nn, x, y)
            self.rockets.append(rocket)
        
        self.best_trail = []
        self.frame = 0
        self.total_generations_completed += 1
        
        self._update_generation_length()
        
        self.deselect_rocket()
    
    def _update_training_phase(self) -> None:
        """Update the training phase label."""
        gen = self.evolution.generation
        if gen <= config.EARLY_GEN_THRESHOLD:
            self.training_phase = "early"
        elif gen <= config.MID_GEN_THRESHOLD:
            self.training_phase = "mid"
        else:
            self.training_phase = "late"
    
    def _update_generation_length(self) -> None:
        """Update generation length based on training phase."""
        gen = self.evolution.generation
        
        if gen <= config.EARLY_GEN_THRESHOLD:
            self.current_generation_length = config.EARLY_GENERATION_LENGTH
        elif gen <= config.MID_GEN_THRESHOLD:
            self.current_generation_length = config.MID_GENERATION_LENGTH
        else:
            self.current_generation_length = config.LATE_GENERATION_LENGTH
        
        if self.frame >= self.current_generation_length - 50:
            if self._check_extension_qualification():
                self.current_generation_length = min(
                    self.current_generation_length + config.MAX_EXTENSION_LENGTH,
                    config.EARLY_GENERATION_LENGTH + config.MAX_EXTENSION_LENGTH
                )
                self.generation_extended = True
            else:
                self.generation_extended = False
    
    def _update_trend(self) -> None:
        """Update trend indicators based on recent history."""
        if len(self.fitness_history) < config.TREND_WINDOW:
            self.success_rate_trend = 0.0
            self.fitness_trend = 0.0
            return
        
        recent = self.fitness_history[-config.TREND_WINDOW:]
        half = len(recent) // 2
        
        first_half = recent[:half]
        second_half = recent[half:]
        
        first_sr = sum(s.get("success_rate", 0) for s in first_half) / len(first_half)
        second_sr = sum(s.get("success_rate", 0) for s in second_half) / len(second_half)
        
        first_fit = sum(s.get("best_fitness", 0) for s in first_half) / len(first_half)
        second_fit = sum(s.get("best_fitness", 0) for s in second_half) / len(second_half)
        
        self.success_rate_trend = second_sr - first_sr
        self.fitness_trend = second_fit - first_fit
    
    def _update_plateau_detection(self) -> None:
        """Detect training plateau based on fitness history."""
        if len(self.recent_fitness_history) < config.PLATEAU_WINDOW:
            self.plateau_detected = False
            return
        
        fitnesses = self.recent_fitness_history
        min_f = min(fitnesses)
        max_f = max(fitnesses)
        
        if min_f == 0:
            fitness_range = 1.0
        else:
            fitness_range = abs(max_f - min_f) / abs(min_f) * 100
        
        if fitness_range < config.PLATEAU_FITNESS_THRESHOLD:
            if not self.plateau_detected:
                self.plateau_detected = True
                self.plateau_start_gen = self.evolution.generation
                self.plateau_duration = 1
            else:
                self.plateau_duration = self.evolution.generation - self.plateau_start_gen
        else:
            self.plateau_detected = False
            self.plateau_duration = 0
    
    def _get_best_alive_rocket(self) -> Optional[Rocket]:
        """Find the best alive rocket."""
        alive = [r for r in self.rockets if r.alive]
        if not alive:
            return None
        return min(alive, key=lambda r: r.distance_to_target)
    
    def get_debug_rocket(self) -> Optional[Rocket]:
        """Get rocket for debug display (closest to target among alive)."""
        if self.selected_rocket and self.selected_rocket.alive:
            return self.selected_rocket
        
        alive = [r for r in self.rockets if r.alive]
        if not alive:
            return None
        alive.sort(key=lambda r: r.distance_to_target)
        idx = min(self.debug_rocket_index, len(alive) - 1)
        return alive[idx]
    
    def get_reward_diagnostics(self) -> Dict[str, float]:
        """Get average reward components for current generation."""
        if not self.rockets:
            return {}
        
        rocket = self.get_debug_rocket()
        if rocket is None:
            return {}
        
        return {
            "Progress": rocket.progress_reward,
            "Danger": rocket.danger_penalty,
            "Stuck": rocket.stuck_penalty,
            "Recovery": rocket.recovery_bonus,
            "Total": rocket.total_reward,
        }
    
    def set_target(self, x: float, y: float) -> None:
        """Set target position."""
        self.environment.set_target(x, y)
    
    def set_random_target(self) -> None:
        """Place target at random position."""
        self.environment.set_random_target()
    
    def cycle_obstacle_mode(self) -> str:
        """Cycle obstacle modes."""
        return self.environment.cycle_obstacle_mode()
    
    def set_obstacle_mode(self, mode: str) -> None:
        """Set a specific obstacle mode."""
        self.environment.set_obstacle_mode(mode)
    
    def generate_new_random_layout(self) -> None:
        """Generate new random obstacle layout."""
        self.environment.generate_new_random_layout()
    
    def toggle_generalization(self) -> None:
        """Toggle generalization mode."""
        self.environment.generalization_mode = not self.environment.generalization_mode
    
    def toggle_curriculum(self) -> None:
        """Toggle curriculum mode."""
        self.environment.curriculum_mode = not self.environment.curriculum_mode
        if self.environment.curriculum_mode:
            self.environment.reset_curriculum()
    
    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self.paused = not self.paused
    
    def toggle_debug(self) -> None:
        """Toggle debug visualization."""
        self.show_debug = not self.show_debug
    
    def toggle_sensors(self) -> None:
        """Toggle sensor ray visualization."""
        self.show_sensors = not self.show_sensors
    
    def toggle_turbo(self) -> None:
        """Toggle turbo training mode."""
        self.turbo_mode = not self.turbo_mode
    
    def reset(self) -> None:
        """Reset entire simulation (full training reset)."""
        self.environment = Environment(self.width, self.height)
        self.evolution = Evolution(config.POPULATION_SIZE)
        self.fitness_history = []
        self.best_trail = []
        self.champion_trail = []
        self.fastest_ever = float('inf')
        self.fastest_ever_trail = []
        self.current_generation_length = config.BASE_GENERATION_LENGTH
        self.generation_extended = False
        self.training_phase = "early"
        self.plateau_detected = False
        self.plateau_duration = 0
        self.recent_fitness_history = []
        self.success_rate_trend = 0.0
        self.fitness_trend = 0.0
        self.all_time_best_success_rate = 0.0
        self.all_time_best_fitness = float('-inf')
        self.all_time_best_generation = 0
        self.all_time_best_trail = []
        self.last_generation_summary = None
        self.total_generations_completed = 0
        self._init_population()
    
    def increase_speed(self) -> None:
        """Increase simulation speed."""
        if self.speed_index < len(config.SPEED_OPTIONS) - 1:
            self.speed_index += 1
            self.simulation_speed = config.SPEED_OPTIONS[self.speed_index]
    
    def decrease_speed(self) -> None:
        """Decrease simulation speed."""
        if self.speed_index > 0:
            self.speed_index -= 1
            self.simulation_speed = config.SPEED_OPTIONS[self.speed_index]
    
    def save_best_network(self) -> str:
        """Save the best network to file."""
        import os
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        filepath = os.path.join(config.SAVE_DIR, config.SAVE_FILE)
        
        if self.evolution.best_network_ever is not None:
            self.evolution.best_network_ever.save(filepath)
        elif self.rockets:
            best = max(self.rockets, key=lambda r: r.fitness)
            best.nn.save(filepath)
        
        return filepath
    
    def load_best_network(self) -> bool:
        """Load a saved network."""
        import os
        filepath = os.path.join(config.SAVE_DIR, config.SAVE_FILE)
        
        if not os.path.exists(filepath):
            return False
        
        try:
            nn = NeuralNetwork([
                config.INPUT_NEURONS,
                config.HIDDEN1_NEURONS,
                config.HIDDEN2_NEURONS,
                config.OUTPUT_NEURONS
            ])
            nn.load(filepath)
            self.evolution.best_network_ever = nn
            self.evolution.best_fitness_ever = 0
            return True
        except Exception as e:
            print(f"Error loading network: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get current simulation statistics."""
        alive_count = sum(1 for r in self.rockets if r.alive)
        reached_count = sum(1 for r in self.rockets if r.reached_target)
        crashed_count = sum(1 for r in self.rockets if r.crashed)
        stuck_count = sum(1 for r in self.rockets if r.stuck_counter > 20)
        
        rewards = [r.total_reward for r in self.rockets]
        best_fitness = max(rewards) if rewards else 0
        avg_fitness = sum(rewards) / len(rewards) if rewards else 0
        
        completion_times = [r.frame_reached for r in self.rockets 
                          if r.reached_target and r.frame_reached > 0]
        
        return {
            "generation": self.evolution.generation,
            "population": config.POPULATION_SIZE,
            "alive": alive_count,
            "reached": reached_count,
            "crashed": crashed_count,
            "stuck": stuck_count,
            "frame": self.frame,
            "generation_length": self.current_generation_length,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "success_rate": reached_count / config.POPULATION_SIZE * 100,
            "best_ever": self.evolution.best_fitness_ever,
            "target_x": self.environment.target_x,
            "target_y": self.environment.target_y,
            "speed": self.simulation_speed,
            "paused": self.paused,
            "obstacle_mode": self.environment.obstacle_mode,
            "generalization": self.environment.generalization_mode,
            "curriculum": self.environment.curriculum_mode,
            "best_completion": min(completion_times) if completion_times else None,
            "avg_completion": sum(completion_times) / len(completion_times) if completion_times else None,
            "fastest_ever": self.fastest_ever if self.fastest_ever < float('inf') else None,
            "show_debug": self.show_debug,
            "show_sensors": self.show_sensors,
            # New stats
            "training_phase": self.training_phase,
            "plateau_detected": self.plateau_detected,
            "plateau_duration": self.plateau_duration,
            "success_rate_trend": self.success_rate_trend,
            "fitness_trend": self.fitness_trend,
            "all_time_best_sr": self.all_time_best_success_rate,
            "all_time_best_gen": self.all_time_best_generation,
            "turbo_mode": self.turbo_mode,
            "selected_rocket": self.selected_rocket is not None,
            "generation_extended": self.generation_extended,
            "total_generations": self.total_generations_completed,
        }
    
    def get_best_network(self) -> Optional[NeuralNetwork]:
        """Get the current best network."""
        if self.evolution.best_network_ever is not None:
            return self.evolution.best_network_ever
        if self.rockets:
            best = max(self.rockets, key=lambda r: r.fitness)
            return best.nn
        return None
    
    def get_champion(self) -> Optional[Rocket]:
        """Get the fastest successful rocket."""
        successful = [r for r in self.rockets if r.reached_target and r.frame_reached > 0]
        if not successful:
            return None
        return min(successful, key=lambda r: r.frame_reached)
