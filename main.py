"""
Neural Rocket Evolution - Main Entry Point
With debug visualization and reward diagnostics.
"""

import pygame
import sys
import math
from typing import List, Tuple, Optional
from simulation import Simulation
from neural_network import NeuralNetwork
import config


class App:
    """Main application with debug visualization."""

    def __init__(self):
        pygame.init()
        pygame.font.init()
        
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("Neural Rocket Evolution")
        
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_title = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_label = pygame.font.SysFont("Consolas", 13)
        self.font_value = pygame.font.SysFont("Consolas", 13, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 11)
        self.font_debug = pygame.font.SysFont("Consolas", 12)
        self.font_micro = pygame.font.SysFont("Consolas", 10)
        
        # Simulation
        self.sim = Simulation(config.SIMULATION_AREA_WIDTH, config.SIMULATION_AREA_HEIGHT)
        
        # UI state
        self.show_help = False
        self.show_nn_inspector = False
        
        # Surfaces for alpha blending
        self.trail_surface = pygame.Surface(
            (config.SIMULATION_AREA_WIDTH, config.SIMULATION_AREA_HEIGHT), 
            pygame.SRCALPHA
        )
        self.rocket_surface = pygame.Surface(
            (config.SIMULATION_AREA_WIDTH, config.SIMULATION_AREA_HEIGHT),
            pygame.SRCALPHA
        )
        self.debug_surface = pygame.Surface(
            (config.SIMULATION_AREA_WIDTH, config.SIMULATION_AREA_HEIGHT),
            pygame.SRCALPHA
        )
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_click(event)
            
            self.sim.update()
            self._render()
            
            pygame.display.flip()
            self.clock.tick(config.FPS)
        
        pygame.quit()
        sys.exit()
    
    def _handle_key(self, key: int):
        """Handle keyboard input."""
        if key == pygame.K_SPACE:
            self.sim.toggle_pause()
        elif key == pygame.K_r:
            self.sim.reset()
        elif key == pygame.K_s:
            path = self.sim.save_best_network()
            print(f"[INFO] Saved: {path}")
        elif key == pygame.K_l:
            if self.sim.load_best_network():
                print("[INFO] Loaded best network!")
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
            self.sim.increase_speed()
        elif key == pygame.K_MINUS:
            self.sim.decrease_speed()
        elif key == pygame.K_o:
            mode = self.sim.cycle_obstacle_mode()
            print(f"[INFO] Obstacle Mode: {mode}")
        elif key == pygame.K_n:
            self.sim.generate_new_random_layout()
        elif key == pygame.K_t:
            self.sim.set_random_target()
        elif key == pygame.K_h:
            self.show_help = not self.show_help
        elif key == pygame.K_g:
            self.sim.toggle_generalization()
        elif key == pygame.K_c:
            self.sim.toggle_curriculum()
        elif key == pygame.K_d:
            self.sim.toggle_debug()
        elif key == pygame.K_v:
            self.sim.toggle_sensors()
        elif key == pygame.K_UP:
            config.POPULATION_SIZE = min(500, config.POPULATION_SIZE + 30)
            self.sim.reset()
            print(f"[INFO] Population: {config.POPULATION_SIZE}")
        elif key == pygame.K_DOWN:
            config.POPULATION_SIZE = max(30, config.POPULATION_SIZE - 30)
            self.sim.reset()
            print(f"[INFO] Population: {config.POPULATION_SIZE}")
        elif key == pygame.K_t:
            self.sim.toggle_turbo()
        elif key == pygame.K_i:
            self.show_nn_inspector = not self.show_nn_inspector
        elif key == pygame.K_ESCAPE:
            self.sim.deselect_rocket()
    
    def _handle_mouse_click(self, event):
        """Handle mouse click."""
        if event.button == 1:
            x, y = event.pos
            if x < config.SIMULATION_AREA_WIDTH:
                if not self.sim.select_rocket_at(x, y):
                    self.sim.set_target(x, y)
        elif event.button == 3:
            self.sim.deselect_rocket()
    
    def _render(self):
        """Render everything."""
        self.screen.fill(config.COLOR_BG)
        
        self._draw_simulation_area()
        self._draw_panel()
        
        if self.show_help:
            self._draw_help()
        if self.sim.paused:
            self._draw_pause()
        if self.sim.selected_rocket:
            self._draw_rocket_inspector()
        if self.show_nn_inspector:
            self._draw_nn_io_inspector()
    
    def _draw_simulation_area(self):
        """Draw main simulation."""
        self._draw_grid()
        self._draw_trails()
        self._draw_obstacles()
        
        if self.sim.show_sensors:
            self._draw_sensor_rays()
        
        self._draw_rockets()
        self._draw_target()
        
        if self.sim.show_debug:
            self._draw_debug_overlay()
        
        if self.sim.selected_rocket:
            self._draw_selection_indicator()
        
        pygame.draw.rect(self.screen, config.COLOR_PANEL, 
                        (0, 0, config.SIMULATION_AREA_WIDTH, config.SIMULATION_AREA_HEIGHT), 2)
    
    def _draw_grid(self):
        """Draw subtle background grid."""
        grid_spacing = 50
        for x in range(0, config.SIMULATION_AREA_WIDTH, grid_spacing):
            pygame.draw.line(self.screen, config.COLOR_GRID, 
                           (x, 0), (x, config.SIMULATION_AREA_HEIGHT), 1)
        for y in range(0, config.SIMULATION_AREA_HEIGHT, grid_spacing):
            pygame.draw.line(self.screen, config.COLOR_GRID,
                           (0, y), (config.SIMULATION_AREA_WIDTH, y), 1)
    
    def _draw_trails(self):
        """Draw smooth trails."""
        self.trail_surface.fill((0, 0, 0, 0))
        
        champion = self.sim.get_champion()
        if champion and len(champion.trail) > 1:
            points = [(int(x), int(y)) for x, y in champion.trail]
            if len(points) > 2:
                for i in range(len(points) - 1):
                    alpha = int(200 * (i / len(points)))
                    color = (*config.COLOR_CHAMPION, alpha)
                    pygame.draw.line(self.trail_surface, color, points[i], points[i+1], 3)
        
        if self.sim.all_time_best_trail and len(self.sim.all_time_best_trail) > 1:
            points = [(int(x), int(y)) for x, y in self.sim.all_time_best_trail]
            if len(points) > 2:
                for i in range(len(points) - 1):
                    alpha = int(120 * (i / len(points)))
                    color = (*config.COLOR_ROCKET_SUCCESS, alpha)
                    pygame.draw.line(self.trail_surface, color, points[i], points[i+1], 2)
        
        if len(self.sim.best_trail) > 1:
            points = [(int(x), int(y)) for x, y in self.sim.best_trail]
            if len(points) > 2:
                for i in range(len(points) - 1):
                    alpha = int(100 * (i / len(points)))
                    color = (*config.COLOR_TRAIL, alpha)
                    pygame.draw.line(self.trail_surface, color, points[i], points[i+1], 2)
        
        self.screen.blit(self.trail_surface, (0, 0))
    
    def _draw_obstacles(self):
        """Draw obstacles."""
        for obs in self.sim.environment.obstacles:
            rect = pygame.Rect(obs["x"], obs["y"], obs["width"], obs["height"])
            pygame.draw.rect(self.screen, config.COLOR_OBSTACLE, rect)
            highlight_rect = pygame.Rect(obs["x"], obs["y"], obs["width"], 2)
            pygame.draw.rect(self.screen, (80, 80, 100), highlight_rect)
    
    def _draw_sensor_rays(self):
        """Draw sensor rays for debug rocket."""
        rocket = self.sim.get_debug_rocket()
        if rocket is None:
            return
        
        self.debug_surface.fill((0, 0, 0, 0))
        
        for i, angle_offset in enumerate(config.SENSOR_ANGLES):
            ray_angle = rocket.rotation + angle_offset
            sensor_val = rocket.sensor_readings[i] if i < len(rocket.sensor_readings) else 1.0
            
            ray_length = config.SENSOR_RANGE * sensor_val
            end_x = rocket.x + math.cos(ray_angle) * ray_length
            end_y = rocket.y + math.sin(ray_angle) * ray_length
            
            r = int(255 * (1 - sensor_val))
            g = int(255 * sensor_val)
            color = (r, g, 50, 80)
            
            pygame.draw.line(self.debug_surface, color, 
                           (int(rocket.x), int(rocket.y)),
                           (int(end_x), int(end_y)), 2)
        
        self.screen.blit(self.debug_surface, (0, 0))
    
    def _draw_rockets(self):
        """Draw all rockets."""
        self.rocket_surface.fill((0, 0, 0, 0))
        
        champion = self.sim.get_champion()
        best_alive = self._get_best_alive()
        debug_rocket = self.sim.get_debug_rocket()
        selected = self.sim.selected_rocket
        
        for rocket in self.sim.rockets:
            if rocket.crashed:
                color = config.COLOR_ROCKET_CRASHED
                alpha = 80
            elif not rocket.alive and not rocket.reached_target:
                color = config.COLOR_ROCKET_DEAD
                alpha = 40
            elif rocket.reached_target:
                color = config.COLOR_ROCKET_SUCCESS
                alpha = 180
            elif rocket == selected:
                color = config.COLOR_SELECTED
                alpha = 255
            elif rocket == debug_rocket and self.sim.show_debug:
                color = (255, 150, 50)
                alpha = 255
            elif rocket == champion:
                color = config.COLOR_CHAMPION
                alpha = 255
            elif rocket == best_alive:
                color = config.COLOR_ROCKET_BEST
                alpha = 220
            else:
                color = config.COLOR_ROCKET
                alpha = 100
            
            vertices = rocket.get_vertices()
            points = [(int(x), int(y)) for x, y in vertices]
            
            if alpha < 255:
                pygame.draw.polygon(self.rocket_surface, (*color, alpha), points)
            else:
                pygame.draw.polygon(self.rocket_surface, color, points)
        
        self.screen.blit(self.rocket_surface, (0, 0))
    
    def _get_best_alive(self):
        """Get the best alive rocket."""
        alive = [r for r in self.sim.rockets if r.alive]
        if not alive:
            return None
        return min(alive, key=lambda r: r.distance_to_target)
    
    def _draw_target(self):
        """Draw the target."""
        x = int(self.sim.environment.target_x)
        y = int(self.sim.environment.target_y)
        radius = config.TARGET_RADIUS
        
        pulse = math.sin(pygame.time.get_ticks() * 0.005) * 3
        pygame.draw.circle(self.screen, config.COLOR_TARGET, (x, y), int(radius + 8 + pulse), 2)
        pygame.draw.circle(self.screen, config.COLOR_TARGET, (x, y), radius, 2)
        pygame.draw.circle(self.screen, config.COLOR_TARGET, (x, y), 4)
    
    def _draw_selection_indicator(self):
        """Draw a ring around the selected rocket."""
        rocket = self.sim.selected_rocket
        if rocket is None or not rocket.alive:
            return
        
        t = pygame.time.get_ticks() * 0.005
        radius = 20 + math.sin(t) * 3
        color = config.COLOR_SELECTED
        pygame.draw.circle(self.screen, color, (int(rocket.x), int(rocket.y)), int(radius), 2)
    
    def _draw_debug_overlay(self):
        """Draw debug information overlay."""
        rocket = self.sim.get_debug_rocket()
        if rocket is None:
            return
        
        debug = rocket.get_debug_info()
        
        x = 10
        y = config.SIMULATION_AREA_HEIGHT - 120
        
        bg_rect = pygame.Rect(x - 5, y - 5, 280, 115)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)
        pygame.draw.rect(self.screen, config.COLOR_DEBUG, bg_rect, 1)
        
        title = self.font_debug.render("DEBUG - Closest Rocket", True, config.COLOR_DEBUG)
        self.screen.blit(title, (x, y))
        y += 18
        
        if rocket.stuck_counter > 20:
            status = "STUCK"
            status_color = config.COLOR_ROCKET_CRASHED
        elif rocket.obstacle_danger > 0.3:
            status = "DANGER"
            status_color = (255, 200, 50)
        elif rocket.recovery_bonus > 0:
            status = "RECOVERED"
            status_color = config.COLOR_ROCKET_SUCCESS
        else:
            status = "OK"
            status_color = config.COLOR_ROCKET
        
        status_text = self.font_debug.render(f"Status: {status}", True, status_color)
        self.screen.blit(status_text, (x, y))
        y += 16
        
        values = [
            (f"Progress: {debug['progress']:.3f}", config.COLOR_TEXT),
            (f"Danger: {debug['danger']:.3f}", config.COLOR_ROCKET_CRASHED if debug['danger'] > 0.3 else config.COLOR_TEXT),
            (f"Stuck: {debug['stuck_level']:.2f} ({debug['stuck_counter']:.0f})", config.COLOR_ROCKET_CRASHED if debug['stuck_level'] > 0.3 else config.COLOR_TEXT),
            (f"Total Reward: {debug['total_reward']:.1f}", config.COLOR_ROCKET_BEST),
        ]
        
        for text, color in values:
            surface = self.font_debug.render(text, True, color)
            self.screen.blit(surface, (x, y))
            y += 15
    
    def _draw_rocket_inspector(self):
        """Draw detailed inspector for the selected rocket."""
        rocket = self.sim.selected_rocket
        if rocket is None or not rocket.alive:
            self.sim.deselect_rocket()
            return
        
        x = 10
        y = 10
        w = 260
        h = 220
        
        bg_rect = pygame.Rect(x - 5, y - 5, w + 10, h + 10)
        pygame.draw.rect(self.screen, (20, 20, 40, 230), bg_rect)
        pygame.draw.rect(self.screen, config.COLOR_SELECTED, bg_rect, 1)
        
        title = self.font_debug.render("ROCKET INSPECTOR", True, config.COLOR_SELECTED)
        self.screen.blit(title, (x, y))
        y += 18
        
        debug = rocket.get_debug_info()
        
        status = "ALIVE"
        status_color = config.COLOR_ROCKET
        if rocket.reached_target:
            status = "REACHED TARGET"
            status_color = config.COLOR_ROCKET_SUCCESS
        elif rocket.crashed:
            status = "CRASHED"
            status_color = config.COLOR_ROCKET_CRASHED
        elif rocket.stuck_counter > 20:
            status = "STUCK"
            status_color = config.COLOR_ROCKET_CRASHED
        
        self.screen.blit(self.font_debug.render(f"Status: {status}", True, status_color), (x, y))
        y += 16
        
        values = [
            (f"Position: ({rocket.x:.0f}, {rocket.y:.0f})", config.COLOR_TEXT),
            (f"Speed: {rocket.get_speed():.2f}", config.COLOR_TEXT),
            (f"Distance: {rocket.distance_to_target:.0f}", config.COLOR_TEXT),
            (f"Path Efficiency: {rocket.get_path_efficiency():.1%}", config.COLOR_ROCKET_SUCCESS if rocket.get_path_efficiency() > 0.5 else config.COLOR_TEXT),
            (f"Frame: {rocket.frame_count}", config.COLOR_TEXT_DIM),
            (f"Total Reward: {rocket.total_reward:.1f}", config.COLOR_ROCKET_BEST),
            (f"Turn: {rocket.last_turn:+.2f}  Thrust: {rocket.last_thrust:.2f}", config.COLOR_TEXT),
        ]
        
        for text, color in values:
            surface = self.font_debug.render(text, True, color)
            self.screen.blit(surface, (x, y))
            y += 15
        
        y += 5
        self.screen.blit(self.font_micro.render("ESC or Right-click to deselect", True, config.COLOR_TEXT_DIM), (x, y))
    
    def _draw_nn_io_inspector(self):
        """Draw neural network input/output inspector panel."""
        rocket = self.sim.get_debug_rocket()
        if rocket is None:
            return
        
        x = config.SIMULATION_AREA_WIDTH - 270
        y = 10
        w = 260
        h = 300
        
        bg_rect = pygame.Rect(x - 5, y - 5, w + 10, h + 10)
        pygame.draw.rect(self.screen, (20, 20, 40, 230), bg_rect)
        pygame.draw.rect(self.screen, (100, 180, 255), bg_rect, 1)
        
        title = self.font_debug.render("NN INPUT/OUTPUT", True, (100, 180, 255))
        self.screen.blit(title, (x, y))
        y += 18
        
        net_inputs = rocket.get_network_inputs(
            self.sim.environment.target_x,
            self.sim.environment.target_y,
            self.sim.width,
            self.sim.height
        )
        
        self.screen.blit(self.font_micro.render("BASE INPUTS:", True, config.COLOR_TEXT_DIM), (x, y))
        y += 14
        
        for name, val in net_inputs.items():
            bar_width = int(abs(val) * 80)
            bar_color = config.COLOR_ROCKET_SUCCESS if val > 0 else config.COLOR_ROCKET_CRASHED
            
            self.screen.blit(self.font_micro.render(f"{name}:", True, config.COLOR_TEXT_DIM), (x, y))
            pygame.draw.rect(self.screen, (40, 40, 60), (x + 100, y + 1, 80, 8))
            if bar_width > 0:
                bx = x + 100 + (40 - bar_width // 2 if val > 0 else 40)
                pygame.draw.rect(self.screen, bar_color, (bx, y + 1, bar_width, 8))
            self.screen.blit(self.font_micro.render(f"{val:+.2f}", True, config.COLOR_TEXT), (x + 185, y))
            y += 12
        
        y += 3
        self.screen.blit(self.font_micro.render("SENSOR INPUTS:", True, config.COLOR_TEXT_DIM), (x, y))
        y += 14
        
        sensor_vals = rocket.get_sensor_inputs()
        for name, val in sensor_vals.items():
            bar_width = int(val * 80)
            bar_color = config.COLOR_ROCKET_SUCCESS if val > 0.5 else config.COLOR_ROCKET_CRASHED
            
            self.screen.blit(self.font_micro.render(f"{name}:", True, config.COLOR_TEXT_DIM), (x, y))
            pygame.draw.rect(self.screen, (40, 40, 60), (x + 80, y + 1, 80, 8))
            pygame.draw.rect(self.screen, bar_color, (x + 80, y + 1, bar_width, 8))
            self.screen.blit(self.font_micro.render(f"{val:.2f}", True, config.COLOR_TEXT), (x + 165, y))
            y += 12
        
        y += 3
        self.screen.blit(self.font_micro.render("NN OUTPUT:", True, config.COLOR_TEXT_DIM), (x, y))
        y += 14
        
        turn_val = rocket.last_turn
        thrust_val = rocket.last_thrust
        
        turn_bar = int(abs(turn_val) * 50)
        turn_color = config.COLOR_ROCKET_SUCCESS if turn_val > 0 else config.COLOR_ROCKET_CRASHED
        self.screen.blit(self.font_micro.render("Turn:", True, config.COLOR_TEXT_DIM), (x, y))
        pygame.draw.rect(self.screen, (40, 40, 60), (x + 50, y + 1, 100, 8))
        pygame.draw.rect(self.screen, turn_color, (x + 100 - turn_bar // 2, y + 1, turn_bar, 8))
        pygame.draw.line(self.screen, config.COLOR_TEXT_DIM, (x + 100, y), (x + 100, y + 10), 1)
        self.screen.blit(self.font_micro.render(f"{turn_val:+.2f}", True, config.COLOR_TEXT), (x + 155, y))
        y += 12
        
        thrust_bar = int(thrust_val * 100)
        thrust_color = config.COLOR_ROCKET_BEST
        self.screen.blit(self.font_micro.render("Thrust:", True, config.COLOR_TEXT_DIM), (x, y))
        pygame.draw.rect(self.screen, (40, 40, 60), (x + 50, y + 1, 100, 8))
        pygame.draw.rect(self.screen, thrust_color, (x + 50, y + 1, thrust_bar, 8))
        self.screen.blit(self.font_micro.render(f"{thrust_val:.2f}", True, config.COLOR_TEXT), (x + 155, y))
    
    def _draw_panel(self):
        """Draw side panel."""
        panel_rect = pygame.Rect(config.SIMULATION_AREA_WIDTH, 0, 
                                config.PANEL_WIDTH, config.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, config.COLOR_PANEL, panel_rect)
        
        stats = self.sim.get_stats()
        self._draw_stats(stats)
        
        nn = self.sim.get_best_network()
        if nn:
            self._draw_neural_network(nn)
        
        self._draw_learning_graph()
        
        if self.sim.show_debug:
            self._draw_reward_diagnostics()
        
        self._draw_training_phase(stats)
        
        self._draw_trend(stats)
        
        self._draw_plateau(stats)
    
    def _draw_stats(self, stats: dict):
        """Draw statistics."""
        x = config.SIMULATION_AREA_WIDTH + 15
        y = 15
        
        title = self.font_title.render("NEURAL ROCKET", True, config.COLOR_ROCKET)
        self.screen.blit(title, (x, y))
        y += 22
        subtitle = self.font_title.render("EVOLUTION", True, config.COLOR_ROCKET_BEST)
        self.screen.blit(subtitle, (x, y))
        y += 30
        
        modes = []
        if stats["generalization"]:
            modes.append(("GEN", config.COLOR_ROCKET_SUCCESS))
        if stats["curriculum"]:
            modes.append(("CURR", config.COLOR_ROCKET))
        if stats["show_debug"]:
            modes.append(("DBG", config.COLOR_DEBUG))
        if stats["show_sensors"]:
            modes.append(("SNS", config.COLOR_SENSOR_RAY[:3]))
        if stats["turbo_mode"]:
            modes.append(("TURBO", config.COLOR_ROCKET_CRASHED))
        if stats["selected_rocket"]:
            modes.append(("INSPECT", config.COLOR_SELECTED))
        
        for i, (mode_text, mode_color) in enumerate(modes):
            mode_surface = self.font_small.render(f"[{mode_text}]", True, mode_color)
            self.screen.blit(mode_surface, (x + i * 55, y))
        y += 20
        
        self._draw_stat_row(x, y, "Gen:", str(stats["generation"]))
        y += 18
        self._draw_stat_row(x, y, "Pop:", str(stats["population"]))
        y += 22
        
        self._draw_stat_row(x, y, "Mode:", stats["obstacle_mode"], config.COLOR_ROCKET)
        y += 22
        
        self._draw_stat_row(x, y, "Alive:", str(stats["alive"]), config.COLOR_ROCKET)
        y += 16
        self._draw_stat_row(x, y, "Reached:", str(stats["reached"]), config.COLOR_ROCKET_SUCCESS)
        y += 16
        self._draw_stat_row(x, y, "Crashed:", str(stats["crashed"]), config.COLOR_ROCKET_CRASHED)
        y += 16
        self._draw_stat_row(x, y, "Stuck:", str(stats["stuck"]), (255, 150, 50))
        y += 20
        
        self._draw_stat_row(x, y, "Success:", f"{stats['success_rate']:.1f}%", config.COLOR_ROCKET_SUCCESS)
        y += 22
        
        self._draw_stat_row(x, y, "Best:", f"{stats['best_fitness']:.0f}", config.COLOR_ROCKET_BEST)
        y += 16
        self._draw_stat_row(x, y, "Avg:", f"{stats['avg_fitness']:.0f}", config.COLOR_TEXT)
        y += 20
        
        if stats["best_completion"] is not None:
            self._draw_stat_row(x, y, "Best Cmp:", f"{stats['best_completion']:.0f}", config.COLOR_ROCKET_SUCCESS)
            y += 16
            self._draw_stat_row(x, y, "Avg Cmp:", f"{stats['avg_completion']:.0f}", config.COLOR_TEXT)
            y += 16
            if stats["fastest_ever"] is not None:
                self._draw_stat_row(x, y, "Fastest:", f"{stats['fastest_ever']}", config.COLOR_CHAMPION)
                y += 16
        else:
            self._draw_stat_row(x, y, "Best Cmp:", "--", config.COLOR_TEXT_DIM)
            y += 16
            self._draw_stat_row(x, y, "Avg Cmp:", "--", config.COLOR_TEXT_DIM)
            y += 16
            self._draw_stat_row(x, y, "Fastest:", "--", config.COLOR_TEXT_DIM)
            y += 20
        
        self._draw_stat_row(x, y, "Speed:", f"{stats['speed']}x", config.COLOR_TEXT)
        y += 16
        
        gen_len = stats['generation_length']
        progress = stats['frame'] / gen_len * 100
        progress_color = config.COLOR_TEXT_DIM
        if stats['generation_extended']:
            progress_color = config.COLOR_ROCKET_SUCCESS
        self._draw_stat_row(x, y, "Frame:", f"{stats['frame']}/{gen_len} ({progress:.0f}%)", progress_color)
        y += 16
        
        self._draw_stat_row(x, y, "Total Gen:", str(stats["total_generations"]), config.COLOR_TEXT_DIM)
        y += 16
        
        self._draw_stat_row(x, y, "All-time Best:", f"{stats['all_time_best_sr']:.1f}%", config.COLOR_CHAMPION)
    
    def _draw_stat_row(self, x: int, y: int, label: str, value: str, 
                       value_color=None):
        """Draw a stat row."""
        if value_color is None:
            value_color = config.COLOR_TEXT
        
        label_surface = self.font_label.render(label, True, config.COLOR_TEXT_DIM)
        value_surface = self.font_value.render(value, True, value_color)
        
        self.screen.blit(label_surface, (x, y))
        self.screen.blit(value_surface, (x + 90, y))
    
    def _draw_reward_diagnostics(self):
        """Draw reward component breakdown."""
        diag = self.sim.get_reward_diagnostics()
        if not diag:
            return
        
        x = config.SIMULATION_AREA_WIDTH + 15
        y = 420
        
        bg_rect = pygame.Rect(x - 5, y - 5, config.PANEL_WIDTH - 10, 115)
        pygame.draw.rect(self.screen, (20, 20, 35), bg_rect)
        pygame.draw.rect(self.screen, config.COLOR_DEBUG, bg_rect, 1)
        
        title = self.font_small.render("REWARD DIAGNOSTICS", True, config.COLOR_DEBUG)
        self.screen.blit(title, (x, y))
        y += 18
        
        components = [
            ("Progress:", diag["Progress"], config.COLOR_ROCKET_SUCCESS if diag["Progress"] > 0 else config.COLOR_ROCKET_CRASHED),
            ("Danger:", diag["Danger"], config.COLOR_ROCKET_CRASHED if diag["Danger"] < -1 else config.COLOR_TEXT),
            ("Stuck:", diag["Stuck"], config.COLOR_ROCKET_CRASHED if diag["Stuck"] < -1 else config.COLOR_TEXT),
            ("Recovery:", diag["Recovery"], config.COLOR_ROCKET_SUCCESS if diag["Recovery"] > 0 else config.COLOR_TEXT),
            ("Total:", diag["Total"], config.COLOR_ROCKET_BEST),
        ]
        
        for label, value, color in components:
            label_surface = self.font_debug.render(label, True, config.COLOR_TEXT_DIM)
            value_surface = self.font_debug.render(f"{value:+.1f}", True, color)
            self.screen.blit(label_surface, (x, y))
            self.screen.blit(value_surface, (x + 80, y))
            y += 16
    
    def _draw_training_phase(self, stats: dict):
        """Draw training phase indicator."""
        x = config.SIMULATION_AREA_WIDTH + 15
        y = 530
        
        phase = stats["training_phase"]
        phase_colors = {
            "early": config.COLOR_ROCKET_SUCCESS,
            "mid": config.COLOR_ROCKET_BEST,
            "late": config.COLOR_TEXT,
        }
        phase_names = {
            "early": "EARLY (Exploration)",
            "mid": "MID (Stabilization)", 
            "late": "LATE (Refinement)",
        }
        
        color = phase_colors.get(phase, config.COLOR_TEXT)
        name = phase_names.get(phase, phase)
        
        self.screen.blit(self.font_small.render(f"Phase: {name}", True, color), (x, y))
    
    def _draw_trend(self, stats: dict):
        """Draw trend indicator."""
        x = config.SIMULATION_AREA_WIDTH + 15
        y = 548
        
        sr_trend = stats["success_rate_trend"]
        fit_trend = stats["fitness_trend"]
        
        if sr_trend > 1.0:
            sr_arrow = "UP"
            sr_color = config.COLOR_ROCKET_SUCCESS
        elif sr_trend < -1.0:
            sr_arrow = "DOWN"
            sr_color = config.COLOR_ROCKET_CRASHED
        else:
            sr_arrow = "FLAT"
            sr_color = config.COLOR_TEXT_DIM
        
        self.screen.blit(self.font_small.render(f"Trend: {sr_arrow} ({sr_trend:+.1f}%)", True, sr_color), (x, y))
    
    def _draw_plateau(self, stats: dict):
        """Draw plateau detection indicator."""
        x = config.SIMULATION_AREA_WIDTH + 15
        y = 566
        
        if stats["plateau_detected"]:
            color = config.COLOR_ROCKET_CRASHED
            text = f"PLATEAU ({stats['plateau_duration']} gens)"
        else:
            color = config.COLOR_TEXT_DIM
            text = "No plateau"
        
        self.screen.blit(self.font_small.render(text, True, color), (x, y))
    
    def _draw_neural_network(self, nn: NeuralNetwork):
        """Draw neural network."""
        x = config.NN_PANEL_X
        y = config.NN_PANEL_Y
        w = config.NN_PANEL_WIDTH
        h = config.NN_PANEL_HEIGHT
        
        pygame.draw.rect(self.screen, (20, 20, 35), (x - 5, y - 5, w + 10, h + 10))
        pygame.draw.rect(self.screen, config.COLOR_TEXT_DIM, (x - 5, y - 5, w + 10, h + 10), 1)
        
        title = self.font_small.render("NETWORK", True, config.COLOR_TEXT)
        self.screen.blit(title, (x + 5, y))
        
        layers = nn.layer_sizes
        num_layers = len(layers)
        
        layer_x = []
        for i in range(num_layers):
            lx = x + 25 + (w - 50) * i / (num_layers - 1)
            layer_x.append(lx)
        
        node_positions: List[List[Tuple[int, int]]] = []
        for i, size in enumerate(layers):
            nodes = []
            node_height = min(h - 40, size * 14)
            start_y = y + 25 + (h - 40 - node_height) / 2
            
            for j in range(size):
                ny = int(start_y + (j + 0.5) * node_height / size)
                nodes.append((int(layer_x[i]), ny))
            node_positions.append(nodes)
        
        for i in range(num_layers - 1):
            for j, src in enumerate(node_positions[i]):
                for k, dst in enumerate(node_positions[i + 1]):
                    weight = nn.weights[i][j, k]
                    if abs(weight) > 0.4:
                        color = config.COLOR_NET_WEIGHT_POS if weight > 0 else config.COLOR_NET_WEIGHT_NEG
                        thickness = max(1, int(abs(weight) * 1.5))
                        pygame.draw.line(self.screen, color, src, dst, thickness)
        
        for i, nodes in enumerate(node_positions):
            for j, (nx, ny) in enumerate(nodes):
                if i == 0:
                    color = (80, 160, 255)
                elif i == num_layers - 1:
                    color = (255, 180, 50)
                else:
                    color = config.COLOR_NET_NODE
                
                pygame.draw.circle(self.screen, color, (nx, ny), config.NN_NODE_RADIUS)
                pygame.draw.circle(self.screen, (255, 255, 255), (nx, ny), config.NN_NODE_RADIUS, 1)
        
        labels = ["In", "H1", "H2", "Out"]
        for i, lx in enumerate(layer_x):
            if i < len(labels):
                label = self.font_small.render(labels[i], True, config.COLOR_TEXT_DIM)
                label_x = lx - label.get_width() // 2
                self.screen.blit(label, (label_x, y + h - 12))
    
    def _draw_learning_graph(self):
        """Draw learning graph."""
        x = config.GRAPH_X
        y = config.GRAPH_Y
        w = config.GRAPH_WIDTH
        h = config.GRAPH_HEIGHT
        
        pygame.draw.rect(self.screen, (20, 20, 35), (x - 5, y - 5, w + 10, h + 10))
        
        title = self.font_small.render("LEARNING", True, config.COLOR_TEXT)
        self.screen.blit(title, (x + 5, y))
        
        history = self.sim.fitness_history
        if len(history) < 2:
            return
        
        pygame.draw.line(self.screen, config.COLOR_TEXT_DIM, 
                        (x + 20, y + 20), (x + 20, y + h - 10), 1)
        pygame.draw.line(self.screen, config.COLOR_TEXT_DIM,
                        (x + 20, y + h - 10), (x + w - 10, y + h - 10), 1)
        
        best_values = [s["best_fitness"] for s in history]
        avg_values = [s["avg_fitness"] for s in history]
        
        max_val = max(max(best_values), max(avg_values), 1)
        
        def scale_x(i: int) -> int:
            return x + 20 + int((w - 30) * i / max(len(history) - 1, 1))
        
        def scale_y(val: float) -> int:
            return y + h - 10 - int((h - 30) * val / max_val)
        
        if len(best_values) > 1:
            points_best = [(scale_x(i), scale_y(v)) for i, v in enumerate(best_values)]
            points_avg = [(scale_x(i), scale_y(v)) for i, v in enumerate(avg_values)]
            
            pygame.draw.lines(self.screen, config.COLOR_ROCKET_BEST, False, points_best, 2)
            pygame.draw.lines(self.screen, config.COLOR_ROCKET, False, points_avg, 2)
        
        legend_y = y + 5
        pygame.draw.circle(self.screen, config.COLOR_ROCKET_BEST, (x + w - 80, legend_y + 5), 4)
        legend = self.font_small.render("Best", True, config.COLOR_ROCKET_BEST)
        self.screen.blit(legend, (x + w - 70, legend_y))
        
        pygame.draw.circle(self.screen, config.COLOR_ROCKET, (x + w - 35, legend_y + 5), 4)
        legend = self.font_small.render("Avg", True, config.COLOR_ROCKET)
        self.screen.blit(legend, (x + w - 25, legend_y))
    
    def _draw_help(self):
        """Draw help overlay."""
        overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        x = config.SCREEN_WIDTH // 2 - 180
        y = 60
        
        lines = [
            ("CONTROLS", config.COLOR_ROCKET, self.font_title),
            ("", None, self.font_label),
            ("LEFT CLICK  Set target / Select rocket", config.COLOR_TEXT, self.font_label),
            ("RIGHT CLICK Deselect rocket", config.COLOR_TEXT, self.font_label),
            ("ESC         Deselect rocket", config.COLOR_TEXT, self.font_label),
            ("SPACE       Pause/Resume", config.COLOR_TEXT, self.font_label),
            ("R           Reset training", config.COLOR_TEXT, self.font_label),
            ("S           Save network", config.COLOR_TEXT, self.font_label),
            ("L           Load network", config.COLOR_TEXT, self.font_label),
            ("+/-         Speed up/down", config.COLOR_TEXT, self.font_label),
            ("UP/DOWN     Population +/-30", config.COLOR_TEXT, self.font_label),
            ("O           Cycle obstacles", config.COLOR_TEXT, self.font_label),
            ("N           New random layout", config.COLOR_TEXT, self.font_label),
            ("T           Random target", config.COLOR_TEXT, self.font_label),
            ("G           Generalization mode", config.COLOR_TEXT, self.font_label),
            ("C           Curriculum mode", config.COLOR_TEXT, self.font_label),
            ("T           Toggle turbo mode", config.COLOR_TEXT, self.font_label),
            ("I           Toggle NN I/O inspector", config.COLOR_TEXT, self.font_label),
            ("D           Toggle debug info", config.COLOR_DEBUG, self.font_label),
            ("V           Toggle sensor rays", config.COLOR_DEBUG, self.font_label),
            ("H           Toggle this help", config.COLOR_TEXT, self.font_label),
            ("", None, self.font_label),
            ("FEATURES", config.COLOR_ROCKET_BEST, self.font_title),
            ("", None, self.font_label),
            ("Click a rocket to inspect its neural net I/O", config.COLOR_TEXT, self.font_small),
            ("Adaptive generation length based on training phase", config.COLOR_TEXT, self.font_small),
            ("Plateau detection and trend tracking", config.COLOR_TEXT, self.font_small),
            ("All-time best trail preserved across resets", config.COLOR_TEXT, self.font_small),
            ("", None, self.font_label),
            ("Press H to close", config.COLOR_TEXT_DIM, self.font_small),
        ]
        
        for text, color, font in lines:
            if text:
                surface = font.render(text, True, color)
                self.screen.blit(surface, (x, y))
            y += 20
    
    def _draw_pause(self):
        """Draw pause overlay."""
        overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        text = self.font_title.render("PAUSED", True, config.COLOR_ROCKET_BEST)
        rect = text.get_rect(center=(config.SIMULATION_AREA_WIDTH // 2, 
                                     config.SIMULATION_AREA_HEIGHT // 2))
        self.screen.blit(text, rect)


def main():
    """Entry point."""
    app = App()
    app.run()


if __name__ == "__main__":
    main()
