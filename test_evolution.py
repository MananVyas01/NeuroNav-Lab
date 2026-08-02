"""
Quick evolutionary test: run 30 generations headless, measure success rate.
"""
import config
from simulation import Simulation


def test_evolution():
    sim = Simulation(config.SIMULATION_AREA_WIDTH, config.SIMULATION_AREA_HEIGHT)
    
    print(f"{'Gen':>4} | {'Best':>8} | {'Avg':>8} | {'Reached':>7} | {'Rate':>5} | {'BestEver':>8}")
    print("-" * 55)
    
    for gen in range(30):
        # Run one full generation
        sim.frame = 0
        sim.generation_extended = False
        
        for rocket in sim.rockets:
            rocket.alive = True
        
        for frame in range(config.BASE_GENERATION_LENGTH):
            for rocket in sim.rockets:
                if rocket.alive:
                    sensor_readings = sim.environment.get_all_sensor_readings(
                        rocket.x, rocket.y, rocket.rotation
                    )
                    rocket.update(
                        sim.environment.target_x,
                        sim.environment.target_y,
                        sim.width,
                        sim.height,
                        sim.environment.obstacles if sim.environment.obstacles else None,
                        sensor_readings
                    )
            
            alive = sum(1 for r in sim.rockets if r.alive)
            if alive == 0:
                break
        
        sim._next_generation()
        
        stats = sim.last_generation_summary
        if stats:
            print(f"{stats['generation']:4d} | {stats['best_fitness']:8.1f} | "
                  f"{stats['avg_fitness']:8.1f} | {stats['num_reached']:7d} | "
                  f"{stats['success_rate']:5.1f}% | {sim.evolution.best_fitness_ever:8.1f}")


if __name__ == "__main__":
    test_evolution()
