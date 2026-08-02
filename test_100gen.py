import config
from simulation import Simulation

config.POPULATION_SIZE = 100

sim = Simulation(config.SIMULATION_AREA_WIDTH, config.SIMULATION_AREA_HEIGHT)

print(f"{'Gen':>4} | {'Best':>8} | {'Avg':>8} | {'Reached':>7} | {'Rate':>5}")
print("-" * 48)

for gen in range(50):
    sim.frame = 0
    sim.generation_extended = False
    for rocket in sim.rockets:
        rocket.alive = True

    for frame in range(config.BASE_GENERATION_LENGTH):
        for rocket in sim.rockets:
            if rocket.alive:
                sr = sim.environment.get_all_sensor_readings(rocket.x, rocket.y, rocket.rotation)
                rocket.update(
                    sim.environment.target_x, sim.environment.target_y,
                    sim.width, sim.height,
                    sim.environment.obstacles if sim.environment.obstacles else None,
                    sr
                )
        if sum(1 for r in sim.rockets if r.alive) == 0:
            break

    sim._next_generation()
    s = sim.last_generation_summary
    if s:
        g = s["generation"]
        bf = s["best_fitness"]
        af = s["avg_fitness"]
        nr = s["num_reached"]
        sr_val = s["success_rate"]
        print(f"{g:4d} | {bf:8.1f} | {af:8.1f} | {nr:7d} | {sr_val:5.1f}%")
