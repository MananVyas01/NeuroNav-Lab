"""
Step 3 & 8: Audit reward magnitudes. Run 300 random rockets for 500 frames,
measure every reward component.
"""
import math
import numpy as np
from collections import defaultdict
from rocket import Rocket
from neural_network import NeuralNetwork
import config


def audit_rewards():
    screen_w = config.SIMULATION_AREA_WIDTH
    screen_h = config.SIMULATION_AREA_HEIGHT
    target_x = screen_w * 0.75
    target_y = screen_h * 0.5
    
    N_ROCKETS = 300
    N_FRAMES = 500
    
    # Accumulators for each reward component
    accum = defaultdict(list)
    reached = 0
    spinning = 0
    
    for i in range(N_ROCKETS):
        nn = NeuralNetwork([config.INPUT_NEURONS, config.HIDDEN1_NEURONS,
                           config.HIDDEN2_NEURONS, config.OUTPUT_NEURONS])
        rocket = Rocket(nn, 100.0, 350.0)
        
        rocket_rewards = defaultdict(float)
        
        for frame in range(N_FRAMES):
            if not rocket.alive:
                break
            
            old_reward = rocket.total_reward
            rocket.update(target_x, target_y, screen_w, screen_h, None, [1.0]*5)
            delta = rocket.total_reward - old_reward
            
            # Classify the delta into components
            rocket_rewards['progress'] += rocket.progress_reward
            rocket_rewards['danger'] += rocket.danger_penalty
            rocket_rewards['stuck'] += rocket.stuck_penalty
            rocket_rewards['spin'] += rocket.spin_penalty
        
        # Boundary penalty is accumulated in total_reward but not tracked separately
        # We'll compute it from total_reward minus tracked components
        tracked = (rocket_rewards['progress'] + rocket_rewards['danger'] + 
                   rocket_rewards['stuck'] + rocket_rewards['spin'])
        rocket_rewards['boundary+other'] = rocket.total_reward - tracked
        
        for key, val in rocket_rewards.items():
            accum[key].append(val)
        
        if rocket.reached_target:
            reached += 1
        
        total_rot = getattr(rocket, 'total_rotation', 0)
        displacement = math.sqrt((rocket.x - 100)**2 + (rocket.y - 350)**2)
        if displacement < 30 and total_rot > math.pi:
            spinning += 1
    
    print(f"\n{'='*65}")
    print(f"  REWARD AUDIT: {N_ROCKETS} rockets × {N_FRAMES} frames")
    print(f"{'='*65}")
    print(f"  Reached target:    {reached}/{N_ROCKETS} ({reached/N_ROCKETS*100:.1f}%)")
    print(f"  Spinning:          {spinning}/{N_ROCKETS} ({spinning/N_ROCKETS*100:.1f}%)")
    
    print(f"\n  {'Component':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'|Mean|':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    total_abs_mean = 0
    component_abs = {}
    for key in ['progress', 'danger', 'stuck', 'spin', 'boundary+other']:
        vals = np.array(accum[key])
        mean = np.mean(vals)
        median = np.median(vals)
        std = np.std(vals)
        min_v = np.min(vals)
        max_v = np.max(vals)
        abs_mean = np.mean(np.abs(vals))
        component_abs[key] = abs_mean
        total_abs_mean += abs_mean
        print(f"  {key:<20} {mean:>+8.2f} {median:>+8.2f} {std:>8.2f} {min_v:>+8.2f} {max_v:>+8.2f} {abs_mean:>8.2f}")
    
    # Also compute the distance reward (tracked inside total_reward differently)
    # Let's compute it separately
    dist_rewards = []
    fwd_rewards = []
    for i in range(N_ROCKETS):
        nn = NeuralNetwork([config.INPUT_NEURONS, config.HIDDEN1_NEURONS,
                           config.HIDDEN2_NEURONS, config.OUTPUT_NEURONS])
        rocket = Rocket(nn, 100.0, 350.0)
        for frame in range(N_FRAMES):
            if not rocket.alive:
                break
            rocket.update(target_x, target_y, screen_w, screen_h, None, [1.0]*5)
        # Approximate from total reward
    # These are embedded in boundary+other, we'll skip for now
    
    print(f"\n  Reward contribution percentages (by |mean|):")
    for key in ['progress', 'danger', 'stuck', 'spin', 'boundary+other']:
        if total_abs_mean > 0:
            pct = component_abs[key] / total_abs_mean * 100
        else:
            pct = 0
        bar = '#' * int(pct / 2)
        print(f"  {key:<20} {pct:>5.1f}%  {bar}")
    
    # Fitness distribution
    fitnesses = []
    for i in range(N_ROCKETS):
        nn = NeuralNetwork([config.INPUT_NEURONS, config.HIDDEN1_NEURONS,
                           config.HIDDEN2_NEURONS, config.OUTPUT_NEURONS])
        rocket = Rocket(nn, 100.0, 350.0)
        for frame in range(N_FRAMES):
            if not rocket.alive:
                break
            rocket.update(target_x, target_y, screen_w, screen_h, None, [1.0]*5)
        rocket.calculate_fitness(N_FRAMES)
        fitnesses.append(rocket.fitness)
    
    fitnesses = np.array(fitnesses)
    print(f"\n  Fitness distribution:")
    print(f"    Mean:    {np.mean(fitnesses):>+10.2f}")
    print(f"    Median:  {np.median(fitnesses):>+10.2f}")
    print(f"    Std:     {np.std(fitnesses):>10.2f}")
    print(f"    Min:     {np.min(fitnesses):>+10.2f}")
    print(f"    Max:     {np.max(fitnesses):>+10.2f}")
    print(f"    Negative: {np.sum(fitnesses < 0)}/{N_ROCKETS} ({np.sum(fitnesses < 0)/N_ROCKETS*100:.0f}%)")


if __name__ == "__main__":
    audit_rewards()
