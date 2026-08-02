"""
Headless diagnostic: trace the full control loop for representative rockets.
Run without pygame. Identifies which spinning case is occurring.
"""
import math
import numpy as np
from rocket import Rocket
from neural_network import NeuralNetwork
from environment import Environment
import config


def trace_rocket(rocket, target_x, target_y, screen_w, screen_h, obstacles, label, n_frames=200):
    """Trace one rocket's full control loop for n_frames, print every 10th frame."""
    print(f"\n{'='*70}")
    print(f"  ROCKET TRACE: {label}")
    print(f"  Start: ({rocket.x:.1f}, {rocket.y:.1f})  Target: ({target_x:.1f}, {target_y:.1f})")
    print(f"  Initial distance: {math.sqrt((target_x-rocket.x)**2 + (target_y-rocket.y)**2):.1f}")
    print(f"{'='*70}")
    
    header = (
        f"{'frame':>5} | {'dist':>6} | {'rel_ang':>7} | {'vx':>5} {'vy':>5} | {'speed':>5} | "
        f"{'heading':>7} | {'raw_turn':>8} {'raw_thrust':>10} | "
        f"{'app_turn':>8} {'app_thr':>7} | {'prog_rw':>7} {'dist_rw':>7} {'fwd_rw':>7} "
        f"{'spin_pn':>7} {'total_rw':>8}"
    )
    print(header)
    print("-" * len(header))
    
    prev_angle_error = None
    total_prog = 0.0
    total_dist = 0.0
    total_fwd = 0.0
    total_spin = 0.0
    turn_values = []
    thrust_values = []
    
    for frame in range(n_frames):
        if not rocket.alive:
            break
        
        # Get the raw inputs
        inputs = rocket.get_inputs(target_x, target_y, screen_w, screen_h)
        
        # Get raw network outputs
        raw_outputs = rocket.nn.forward(inputs)
        raw_turn = float(raw_outputs[0, 0])
        raw_thrust = float(raw_outputs[0, 1])
        
        # What think() actually returns
        turn, thrust = rocket.think(target_x, target_y, screen_w, screen_h)
        
        # Distance
        dx = target_x - rocket.x
        dy = target_y - rocket.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Relative angle
        angle_to_target = math.atan2(dy, dx)
        angle_diff = angle_to_target - rocket.rotation
        while angle_diff > math.pi: angle_diff -= 2*math.pi
        while angle_diff < -math.pi: angle_diff += 2*math.pi
        
        # Speed
        speed = math.sqrt(rocket.vx**2 + rocket.vy**2)
        
        # Reward components (before update)
        old_total = rocket.total_reward
        
        # Step the rocket
        sensor_readings = [1.0]*5  # no obstacles mode
        rocket.update(target_x, target_y, screen_w, screen_h, obstacles, sensor_readings)
        
        # Reward deltas
        prog_delta = rocket.progress_reward
        # approximate distance reward
        new_dist = math.sqrt((target_x-rocket.x)**2 + (target_y-rocket.y)**2)
        
        turn_values.append(raw_turn)
        thrust_values.append(thrust)
        
        if frame % 10 == 0 or frame < 5:
            print(
                f"{frame:5d} | {dist:6.1f} | {math.degrees(angle_diff):+7.1f} | "
                f"{rocket.vx:+5.2f} {rocket.vy:+5.2f} | {speed:5.2f} | "
                f"{math.degrees(rocket.rotation):+7.1f} | "
                f"{raw_turn:+8.4f} {raw_thrust:+10.4f} | "
                f"{turn:+8.4f} {thrust:7.4f} | "
                f"{rocket.progress_reward:+7.3f} "
            )
        
        if prev_angle_error is not None:
            improvement = abs(prev_angle_error) - abs(angle_diff)
        
        prev_angle_error = angle_diff
    
    # Summary stats
    if turn_values:
        mean_turn = np.mean(turn_values)
        std_turn = np.std(turn_values)
        mean_abs_turn = np.mean(np.abs(turn_values))
        turn_sat_near_pos = sum(1 for t in turn_values if t > 0.9) / len(turn_values) * 100
        turn_sat_near_neg = sum(1 for t in turn_values if t < -0.9) / len(turn_values) * 100
    else:
        mean_turn = std_turn = mean_abs_turn = turn_sat_near_pos = turn_sat_near_neg = 0
    
    if thrust_values:
        mean_thrust = np.mean(thrust_values)
        thrust_near_zero = sum(1 for t in thrust_values if t < 0.2) / len(thrust_values) * 100
    else:
        mean_thrust = thrust_near_zero = 0
    
    final_dist = math.sqrt((target_x-rocket.x)**2 + (target_y-rocket.y)**2) if rocket.alive else 0
    initial_dist = math.sqrt((target_x-rocket.start_x)**2 + (target_y-rocket.start_y)**2)
    
    # Compute total rotation
    total_rot = getattr(rocket, 'total_rotation', 0)
    
    # Displacement
    displacement = math.sqrt((rocket.x - rocket.start_x)**2 + (rocket.y - rocket.start_y)**2)
    
    print(f"\n--- SUMMARY ---")
    print(f"  Frames alive:         {rocket.frame_count}")
    print(f"  Final distance:       {final_dist:.1f} (started at {initial_dist:.1f})")
    print(f"  Displacement:         {displacement:.1f}")
    print(f"  Total rotation:       {math.degrees(total_rot):.1f} deg")
    print(f"  Total dist traveled:  {rocket.total_distance_traveled:.1f}")
    print(f"  Reached target:       {rocket.reached_target}")
    print(f"  Mean turn output:     {mean_turn:+.4f} (std={std_turn:.4f}, |mean|={mean_abs_turn:.4f})")
    print(f"  Turn saturation:      +1={turn_sat_near_pos:.0f}%  -1={turn_sat_near_neg:.0f}%")
    print(f"  Mean thrust:          {mean_thrust:.4f}")
    print(f"  Thrust near zero:     {thrust_near_zero:.0f}%")
    print(f"  Total reward:         {rocket.total_reward:.2f}")
    print(f"  Fitness:              {rocket.fitness:.2f}")
    
    # Diagnose case
    print(f"\n--- DIAGNOSIS ---")
    if displacement < 20 and total_rot > math.pi:
        print("  >>> CASE A/B: Spinning in place (high rotation, low displacement)")
    elif mean_thrust < 0.2:
        print("  >>> CASE C: Insufficient thrust")
    elif abs(mean_turn) > 0.7:
        print("  >>> CASE A: Saturated turn output")
    elif displacement > 100 and final_dist < initial_dist * 0.5:
        print("  >>> MAKING PROGRESS toward target")
    elif displacement < 50:
        print("  >>> CASE D/E: Not moving meaningfully")
    else:
        print("  >>> MIXED: Some movement but not reaching target")
    
    # Input sensitivity test
    print(f"\n--- INPUT SENSITIVITY TEST ---")
    test_network_sensitivity(rocket.nn, target_x, target_y, rocket, screen_w, screen_h)


def test_network_sensitivity(nn, target_x, target_y, rocket, screen_w, screen_h):
    """Test how sensitive network outputs are to changes in angle_to_target."""
    base_inputs = rocket.get_inputs(target_x, target_y, screen_w, screen_h)
    base_out = nn.forward(base_inputs)
    print(f"  Base input[0] (angle_to_target): {base_inputs[0]:+.4f}")
    print(f"  Base output:  turn={float(base_out[0,0]):+.4f}  thrust={float(base_out[0,1]):+.4f}")
    
    # Vary angle_to_target while keeping everything else fixed
    for test_angle in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
        test_inputs = base_inputs.copy()
        test_inputs[0] = test_angle
        out = nn.forward(test_inputs)
        t = float(out[0,0])
        th = float(out[0,1])
        print(f"  angle={test_angle:+.2f} -> turn={t:+.4f}  thrust={th:+.4f}")
    
    # Vary distance while keeping everything else fixed
    print(f"\n  --- Distance sensitivity ---")
    for test_dist in [0.1, 0.3, 0.5, 0.7, 0.9]:
        test_inputs = base_inputs.copy()
        test_inputs[1] = test_dist
        out = nn.forward(test_inputs)
        t = float(out[0,0])
        th = float(out[0,1])
        print(f"  dist={test_dist:.2f} -> turn={t:+.4f}  thrust={th:+.4f}")


def main():
    print("="*70)
    print("  NEUROEVOLUTION SPINNING DIAGNOSTIC")
    print("="*70)
    print(f"  Network: {config.INPUT_NEURONS} -> {config.HIDDEN1_NEURONS} -> {config.HIDDEN2_NEURONS} -> {config.OUTPUT_NEURONS}")
    print(f"  Total weights: {sum(np.prod(s) for s in [
        (config.INPUT_NEURONS, config.HIDDEN1_NEURONS),
        (1, config.HIDDEN1_NEURONS),
        (config.HIDDEN1_NEURONS, config.HIDDEN2_NEURONS),
        (1, config.HIDDEN2_NEURONS),
        (config.HIDDEN2_NEURONS, config.OUTPUT_NEURONS),
        (1, config.OUTPUT_NEURONS)
    ])}")
    print(f"  Population: {config.POPULATION_SIZE}")
    print(f"  ROTATION_SPEED: {config.ROTATION_SPEED}")
    print(f"  THRUST_POWER: {config.THRUST_POWER}")
    print(f"  FRICTION: {config.FRICTION}")
    
    screen_w = config.SIMULATION_AREA_WIDTH
    screen_h = config.SIMULATION_AREA_HEIGHT
    target_x = screen_w * 0.75
    target_y = screen_h * 0.5
    
    print(f"\n  Screen: {screen_w}x{screen_h}")
    print(f"  Target: ({target_x:.0f}, {target_y:.0f})")
    
    # --- Test 1: 10 random rockets, see how many spin ---
    print(f"\n{'='*70}")
    print(f"  POPULATION SNAPSHOT: 10 random rockets, 300 frames")
    print(f"{'='*70}")
    
    spinning_count = 0
    moving_count = 0
    reached_count = 0
    
    for i in range(10):
        nn = NeuralNetwork([config.INPUT_NEURONS, config.HIDDEN1_NEURONS, 
                           config.HIDDEN2_NEURONS, config.OUTPUT_NEURONS])
        rocket = Rocket(nn, 100.0, 350.0)
        
        for frame in range(300):
            if not rocket.alive:
                break
            rocket.update(target_x, target_y, screen_w, screen_h, None, [1.0]*5)
        
        dx = target_x - rocket.x
        dy = target_y - rocket.y
        final_dist = math.sqrt(dx*dx + dy*dy)
        displacement = math.sqrt((rocket.x-100)**2 + (rocket.y-350)**2)
        total_rot = getattr(rocket, 'total_rotation', 0)
        
        status = ""
        if rocket.reached_target:
            reached_count += 1
            status = "REACHED"
        elif displacement < 30 and total_rot > math.pi:
            spinning_count += 1
            status = "SPINNING"
        elif displacement > 50:
            moving_count += 1
            status = "MOVING"
        else:
            status = "IDLE"
        
        print(f"  Rocket {i}: dist={final_dist:.0f} disp={displacement:.0f} "
              f"rot={math.degrees(total_rot):.0f}deg "
              f"speed={rocket.get_speed():.2f} -> {status}")
    
    print(f"\n  Summary: {reached_count} reached, {moving_count} moving, "
          f"{spinning_count} spinning, {10-reached_count-moving_count-spinning_count} idle")
    
    # --- Test 2: Detailed trace of one spinning rocket ---
    print(f"\n{'='*70}")
    print(f"  DETAILED TRACE")
    print(f"{'='*70}")
    
    # Find a spinning rocket
    for attempt in range(20):
        nn = NeuralNetwork([config.INPUT_NEURONS, config.HIDDEN1_NEURONS,
                           config.HIDDEN2_NEURONS, config.OUTPUT_NEURONS])
        rocket = Rocket(nn, 100.0, 350.0)
        
        for frame in range(300):
            if not rocket.alive:
                break
            rocket.update(target_x, target_y, screen_w, screen_h, None, [1.0]*5)
        
        total_rot = getattr(rocket, 'total_rotation', 0)
        displacement = math.sqrt((rocket.x-100)**2 + (rocket.y-350)**2)
        
        if displacement < 30 and total_rot > math.pi:
            trace_rocket(rocket, target_x, target_y, screen_w, screen_h, None, 
                        f"Spinning rocket #{attempt}")
            break
    else:
        # If no spinning rocket found, trace the worst one
        nn = NeuralNetwork([config.INPUT_NEURONS, config.HIDDEN1_NEURONS,
                           config.HIDDEN2_NEURONS, config.OUTPUT_NEURONS])
        rocket = Rocket(nn, 100.0, 350.0)
        for frame in range(300):
            if not rocket.alive:
                break
            rocket.update(target_x, target_y, screen_w, screen_h, None, [1.0]*5)
        trace_rocket(rocket, target_x, target_y, screen_w, screen_h, None,
                    "Worst displacement rocket")
    
    # --- Test 3: One rocket that moves well ---
    for attempt in range(30):
        nn = NeuralNetwork([config.INPUT_NEURONS, config.HIDDEN1_NEURONS,
                           config.HIDDEN2_NEURONS, config.OUTPUT_NEURONS])
        rocket = Rocket(nn, 100.0, 350.0)
        for frame in range(500):
            if not rocket.alive:
                break
            rocket.update(target_x, target_y, screen_w, screen_h, None, [1.0]*5)
        if rocket.reached_target:
            trace_rocket(rocket, target_x, target_y, screen_w, screen_h, None,
                        f"TARGET-REACHING rocket #{attempt}")
            break


if __name__ == "__main__":
    main()
