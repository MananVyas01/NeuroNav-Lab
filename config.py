"""
Configuration parameters for Neural Rocket Evolution.
All tunable parameters are defined here for easy experimentation.
"""

# Screen settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
SIMULATION_AREA_WIDTH = 800
SIMULATION_AREA_HEIGHT = 700
PANEL_WIDTH = SCREEN_WIDTH - SIMULATION_AREA_WIDTH
FPS = 60

# Colors (R, G, B)
COLOR_BG = (18, 18, 30)
COLOR_PANEL = (28, 28, 45)
COLOR_ROCKET = (100, 200, 255)
COLOR_ROCKET_BEST = (255, 220, 50)
COLOR_ROCKET_SUCCESS = (50, 255, 100)
COLOR_ROCKET_DEAD = (60, 60, 80)
COLOR_ROCKET_CRASHED = (255, 100, 50)
COLOR_TARGET = (255, 80, 80)
COLOR_OBSTACLE = (50, 50, 70)
COLOR_TRAIL = (100, 200, 255)
COLOR_TEXT = (220, 220, 230)
COLOR_TEXT_DIM = (120, 120, 140)
COLOR_NET_NODE = (100, 180, 255)
COLOR_NET_WEIGHT_POS = (80, 200, 120)
COLOR_NET_WEIGHT_NEG = (255, 80, 80)
COLOR_CHAMPION = (255, 215, 0)
COLOR_GRID = (30, 30, 45)
COLOR_DEBUG = (255, 255, 100)
COLOR_SENSOR_RAY = (100, 180, 100, 60)
COLOR_SELECTED = (255, 150, 50)

# Population
POPULATION_SIZE = 300

# --- ADAPTIVE GENERATION ---
BASE_GENERATION_LENGTH = 500
EARLY_GENERATION_LENGTH = 800      # gen 1-10: longer exploration
MID_GENERATION_LENGTH = 650        # gen 11-30: moderate
LATE_GENERATION_LENGTH = 500       # gen 31+: normal
EARLY_GEN_THRESHOLD = 10
MID_GEN_THRESHOLD = 30
ALLOW_GENERATION_EXTENSION = True
MAX_EXTENSION_LENGTH = 150         # max extra frames for active rockets
GENERATION_LENGTH_MAX = 1200       # absolute cap for generation length
EXTENSION_PROGRESS_THRESHOLD = 5.0 # min distance improvement to qualify

# Evolution
MUTATION_RATE = 0.3
MUTATION_STRENGTH = 0.3
MUTATION_STRENGTH_MIN = 0.05
MUTATION_STRENGTH_DECAY = 0.995
ELITE_PERCENTAGE = 0.05
CROSSOVER_RATE = 0.5
ENABLE_CROSSOVER = True

# Neural network - 10 inputs
INPUT_NEURONS = 10
HIDDEN1_NEURONS = 12
HIDDEN2_NEURONS = 8
OUTPUT_NEURONS = 2  # turn, thrust

# Physics
MAX_SPEED = 5.0
THRUST_POWER = 0.20
ROTATION_SPEED = 0.15
FRICTION = 0.98
MIN_SPEED_THRESHOLD = 0.01

# Rocket
ROCKET_SIZE = 10
START_X_MIN = 60
START_X_MAX = 150
START_Y_MIN = 150
START_Y_MAX = 550

# Target
TARGET_RADIUS = 22
TARGET_REWARD = 1000
SPEED_REWARD = 1500

# --- REWARD SYSTEM ---
PROGRESS_WINDOW = 30
PROGRESS_REWARD_SCALE = 5.0        # stronger signal for getting closer to target
DISTANCE_REWARD_PER_FRAME = 0.3    # per-frame bonus proportional to how close we are vs start
FORWARD_REWARD_SCALE = 0.5         # reward for velocity component toward target
SPIN_PENALTY_SCALE = 0.2           # gentle nudge away from spinning, not a death sentence

DANGER_PENALTY_SCALE = 0.8         # was 3.0 - much lighter, don't punish for being near obstacles
DANGER_APPROACH_PENALTY = 0.3      # was 1.5 - only punish when moving TOWARD obstacle

STUCK_DISPLACEMENT_THRESHOLD = 2.0
STUCK_PROGRESS_THRESHOLD = 0.5
STUCK_SENSOR_THRESHOLD = 0.3
STUCK_WINDOW = 40
STUCK_PENALTY_PER_FRAME = 0.1      # was 0.3 - lighter
STUCK_MAX_PENALTY = 20.0           # was 50.0

RECOVERY_BONUS = 25.0              # was 15.0 - reward getting unstuck
RECOVERY_COOLDOWN = 40             # was 60

OBSTACLE_CRASH_PENALTY = -100      # was -200 - much lighter
BOUNDARY_PENALTY = -5              # was -20 - light tap, not death sentence

# Obstacle sensors
SENSOR_RANGE = 200
SENSOR_ANGLES = [-0.8, -0.4, 0.0, 0.4, 0.8]

# Obstacle modes
OBSTACLE_MODES = ["OFF", "SIMPLE", "MEDIUM", "HARD", "RANDOM", "MAZE"]

# Obstacle layouts (huge gaps, thin walls for easy learning)
SIMPLE_OBSTACLES = [
    {"x": 350, "y": 0, "width": 10, "height": 150},
    {"x": 350, "y": 450, "width": 10, "height": 250},
]

MEDIUM_OBSTACLES = [
    {"x": 280, "y": 0, "width": 10, "height": 150},
    {"x": 280, "y": 350, "width": 10, "height": 150},
    {"x": 280, "y": 620, "width": 10, "height": 80},
    {"x": 480, "y": 100, "width": 10, "height": 180},
    {"x": 620, "y": 0, "width": 10, "height": 250},
]

HARD_OBSTACLES = [
    {"x": 220, "y": 0, "width": 10, "height": 120},
    {"x": 220, "y": 320, "width": 10, "height": 100},
    {"x": 220, "y": 550, "width": 10, "height": 150},
    {"x": 400, "y": 80, "width": 10, "height": 150},
    {"x": 400, "y": 430, "width": 10, "height": 270},
    {"x": 550, "y": 0, "width": 10, "height": 180},
    {"x": 550, "y": 400, "width": 10, "height": 100},
    {"x": 550, "y": 620, "width": 10, "height": 80},
    {"x": 680, "y": 80, "width": 10, "height": 180},
]

MAZE_OBSTACLES = [
    {"x": 150, "y": 0, "width": 10, "height": 180},
    {"x": 150, "y": 380, "width": 10, "height": 320},
    {"x": 300, "y": 60, "width": 10, "height": 180},
    {"x": 300, "y": 480, "width": 10, "height": 220},
    {"x": 450, "y": 0, "width": 10, "height": 120},
    {"x": 450, "y": 330, "width": 10, "height": 120},
    {"x": 450, "y": 600, "width": 10, "height": 100},
    {"x": 600, "y": 30, "width": 10, "height": 220},
    {"x": 600, "y": 500, "width": 10, "height": 200},
]

# Random obstacle generation
RANDOM_MIN_OBSTACLES = 3
RANDOM_MAX_OBSTACLES = 6
RANDOM_MIN_WIDTH = 14
RANDOM_MAX_WIDTH = 20
RANDOM_MIN_HEIGHT = 80
RANDOM_MAX_HEIGHT = 250

# Training modes
GENERALIZATION_MODE = False
GENERALIZATION_MOVE_INTERVAL = 10

# Curriculum mode
CURRICULUM_MODE = False
CURRICULUM_SUCCESS_THRESHOLD = 0.70
CURRICULUM_REQUIRED_GENERATIONS = 5

# Simulation speed
SPEED_OPTIONS = [1, 2, 5, 10, 20]
DEFAULT_SPEED_INDEX = 0

# --- PLATEAU DETECTION ---
PLATEAU_WINDOW = 20               # generations to check
PLATEAU_MIN_IMPROVEMENT = 10.0    # min absolute fitness improvement to not be plateau
PLATEAU_MIN_IMPROVEMENT_RATIO = 0.01  # relative: min (improvement / |baseline|) ratio

# --- TREND INDICATOR ---
TREND_WINDOW = 10                 # generations to compare

# Neural network visualization
NN_PANEL_X = SIMULATION_AREA_WIDTH + 10
NN_PANEL_Y = 260
NN_PANEL_WIDTH = PANEL_WIDTH - 20
NN_PANEL_HEIGHT = 280
NN_NODE_RADIUS = 5

# Learning graph
GRAPH_X = SIMULATION_AREA_WIDTH + 10
GRAPH_Y = 570
GRAPH_WIDTH = PANEL_WIDTH - 20
GRAPH_HEIGHT = 100
GRAPH_HISTORY_LENGTH = 100

# File paths
SAVE_DIR = "models"
SAVE_FILE = "best_network.npz"
