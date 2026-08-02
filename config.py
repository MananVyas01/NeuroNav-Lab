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
POPULATION_SIZE = 120

# --- ADAPTIVE GENERATION ---
BASE_GENERATION_LENGTH = 500
EARLY_GENERATION_LENGTH = 800      # gen 1-10: longer exploration
MID_GENERATION_LENGTH = 650        # gen 11-30: moderate
LATE_GENERATION_LENGTH = 500       # gen 31+: normal
EARLY_GEN_THRESHOLD = 10
MID_GEN_THRESHOLD = 30
MAX_EXTENSION_LENGTH = 150         # max extra frames for active rockets
EXTENSION_PROGRESS_THRESHOLD = 5.0 # min distance improvement to qualify

# Evolution
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2
ELITE_PERCENTAGE = 0.1
CROSSOVER_RATE = 0.4

# Neural network - 16 inputs
INPUT_NEURONS = 16
HIDDEN1_NEURONS = 20
HIDDEN2_NEURONS = 16
OUTPUT_NEURONS = 2  # turn, thrust

# Physics
MAX_SPEED = 3.0
THRUST_POWER = 0.08
ROTATION_SPEED = 0.06
FRICTION = 0.96
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
PROGRESS_REWARD_SCALE = 0.5
NO_PROGRESS_PENALTY = -0.2

DANGER_PENALTY_SCALE = 3.0
DANGER_APPROACH_PENALTY = 1.5

STUCK_DISPLACEMENT_THRESHOLD = 2.0
STUCK_PROGRESS_THRESHOLD = 0.5
STUCK_SENSOR_THRESHOLD = 0.3
STUCK_WINDOW = 40
STUCK_PENALTY_PER_FRAME = 0.3
STUCK_MAX_PENALTY = 50.0

RECOVERY_BONUS = 15.0
RECOVERY_COOLDOWN = 60

OBSTACLE_CRASH_PENALTY = -300
BOUNDARY_PENALTY = -50

# Obstacle sensors
SENSOR_RANGE = 120
SENSOR_ANGLES = [-0.8, -0.4, 0.0, 0.4, 0.8]

# Obstacle modes
OBSTACLE_MODES = ["OFF", "SIMPLE", "MEDIUM", "HARD", "RANDOM", "MAZE"]

# Obstacle layouts
SIMPLE_OBSTACLES = [
    {"x": 320, "y": 0, "width": 18, "height": 280},
    {"x": 320, "y": 380, "width": 18, "height": 320},
]

MEDIUM_OBSTACLES = [
    {"x": 260, "y": 0, "width": 18, "height": 200},
    {"x": 260, "y": 280, "width": 18, "height": 220},
    {"x": 260, "y": 580, "width": 18, "height": 120},
    {"x": 460, "y": 150, "width": 18, "height": 250},
    {"x": 620, "y": 0, "width": 18, "height": 350},
]

HARD_OBSTACLES = [
    {"x": 200, "y": 0, "width": 18, "height": 180},
    {"x": 200, "y": 250, "width": 18, "height": 150},
    {"x": 200, "y": 470, "width": 18, "height": 230},
    {"x": 380, "y": 100, "width": 18, "height": 200},
    {"x": 380, "y": 380, "width": 18, "height": 320},
    {"x": 530, "y": 0, "width": 18, "height": 280},
    {"x": 530, "y": 350, "width": 18, "height": 150},
    {"x": 530, "y": 570, "width": 18, "height": 130},
    {"x": 680, "y": 150, "width": 18, "height": 250},
]

MAZE_OBSTACLES = [
    {"x": 150, "y": 0, "width": 18, "height": 250},
    {"x": 150, "y": 320, "width": 18, "height": 380},
    {"x": 300, "y": 100, "width": 18, "height": 250},
    {"x": 300, "y": 420, "width": 18, "height": 280},
    {"x": 450, "y": 0, "width": 18, "height": 200},
    {"x": 450, "y": 270, "width": 18, "height": 180},
    {"x": 450, "y": 520, "width": 18, "height": 180},
    {"x": 600, "y": 80, "width": 18, "height": 300},
    {"x": 600, "y": 450, "width": 18, "height": 250},
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
PLATEAU_WINDOW = 15               # generations to check
PLATEAU_SUCCESS_THRESHOLD = 2.0   # min % change to not be plateau
PLATEAU_FITNESS_THRESHOLD = 5.0   # min fitness % change

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
