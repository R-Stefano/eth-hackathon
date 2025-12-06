SEED = 42
NUM_EPISODES = 200  # Reduced for faster debugging
MAX_STEPS_PER_EPISODE = 10  # Single step per episode for clearer learning signal

# Simplified for debugging - only HLA-B with ON/OFF
# Read the genes from seneset.txt
with open("SenCID/SenCID/resource/seneset.txt") as f:
    AVAILABLE_GENES_TO_INTERVENE = [line.strip() for line in f if line.strip()]

# AVAILABLE_GENES_TO_INTERVENE = ["HLA-B"]
AVAILABLE_ACTIONS = ["ON", "OFF"]
TARGET_SID = "SID1"
OUTPUT_SIZE = len(AVAILABLE_GENES_TO_INTERVENE) * len(AVAILABLE_ACTIONS)  # = 2
HIDDEN_SIZE = 64
INPUT_SIZE = 1290  # Number of senescence-related genes after ScaleData
LR_RATE = 1e-3

HARDOCODED_REWARD = False
VERBOSE = True