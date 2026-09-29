"""
Configuration for STAM - 15 features, TM1/TM2/TM3 attacks
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
NORMAL_DATA_DIR = os.path.join(OUTPUT_DIR, 'normal')
ATTACK_DATA_DIR = os.path.join(OUTPUT_DIR, 'attack')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

# Create directories safely
try:
    os.makedirs(NORMAL_DATA_DIR, exist_ok=True)
except FileExistsError:
    pass

try:
    os.makedirs(ATTACK_DATA_DIR, exist_ok=True)
except FileExistsError:
    pass

try:
    os.makedirs(MODELS_DIR, exist_ok=True)
except FileExistsError:
    pass

# Data parameters
N_FEATURES = 15
WINDOW_SIZE = 10
STRIDE = 1
SEQUENCE_LENGTH = 10

# Training parameters
MAX_EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
RANDOM_SEED = 42

# LSTM-AE architecture
LATENT_DIM = 8
ENCODE_UNITS = [64, 32]
DECODE_UNITS = [64, 32]

# Threshold calibration - UPDATED
THRESHOLD_PERCENTILE = 97
HUBER_DELTA = 1.0

# Feature columns (must match simulator)
FEATURE_COLS = [
    "rrc_conn_req_count",
    "rrc_setup_success_ratio",
    "prach_attempt_count",
    "prach_failure_rate",
    "ho_attempt_count",
    "ho_failure_ratio",
    "rrc_reconfig_count",
    "rrc_conn_reestab_req_count",
    "reestab_to_release_ratio",
    "active_ue_count",
    "prb_utilization_ul",
    "avg_cqi",
    "signaling_resource_util",
    "burst_intensity",
    "access_synchrony_score"
]

# Attack type mapping - CORRECTED
# TM1 = Botnet/RRC Flood
# TM2 = Rogue BS / Fake BS
# TM3 = HO Storm
ATTACK_LABELS = {
    "normal": 0,
    "tm1_flood": 1,
    "tm2_fake_bs": 2,
    "tm3_ho_storm": 3
}
