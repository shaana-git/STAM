# -*- coding: utf-8 -*-
"""
Global configuration parameters for the 5G V2X anomaly detection project.
All constants are defined here for easy tuning.
"""

import numpy as np

# ─── NETWORK PARAMETERS (3GPP TR 37.885) ──────────────────────────────
NUM_CELLS          = 3        # Tri‑sector gNB
SIM_DURATION_S     = 120      # Total simulation time per run (seconds)
ATTACK_ONSET_S     = 30       # Attack starts at t=30s (30s clean baseline first)
T                  = SIM_DURATION_S  # alias

# ─── SLIDING WINDOW PARAMETERS ────────────────────────────────────────
WINDOW_SIZE        = 10       # 10‑second window
WINDOW_STRIDE      = 1        # 1‑second stride
N_WINDOWS          = T - WINDOW_SIZE + 1  # 110 windows per run

# ─── DATASET SIZE ─────────────────────────────────────────────────────
# Normal: 5 sub‑scenarios × 40 runs = 200 runs × 110 windows = 22,000 training windows
#        + 5 sub‑scenarios × 10 runs =  50 runs × 110 windows =  5,500 validation windows
RUNS_PER_SUBSCENARIO_TRAIN = 40
RUNS_PER_SUBSCENARIO_VAL   = 10

# ─── SUB‑SCENARIO DEFINITIONS ─────────────────────────────────────────
SUBSCENARIOS = [
    {
        'id':           1,
        'name':         'Urban_Low',
        'n_vehicles':   30,
        'speed_kmh':    40,
        'mobility':     'urban',
        'tl_burst_n':   (8, 10),    # traffic‑light burst: min,max vehicles
        'tl_freq':      0.10,
        'convoy_ho_n':  (0, 0),
    },
    {
        'id':           2,
        'name':         'Urban_Medium',
        'n_vehicles':   70,
        'speed_kmh':    50,
        'mobility':     'urban',
        'tl_burst_n':   (15, 20),
        'tl_freq':      0.10,
        'convoy_ho_n':  (0, 0),
    },
    {
        'id':           3,
        'name':         'Urban_High',
        'n_vehicles':   120,
        'speed_kmh':    60,
        'mobility':     'urban',
        'tl_burst_n':   (25, 30),
        'tl_freq':      0.10,
        'convoy_ho_n':  (0, 0),
    },
    {
        'id':           4,
        'name':         'Highway_Low',
        'n_vehicles':   40,
        'speed_kmh':    100,
        'mobility':     'highway',
        'tl_burst_n':   (0, 0),
        'tl_freq':      0.0,
        'convoy_ho_n':  (8, 12),
    },
    {
        'id':           5,
        'name':         'Highway_High',
        'n_vehicles':   100,
        'speed_kmh':    120,
        'mobility':     'highway',
        'tl_burst_n':   (0, 0),
        'tl_freq':      0.0,
        'convoy_ho_n':  (20, 25),
    },
]

# ─── ATTACK INTENSITIES ───────────────────────────────────────────────
INTENSITIES = {
    'mild':     0.10,
    'moderate': 0.30,
    'severe':   0.60,
}
RUNS_PER_INTENSITY = 17   # per (attack type, sub‑scenario, intensity)

# ─── FEATURE COLUMNS (26 KPMs) ────────────────────────────────────────
FEATURE_COLS = [
    'rrc_conn_req_count', 'rrc_setup_complete_count', 'rrc_setup_success_ratio',
    'rrc_release_count', 'rrc_conn_reestab_req_count', 'rrc_conn_reestab_success_ratio',
    'prach_attempt_count', 'prach_success_count', 'prach_failure_rate',
    'rar_granted_count', 'msg3_failure_count',
    'ho_attempt_count', 'ho_success_count', 'ho_failure_ratio',
    'ho_prep_timeout_count', 'rrc_reconfig_count',
    'active_ue_count', 'prb_utilization_ul', 'prb_utilization_dl',
    'avg_cqi', 'avg_throughput_dl_mbps', 'signaling_resource_utilization',
    'burst_intensity', 'reestab_to_release_ratio',
    'access_synchrony_score', 'ue_attach_latency_ms'
]
N_FEATURES = len(FEATURE_COLS)

# ─── EXPECTED RANGES FOR VALIDATION ───────────────────────────────────
EXPECTED_RANGES = {
    'rrc_conn_req_count':             (0,  60),
    'rrc_setup_success_ratio':        (0.72, 1.0),
    'rrc_release_count':              (0,  20),
    'rrc_conn_reestab_req_count':     (0,  5),
    'prach_failure_rate':             (0.0, 0.15),
    'ho_attempt_count':               (0,  50),
    'ho_failure_ratio':               (0.0, 0.15),
    'active_ue_count':                (3,  150),
    'prb_utilization_ul':             (0.03, 0.80),
    'prb_utilization_dl':             (0.05, 0.80),
    'avg_cqi':                        (6,  15),
    'signaling_resource_utilization': (0.02, 0.70),
    'burst_intensity':                (0.0, 12.0),
    'reestab_to_release_ratio':       (0.0, 5.0),
    'access_synchrony_score':         (0.0, 1.0),
    'ue_attach_latency_ms':           (5,  55),
}

# ─── LSTM‑AE HYPERPARAMETERS ──────────────────────────────────────────
LATENT_DIM   = 32
ENCODER_DIM  = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE   = 256
MAX_EPOCHS   = 100

# Random seed for reproducibility
RANDOM_SEED = 42
