#!/usr/bin/env python3
"""
Read KPM data from CSV file and run LSTM inference
UPDATED for new simulator (15 features, TM1/TM2/TM3 attacks)
CSV format: 26 comma-separated values per line (first 15 are real)

PROPOSED WORK IMPLEMENTATION:
- Algorithm 1: Fast Path (TM1 detection)
- Algorithm 2: Slow Path (LSTM-AE + Binary RF for TM2/TM3)
- Algorithm 3: TM3 Confirmer (secondary verification)
"""

import tensorflow as tf
import numpy as np
import pickle
import sys
import os
import time
import json
from collections import deque, Counter

# ============================================================================
# CONFIGURATION - ALL PATHS RELATIVE OR FROM CONFIG
# ============================================================================

# Base directory (where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Paths - using best model from training
MODEL_PATH = os.path.join(OUTPUT_DIR, 'models', 'lstm_ae_best.keras')
SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl')
RF_PATH = os.path.join(OUTPUT_DIR, 'rf_binary_tm2_tm3.pkl')
CONFIG_PATH = os.path.join(OUTPUT_DIR, 'config.json')
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, 'threshold.txt')

# Use /tmp for communication with xApp and E2 Agent
CSV_PATH = '/tmp/kpm_values.csv'
ANOMALY_OUTPUT_PATH = '/tmp/anomaly_results.csv'
WINDOW_SIZE = 10
N_FEATURES = 15

# Load configuration from config.json (created during training)
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    THRESHOLD = cfg.get('THRESHOLD', 1.0)
    WINDOW_SIZE = cfg.get('WINDOW_SIZE', 10)
    N_FEATURES = cfg.get('N_FEATURES', 15)
    print(f"Loaded config from {CONFIG_PATH}")
    print(f"  Threshold: {THRESHOLD:.6f}")
    print(f"  Window Size: {WINDOW_SIZE}")
    print(f"  Features: {N_FEATURES}")
else:
    # Fallback to threshold.txt if config.json missing
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, 'r') as f:
            THRESHOLD = float(f.read().strip())
        print(f"Loaded threshold from {THRESHOLD_PATH}: {THRESHOLD:.6f}")
    else:
        THRESHOLD = 1.398223
        print(f"Using default threshold: {THRESHOLD:.6f}")

# Fast Path parameters (from paper, can be tuned)
FAST_PATH_ZSCORE = 4.0
FAST_PATH_BURST = 2.5
FAST_PATH_SR_THRESHOLD = 0.7

# ============================================================================
# Feature indices for key metrics (15 features)
# ============================================================================
RRC_CONN_REQ_IDX = 0
RRC_SUCCESS_IDX = 1
PRACH_ATTEMPT_IDX = 2
PRACH_FAILURE_IDX = 3
HO_ATTEMPT_IDX = 4
HO_FAILURE_IDX = 5
RRC_RECONFIG_IDX = 6
RRC_REESTAB_IDX = 7
REESTAB_RELEASE_IDX = 8
ACTIVE_UE_IDX = 9
PRB_UTIL_IDX = 10
AVG_CQI_IDX = 11
SIG_UTIL_IDX = 12
BURST_INTENSITY_IDX = 13
SYNC_SCORE_IDX = 14

# Attack type mapping
ATTACK_TYPES = {
    1: "tm1_flood",
    2: "tm2_fake_bs",
    3: "tm3_ho_storm"
}

# Legacy names for xapp compatibility
ATTACK_LEGACY_NAMES = {
    "tm1_flood": "botnet",
    "tm2_fake_bs": "rogue_bs",
    "tm3_ho_storm": "handover_storm"
}

# Feature columns for error vector
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

# ============================================================================
# Fast Path Detector (TM1 ONLY)
# ============================================================================

class FastPathDetector:
    def __init__(self, theta_z, theta_z_burst, theta_sr):
        self.theta_z = theta_z
        self.theta_z_burst = theta_z_burst
        self.theta_sr = theta_sr

    def detect(self, row_scaled, rrc_success, rrc_req, prach_att, burst):
        if rrc_success >= self.theta_sr:
            return False
        z_rrc = row_scaled[RRC_CONN_REQ_IDX]
        z_prach = row_scaled[PRACH_ATTEMPT_IDX]
        count_gate = (z_rrc > self.theta_z and z_prach > self.theta_z)
        burst_gate = (burst > self.theta_z_burst and z_rrc > 0.8 and z_prach > 0.5)
        return count_gate or burst_gate

# ============================================================================
# TM3 Confirmer (Secondary Verification for HO Storm)
# ============================================================================

class TM3Confirmer:
    def __init__(self, theta_cqi=8.0, reestab_nmin=3, window_s=30):
        self.theta_cqi = theta_cqi
        self.reestab_nmin = reestab_nmin
        self.window_s = window_s
        self.reestab_buf = []
        self.cqi_buf = []
        self.confirmed = False

    def update(self, timestamp, reestab_count, avg_cqi):
        if self.confirmed:
            return True

        self.reestab_buf.append((timestamp, reestab_count))
        self.cqi_buf.append((timestamp, avg_cqi))

        cutoff = timestamp - self.window_s

        self.reestab_buf = [(ts, v) for ts, v in self.reestab_buf if ts >= cutoff]
        self.cqi_buf = [(ts, v) for ts, v in self.cqi_buf if ts >= cutoff]

        reestab_active = sum(1 for _, v in self.reestab_buf if v > 0)
        mean_cqi = np.mean([v for _, v in self.cqi_buf]) if self.cqi_buf else 15.0

        if reestab_active >= self.reestab_nmin and mean_cqi < self.theta_cqi:
            self.confirmed = True
            return True

        return False

    def reset(self):
        self.reestab_buf = []
        self.cqi_buf = []
        self.confirmed = False

# ============================================================================
# Helper Functions
# ============================================================================

def preprocess_row(values, scaler):
    """Normalize a single row of 15 features"""
    values_array = np.array(values[:N_FEATURES]).reshape(1, -1)
    return scaler.transform(values_array).flatten()

def compute_reconstruction_errors(model, window_norm):
    """Compute per-feature reconstruction errors for a window"""
    window_input = window_norm.reshape(1, WINDOW_SIZE, N_FEATURES)
    reconstructed = model.predict(window_input, verbose=0)
    sq_err = (window_input - reconstructed) ** 2
    return sq_err.mean(axis=1).flatten()

# ============================================================================
# MAIN
# ============================================================================

print("=" * 60)
print("STAM Inference Script (Proposed Work Implementation)")
print(f"LSTM Threshold: {THRESHOLD:.6f}")
print(f"Fast Path: Z={FAST_PATH_ZSCORE}, Burst={FAST_PATH_BURST}, SR={FAST_PATH_SR_THRESHOLD}")
print("Attack Mapping: TM1=Flood, TM2=Rogue_BS, TM3=HO_Storm")
print("=" * 60)

# Remove old CSV files at startup
if os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)
    print(f"Removed old {CSV_PATH}")

if os.path.exists(ANOMALY_OUTPUT_PATH):
    os.remove(ANOMALY_OUTPUT_PATH)
    print(f"Removed old {ANOMALY_OUTPUT_PATH}")

print("Fresh start - waiting for new data...")

# Load model
print("\nLoading LSTM model...")
if not os.path.exists(MODEL_PATH):
    # Fallback to final model if best not found
    MODEL_PATH = os.path.join(OUTPUT_DIR, 'lstm_ae_final.keras')
    print(f"Best model not found, using: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"LSTM model loaded from {MODEL_PATH}")

# Load scaler
print("Loading scaler...")
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
print("Scaler loaded")

# Load Binary RF (TM2 vs TM3)
rf_model = None
if os.path.exists(RF_PATH):
    try:
        print("Loading Binary RF (TM2 vs TM3)...")
        with open(RF_PATH, 'rb') as f:
            rf_model = pickle.load(f)
        print("Binary RF loaded")
    except Exception as e:
        print(f"Could not load RF: {e}")
else:
    print("Binary RF model not found. Only Fast Path will work.")

# Initialize detectors
fast_detector = FastPathDetector(FAST_PATH_ZSCORE, FAST_PATH_BURST, FAST_PATH_SR_THRESHOLD)

# Buffers
buffer = deque(maxlen=WINDOW_SIZE)
last_position = 0
window_count = 0
file_size_last = 0
tm3_verifier = TM3Confirmer()

print(f"\nWatching {CSV_PATH} for new data...")
print("-" * 60)

while True:
    try:
        if not os.path.exists(CSV_PATH):
            time.sleep(0.5)
            continue

        current_size = os.path.getsize(CSV_PATH)
        if current_size < file_size_last:
            last_position = 0
            buffer.clear()
            tm3_verifier.reset()
        file_size_last = current_size

        with open(CSV_PATH, 'r') as f:
            f.seek(last_position)
            new_lines = f.readlines()
            last_position = f.tell()

        for line in new_lines:
            line = line.strip()
            if not line:
                continue

            values_raw = [float(x) for x in line.split(',')]
            values = values_raw[:N_FEATURES]
            buffer.append(values)

            if len(buffer) == WINDOW_SIZE:
                window_count += 1
                window = np.array(buffer)

                # Calculate window averages
                avg_rrc = np.mean(window[:, RRC_CONN_REQ_IDX])
                avg_prach = np.mean(window[:, PRACH_ATTEMPT_IDX])
                avg_ho = np.mean(window[:, HO_ATTEMPT_IDX])
                avg_cqi = np.mean(window[:, AVG_CQI_IDX])
                avg_burst = np.mean(window[:, BURST_INTENSITY_IDX])
                avg_reestab = np.mean(window[:, RRC_REESTAB_IDX])
                avg_rrc_success = np.mean(window[:, RRC_SUCCESS_IDX])

                # Normalize window
                window_norm = np.zeros((WINDOW_SIZE, N_FEATURES))
                for i in range(WINDOW_SIZE):
                    try:
                        window_norm[i] = preprocess_row(window[i], scaler)
                    except Exception:
                        window_norm[i] = window[i]

                # LSTM-AE inference
                window_input = window_norm.reshape(1, WINDOW_SIZE, N_FEATURES)
                reconstructed = model.predict(window_input, verbose=0)
                mse = np.mean(np.square(window_input - reconstructed))
                is_anomaly = mse > THRESHOLD

                # Classification
                attack_type = "unknown"
                legacy_type = "unknown"

                if is_anomaly:
                    # Step 1: Fast Path for TM1 (Algorithm 1)
                    last_row_norm = window_norm[-1]
                    is_tm1 = fast_detector.detect(
                        last_row_norm,
                        window[-1][RRC_SUCCESS_IDX],
                        window[-1][RRC_CONN_REQ_IDX],
                        window[-1][PRACH_ATTEMPT_IDX],
                        window[-1][BURST_INTENSITY_IDX]
                    )

                    if is_tm1:
                        attack_type = "tm1_flood"
                        legacy_type = "botnet"
                    else:
                        # Step 2: Binary RF for TM2 vs TM3 (Algorithm 2)
                        if rf_model is not None:
                            ef_vector = compute_reconstruction_errors(model, window_norm)
                            ef_vector = ef_vector.reshape(1, -1)
                            rf_pred = rf_model.predict(ef_vector)[0]

                            if rf_pred == 2:
                                attack_type = "tm2_fake_bs"
                                legacy_type = "rogue_bs"
                            elif rf_pred == 3:
                                attack_type = "tm3_ho_storm"
                                legacy_type = "handover_storm"
                            else:
                                attack_type = "tm1_flood"
                                legacy_type = "botnet"
                    
                    # Step 3: TM3 Confirmer for HO Storm (Algorithm 3)
                    if attack_type == "tm3_ho_storm":
                        verified = tm3_verifier.update(time.time(), avg_reestab, avg_cqi)
                        if not verified:
                            attack_type = "unconfirmed_tm3"
                            legacy_type = "unknown"
                else:
                    tm3_verifier.reset()

                status = "ANOMALY" if is_anomaly else "NORMAL"
                print(f"\nWINDOW #{window_count} | MSE={mse:.4f} | {status} | Type={attack_type}")
                print(f"   RRC={avg_rrc:.1f}, PRACH={avg_prach:.1f}, HO={avg_ho:.1f}, CQI={avg_cqi:.1f}")

                # Write to anomaly_results.csv (7 columns for xapp)
                with open(ANOMALY_OUTPUT_PATH, 'a') as f:
                    f.write(f"{time.time()},{mse:.6f},{1 if is_anomaly else 0},{legacy_type},{avg_rrc:.2f},{avg_prach:.2f},{avg_ho:.2f}\n")

                # Slide window (stride=1 as per new config)
                if buffer:
                    buffer.popleft()

        time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped by user")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)
