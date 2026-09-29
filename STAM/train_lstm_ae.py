"""
Train LSTM Autoencoder on normal data
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import config
from src.model import build_lstm_ae
from src.data_preprocessing import prepare_training_data

# Set seeds
tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

print("=" * 60)
print("LSTM-AE Training (15 features)")
print("=" * 60)

# Prepare data if not already done
if not os.path.exists(os.path.join(config.OUTPUT_DIR, 'X_train_normal.npy')):
    print("\nPreparing data first...")
    prepare_training_data()

# Load data
print("\nLoading training data...")
X_train = np.load(os.path.join(config.OUTPUT_DIR, 'X_train_normal.npy'))
X_val = np.load(os.path.join(config.OUTPUT_DIR, 'X_val_normal.npy'))

print(f"   X_train: {X_train.shape}")
print(f"   X_val:   {X_val.shape}")

# Build model
print("\nBuilding LSTM-AE...")
model = build_lstm_ae()
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(
        filepath=os.path.join(config.MODELS_DIR, 'lstm_ae_best.keras'),
        monitor='val_loss', save_best_only=True, verbose=1
    )
]

# Train
print("\nTraining...")
history = model.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=config.MAX_EPOCHS,
    batch_size=config.BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# Save final model
final_path = os.path.join(config.OUTPUT_DIR, 'lstm_ae_final.keras')
model.save(final_path)
print(f"\nModel saved to {final_path}")

# Calculate threshold using Huber loss (as per paper C5)
print("\nCalibrating threshold with Huber loss...")

# Load validation data
df_val = pd.read_csv(os.path.join(config.NORMAL_DATA_DIR, 'normal_dataset.csv'))
df_val = df_val[df_val['split'] == 'val'].copy()

# Load scaler
import pickle
scaler_path = os.path.join(config.OUTPUT_DIR, 'scaler.pkl')
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Scale and create sequences
feature_cols = [c for c in config.FEATURE_COLS if c in df_val.columns]
df_val_scaled = df_val.copy()
df_val_scaled[feature_cols] = scaler.transform(df_val[feature_cols].values)

X_val_seq = []
for run_id, group in df_val_scaled.groupby('run_id'):
    group = group.sort_values('timestamp')
    vals = group[feature_cols].values
    for i in range(0, len(vals) - config.WINDOW_SIZE + 1, config.STRIDE):
        X_val_seq.append(vals[i:i + config.WINDOW_SIZE])

X_val_seq = np.array(X_val_seq, dtype=np.float32)

# Compute Huber loss errors
huber_fn = tf.keras.losses.Huber(delta=config.HUBER_DELTA, reduction="none")
X_hat = model.predict(X_val_seq, batch_size=256, verbose=0)
errors = huber_fn(X_val_seq, X_hat).numpy()
errors_val = errors.mean(axis=1)

threshold = np.percentile(errors_val, config.THRESHOLD_PERCENTILE)

# Save threshold
threshold_path = os.path.join(config.OUTPUT_DIR, 'threshold.txt')
with open(threshold_path, 'w') as f:
    f.write(str(threshold))
print(f"Threshold ({config.THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}")

# Save config.json for inference
config_data = {
    "WINDOW_SIZE": config.WINDOW_SIZE,
    "N_FEATURES": config.N_FEATURES,
    "THRESHOLD": float(threshold),
    "THRESHOLD_PCT": config.THRESHOLD_PERCENTILE,
    "HUBER_DELTA": config.HUBER_DELTA,
    "SCORE_LOSS": "huber",
    "FEATURE_COLS": config.FEATURE_COLS,
}

config_json_path = os.path.join(config.OUTPUT_DIR, 'config.json')
with open(config_json_path, 'w') as f:
    json.dump(config_data, f, indent=2)
print(f"Config saved to {config_json_path}")

print("\nTraining complete!")
