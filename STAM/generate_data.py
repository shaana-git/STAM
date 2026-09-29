"""
Data preprocessing for LSTM-AE training
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import pickle
import os
import config


def load_normal_data():
    """Load all normal data"""
    combined = os.path.join(config.NORMAL_DATA_DIR, 'normal_dataset.csv')
    if os.path.exists(combined):
        return pd.read_csv(combined)
    
    # Otherwise combine individual files
    all_dfs = []
    for f in os.listdir(config.NORMAL_DATA_DIR):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(config.NORMAL_DATA_DIR, f))
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def load_attack_data():
    """Load attack dataset"""
    combined = os.path.join(config.ATTACK_DATA_DIR, 'attack_dataset.csv')
    if os.path.exists(combined):
        return pd.read_csv(combined)
    
    all_dfs = []
    for f in os.listdir(config.ATTACK_DATA_DIR):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(config.ATTACK_DATA_DIR, f))
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def create_sequences(data, window_size=config.WINDOW_SIZE, stride=config.STRIDE):
    """Create sliding window sequences"""
    X = []
    for i in range(0, len(data) - window_size + 1, stride):
        X.append(data[i:i+window_size])
    return np.array(X)


def build_sequences_with_meta(df, window_size=config.WINDOW_SIZE):
    """Build sliding windows and return sequences with metadata for attack data"""
    X_list = []
    meta_list = []
    for (run_id, cell_id), grp in df.groupby(["run_id", "cell_id"]):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        vals = grp[config.FEATURE_COLS].values
        for i in range(len(vals) - window_size + 1):
            X_list.append(vals[i:i + window_size])
            last = grp.iloc[i + window_size - 1]
            meta_list.append({
                "run_id": run_id,
                "cell_id": cell_id,
                "label": int(last["label"]),
            })
    return np.array(X_list, dtype=np.float32), pd.DataFrame(meta_list)


def compute_epsilon_huber(X_seq, model, delta=config.HUBER_DELTA):
    """
    Compute reconstruction error using Huber loss
    This matches the paper's requirement (C5 fix)
    """
    huber_fn = tf.keras.losses.Huber(delta=delta, reduction="none")
    X_hat = model.predict(X_seq, batch_size=256, verbose=0)
    errors = huber_fn(X_seq, X_hat).numpy()
    return errors.mean(axis=1)  # Shape: (n_samples,)


def compute_epsilon_mse(X_seq, model):
    """Compute reconstruction error using MSE (fallback)"""
    X_hat = model.predict(X_seq, batch_size=256, verbose=0)
    sq_err = (X_seq - X_hat) ** 2
    return sq_err.mean(axis=1)


def prepare_training_data():
    """Main function to prepare training data"""
    
    print("=" * 60)
    print("Preparing Training Data")
    print("=" * 60)
    
    # Load data
    print("\nLoading normal data...")
    df = load_normal_data()
    print(f"   Total rows: {len(df):,}")
    print(f"   Unique runs: {df['run_id'].nunique()}")
    
    # Filter to training split
    df_train = df[df['split'] == 'train'].copy()
    print(f"   Training rows: {len(df_train):,}")
    
    # Extract features
    feature_cols = [c for c in config.FEATURE_COLS if c in df_train.columns]
    data_raw = df_train[feature_cols].values
    
    # Fit scaler
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_raw)
    
    # Create sequences per run (avoid mixing runs)
    print("\nCreating sliding windows...")
    X_sequences = []
    for run_id, group in df_train.groupby('run_id'):
        run_indices = group.index
        run_start = run_indices[0]
        run_end = run_indices[-1] + 1
        run_data = data_scaled[run_start:run_end]
        seqs = create_sequences(run_data)
        if len(seqs) > 0:
            X_sequences.append(seqs)
    
    X = np.concatenate(X_sequences, axis=0)
    print(f"   Total sequences: {len(X):,}")
    print(f"   Shape: {X.shape}")
    
    # Save scaler
    scaler_path = os.path.join(config.OUTPUT_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nScaler saved to {scaler_path}")
    
    # Save training data
    X_path = os.path.join(config.OUTPUT_DIR, 'X_train_normal.npy')
    np.save(X_path, X)
    print(f"Training data saved to {X_path}")
    
    # Also prepare validation data
    df_val = df[df['split'] == 'val'].copy()
    if len(df_val) > 0:
        val_raw = df_val[feature_cols].values
        val_scaled = scaler.transform(val_raw)
        
        X_val_sequences = []
        for run_id, group in df_val.groupby('run_id'):
            run_indices = group.index
            run_start = run_indices[0]
            run_end = run_indices[-1] + 1
            run_data = val_scaled[run_start:run_end]
            seqs = create_sequences(run_data)
            if len(seqs) > 0:
                X_val_sequences.append(seqs)
        
        X_val = np.concatenate(X_val_sequences, axis=0)
        X_val_path = os.path.join(config.OUTPUT_DIR, 'X_val_normal.npy')
        np.save(X_val_path, X_val)
        print(f"Validation data saved to {X_val_path}")
    
    return X


def prepare_attack_error_data(model_path=None, scaler_path=None):
    """
    Prepare reconstruction error data for attack classification
    Returns DataFrame with ef_* features and labels
    """
    print("=" * 60)
    print("Preparing Attack Error Data")
    print("=" * 60)
    
    # Load model
    if model_path is None:
        model_path = os.path.join(config.OUTPUT_DIR, 'lstm_ae_final.keras')
    
    if scaler_path is None:
        scaler_path = os.path.join(config.OUTPUT_DIR, 'scaler.pkl')
    
    print("\nLoading LSTM model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading scaler...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load attack data
    print("Loading attack data...")
    df_attack = load_attack_data()
    
    # Filter to TM2 and TM3 only (TM1 handled by Fast Path)
    df_tm23 = df_attack[df_attack["label"].isin([2, 3])].copy()
    print(f"   TM2+TM3 rows: {len(df_tm23):,}")
    
    # Scale features
    df_scaled = df_tm23.copy()
    df_scaled[config.FEATURE_COLS] = scaler.transform(df_tm23[config.FEATURE_COLS].values)
    
    # Build sequences
    print("Building sequences...")
    X_seq, meta = build_sequences_with_meta(df_scaled)
    print(f"   Sequences: {X_seq.shape}")
    
    # Extract reconstruction errors using Huber loss
    print("Extracting reconstruction errors (Huber loss)...")
    ef = compute_epsilon_huber(X_seq, model)
    
    # Build error feature DataFrame
    EF_COLS = [f"ef_{feat}" for feat in config.FEATURE_COLS]
    ef_dict = {}
    for i, feat in enumerate(config.FEATURE_COLS):
        ef_dict[f"ef_{feat}"] = ef[:, i]
    ef_dict["epsilon"] = ef.mean(axis=1)
    ef_dict["label"] = meta["label"].values
    ef_dict["run_id"] = meta["run_id"].values
    
    df_ef = pd.DataFrame(ef_dict)
    
    # Load threshold
    threshold_path = os.path.join(config.OUTPUT_DIR, 'threshold.txt')
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            THETA = float(f.read().strip())
    else:
        THETA = np.percentile(ef.mean(axis=1), config.THRESHOLD_PERCENTILE)
    
    print(f"   Threshold (theta): {THETA:.6f}")
    
    # Filter flagged windows
    df_flagged = df_ef[df_ef["epsilon"] > THETA].copy()
    print(f"   Flagged windows (epsilon > theta): {len(df_flagged):,}")
    
    return df_flagged, THETA
