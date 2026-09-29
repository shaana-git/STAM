"""
Data preprocessing for LSTM-AE training
"""

import numpy as np
import pandas as pd
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


def create_sequences(data, window_size=config.WINDOW_SIZE, stride=config.STRIDE):
    """Create sliding window sequences"""
    X = []
    for i in range(0, len(data) - window_size + 1, stride):
        X.append(data[i:i+window_size])
    return np.array(X)


def prepare_training_data():
    """Main function to prepare training data"""
    
    print("=" * 60)
    print("Preparing Training Data")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading normal data...")
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
    print("\n📏 Fitting StandardScaler...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_raw)
    
    # Create sequences per run (avoid mixing runs)
    print("\n🪟 Creating sliding windows...")
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
    print(f"\n💾 Scaler saved to {scaler_path}")
    
    # Save training data
    X_path = os.path.join(config.OUTPUT_DIR, 'X_train_normal.npy')
    np.save(X_path, X)
    print(f"💾 Training data saved to {X_path}")
    
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
        print(f"💾 Validation data saved to {X_val_path}")
    
    return X