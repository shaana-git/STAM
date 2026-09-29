"""
Train Binary Random Forest for TM2 vs TM3 classification
Uses reconstruction error vectors from LSTM-AE as features
TM1 is handled by Fast Path detector (not RF)
"""

import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
import config

# Paths
MODEL_PATH = os.path.join(config.OUTPUT_DIR, 'lstm_ae_final.keras')
SCALER_PATH = os.path.join(config.OUTPUT_DIR, 'scaler.pkl')

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

def build_sequences_with_meta(df, window_size=config.WINDOW_SIZE):
    """Build sliding windows and return sequences with metadata"""
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

def extract_reconstruction_errors(model, X_seq):
    """Extract per-feature reconstruction errors using MSE"""
    X_hat = model.predict(X_seq, batch_size=256, verbose=0)
    sq_err = (X_seq - X_hat) ** 2
    return sq_err.mean(axis=1)  # Shape: (n_samples, n_features)

print("=" * 60)
print("Binary Random Forest Training (TM2 vs TM3)")
print("TM1 is handled by Fast Path detector")
print("=" * 60)

# Load data
print("\nLoading attack data...")
df = load_attack_data()
print(f"   Total rows: {len(df):,}")

# Filter to TM2 and TM3 only (TM1 handled by Fast Path)
df_tm23 = df[df["label"].isin([2, 3])].copy()
print(f"   TM2+TM3 rows: {len(df_tm23):,}")

# Load scaler and LSTM model
print("\nLoading scaler...")
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

print("Loading LSTM-AE model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("   Model loaded")

# Scale features
df_tm23_scaled = df_tm23.copy()
df_tm23_scaled[config.FEATURE_COLS] = scaler.transform(df_tm23[config.FEATURE_COLS].values)

# Build sequences
print("\nBuilding sequences...")
X_tm23, meta_tm23 = build_sequences_with_meta(df_tm23_scaled)
print(f"   Sequences: {X_tm23.shape}")

# Extract reconstruction errors
print("Extracting reconstruction errors...")
ef_tm23 = extract_reconstruction_errors(model, X_tm23)

# Build error feature DataFrame
EF_COLS = [f"ef_{feat}" for feat in config.FEATURE_COLS]
ef_dict = {}
for i, feat in enumerate(config.FEATURE_COLS):
    ef_dict[f"ef_{feat}"] = ef_tm23[:, i]
ef_dict["epsilon"] = ef_tm23.mean(axis=1)
ef_dict["label"] = meta_tm23["label"].values
ef_dict["run_id"] = meta_tm23["run_id"].values
df_ef = pd.DataFrame(ef_dict)

# Load threshold from trained model (or use default)
THETA_PATH = os.path.join(config.OUTPUT_DIR, 'threshold.txt')
if os.path.exists(THETA_PATH):
    with open(THETA_PATH, 'r') as f:
        THETA = float(f.read().strip())
else:
    # Fallback percentile
    THETA = np.percentile(df_ef["epsilon"].values, config.THRESHOLD_PERCENTILE)

print(f"\nAnomaly threshold (theta): {THETA:.6f}")

# Filter flagged windows (epsilon > theta)
df_flagged = df_ef[df_ef["epsilon"] > THETA].copy()
print(f"Flagged windows (epsilon > theta): {len(df_flagged):,}")

# Check class distribution
n_tm2 = (df_flagged["label"] == 2).sum()
n_tm3 = (df_flagged["label"] == 3).sum()
print(f"\nClass distribution in flagged windows:")
print(f"   TM2 (Rogue BS): {n_tm2}")
print(f"   TM3 (HO Storm): {n_tm3}")

# Balance classes by undersampling
n_per = min(n_tm2, n_tm3)
print(f"\nBalancing to {n_per} samples per class...")

def stratified_sample(df, label, n_samples, seed=42):
    df_l = df[df["label"] == label].copy()
    if len(df_l) <= n_samples:
        return df_l
    return df_l.sample(n=n_samples, random_state=seed)

df_tm2_bal = stratified_sample(df_flagged, 2, n_per)
df_tm3_bal = stratified_sample(df_flagged, 3, n_per)
df_rf = pd.concat([df_tm2_bal, df_tm3_bal], ignore_index=True)
print(f"   Balanced dataset: TM2={len(df_tm2_bal)}, TM3={len(df_tm3_bal)}")

# Prepare features (error vectors)
FEATURE_EF_COLS = [f"ef_{feat}" for feat in config.FEATURE_COLS]
X_rf = df_rf[FEATURE_EF_COLS].values
y_rf = df_rf["label"].values
groups = df_rf["run_id"].values

# Split by run_id (no leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_SEED)
train_idx, test_idx = next(gss.split(X_rf, y_rf, groups=groups))

X_train, X_test = X_rf[train_idx], X_rf[test_idx]
y_train, y_test = y_rf[train_idx], y_rf[test_idx]

print(f"\nTrain/Test split:")
print(f"   Train: {len(X_train)} samples")
print(f"   Test:  {len(X_test)} samples")

# Train Binary Random Forest
print("\nTraining Binary Random Forest (TM2 vs TM3)...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=config.RANDOM_SEED
)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("\nValidation Results (Binary RF):")
print(classification_report(y_test, y_pred, 
                           target_names=["TM2 (Rogue BS)", "TM3 (HO Storm)"],
                           labels=[2, 3]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[2, 3])
print("\nConfusion Matrix:")
print(f"              TM2    TM3")
print(f"    TM2   {cm[0,0]:>6,} {cm[0,1]:>6,}")
print(f"    TM3   {cm[1,0]:>6,} {cm[1,1]:>6,}")

# Save model
model_path = os.path.join(config.OUTPUT_DIR, 'rf_binary_tm2_tm3.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(rf, f)
print(f"\nModel saved to {model_path}")

# Feature importance
importance = pd.DataFrame({
    'feature': FEATURE_EF_COLS,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Important Error Features:")
print(importance.head(10))

print("\nBinary RF Training complete!")
print("Note: TM1 is handled by Fast Path detector, not RF")
