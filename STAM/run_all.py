"""
Run the complete training pipeline
Execute this file in VS Code
"""

import os
import sys
import traceback

print("=" * 60)
print("STAM Complete Training Pipeline")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")

# Step 1: Generate data
print("\n[1/4] Generating data...")
try:
    exec(open('generate_data.py', encoding='utf-8').read())
    print("✅ Data generation completed")
except Exception as e:
    print(f"❌ Error in data generation: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Train LSTM-AE
print("\n[2/4] Training LSTM-AE...")
try:
    exec(open('train_lstm_ae.py', encoding='utf-8').read())
    print("✅ LSTM-AE training completed")
except Exception as e:
    print(f"❌ Error in LSTM-AE training: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Train Random Forest
print("\n[3/4] Training Random Forest...")
try:
    exec(open('train_rf.py', encoding='utf-8').read())
    print("✅ Random Forest training completed")
except Exception as e:
    print(f"❌ Error in RF training: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Show results
print("\n[4/4] Training Complete!")
print("\n✅ All models saved in outputs/ folder:")
print("   - lstm_ae_final.keras  (LSTM Autoencoder)")
print("   - rf_model.pkl          (Random Forest)")
print("   - scaler.pkl            (StandardScaler)")
print("   - threshold.txt         (Anomaly threshold)")

# List files
output_dir = os.path.join(os.getcwd(), 'outputs')
if os.path.exists(output_dir):
    print("\n📁 Files in outputs/:")
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"   - {f} ({size:,} bytes)")