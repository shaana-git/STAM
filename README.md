# STAM-Signalling-Threat-Anomaly-detection-and-Mitigation-Framework
xApp to detect and mitigate Signalling storm threats in ORAN based V2X communications

## Overview:
This repository implements a **deep learning-based anomaly detection framework** for identifying and classifying signaling storms in ORAN based  V2X (Vehicle-to-Everything) networks. The framework detects three types of control-plane attacks that threaten network stability:

| Attack Type | Description |
|------------|-------------|
| **Botnet RRC Storm** | Compromised UEs send multiple  RRC Connection Request | 
| **Group Handover Storm** | Malicious UEs simultaneously trigger HO procedures, overwhelming target cells | 
| **Rogue Base Station** | Fake gNB attracts UEs and forces periodic re-establishment | 

## Key Features

- **Realistic Data Generation**: 3GPP compliant KPI simulation for 5 urban/highway scenarios and validated by generating dataset using ns3 5gLena 
- **LSTM Autoencoder**: Unsupervised learning on normal traffic patterns
- **Multi-class Classification**: Random Forest on reconstruction errors identifies attack types


## Tech Stack

- **Languages**: Python 3.10, C
- **ML Framework**: TensorFlow 2.21 / Keras 3
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **O-RAN Standards**: 3GPP TR 37.885, TS 28.552, TS 28.554, TS 38.331, TS 38.321, TS 38.300
- **xApp**: FlexRIC
- **Policy Engine**: OPA (Open Policy Agent)
- **Network Simulator**: ns-3 with 5G-LENA


## Repository Structure:
- `config.py` — Global parameters (simulation settings, hyperparameters, feature lists)
- `src/` — Source modules for data generation and model architecture
  - `simulators.py` — KPI data generators for normal traffic and 3 attack types
  - `preprocessing.py` — Windowing, normalization, and dataset building functions
  - `model.py` — LSTM autoencoder architecture
  - `utils.py` — Helper functions (validation, correlation plots)
- `scripts/` — Execution scripts to run the complete pipeline
  - `generate_data.py` — Builds and saves normal + attack datasets
  - `train_ae.py` — Trains LSTM autoencoder on normal data
  - `evaluate.py` — Evaluates model, computes metrics, and runs attack classification
- `outputs/` — Generated datasets (.npy, .csv), figures, and evaluation results
- `models/` — Saved trained models (.keras format)
- `examples` — Modified Flexric folder


## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/NGNLab-Projects/STAM_Framework.git
cd STAM_Framework
```

### 2. Install FlexRIC

```bash
cd ~

git clone https://gitlab.eurecom.fr/mosaic5g/flexric
cd flexric

git checkout v1.0.0

sudo apt update
sudo apt install -y build-essential cmake libsctp-dev libpcre2-dev

rm -rf build
mkdir build && cd build

cmake ..
make -j$(nproc)

sudo make install
sudo ldconfig
```

### 3. Install Python Dependencies

```bash
pip install pandas==2.3.3
pip install numpy==2.2.6
pip install tensorflow==2.21.0
pip install keras==3.12.1
pip install scikit-learn==1.7.2
```

### 4. Replace FlexRIC Examples

Replace the default **`examples/`** directory in the FlexRIC source with the modified **`examples/`** directory provided in this repository.

Rebuild FlexRIC:

```bash
cd ~/flexric

rm -rf build
mkdir build && cd build

cmake ..
make -j$(nproc)
```

### 5. Generate Dataset

```bash
python3 generate.py
```

### 6. Train Models

```bash
python3 train_lstm_ae.py
python3 train_rf.py
```

### 7. Start Open Policy Agent

```bash
opa run -s mitigation.rego
```

---

## Execution

Launch each component in a separate terminal.

### Terminal 1 – Near-RT RIC

```bash
cd FlexRIC/build/examples/ric
./nearRT-RIC
```

### Terminal 2 – gNB Agent

```bash
cd FlexRIC/build/examples/emulator/agent
./emu_gnb_agent
```

### Terminal 3 – STAM xApp

```bash
cd FlexRIC/build/examples/xApp/c/monitor
./xapp_stam
```

### Terminal 4 – Python Inference Engine

```bash
cd STAM_Framework
python3 xapp_inference.py
```

### Terminal 5 – Open Policy Agent

```bash
opa run -s mitigation.rego
```
   
