#!/usr/bin/env python3
"""
KPM Data Provider for FlexRIC E2 Agent
UPDATED to use the new STAM simulator (IEEE Journal Version)
Supports TM1 (flood), TM2 (HO storm), TM3 (fake BS) attacks
"""

import sys
import os
import numpy as np

# Add paths to your STAM code
sys.path.append('/home/saahna_arisu/STAM')
sys.path.append('/home/saahna_arisu/STAM/src')
sys.path.append('/home/saahna_arisu/STAM/flexric_integration')

# Import what's available at module level
from simulator import NormalKPMSimulator, SCENARIOS, RUN_DURATION, N_CELLS
from simulator import build_attack_dataset  # We'll use this differently

class KPMDataProvider:
    def __init__(self):
        self.simulator = None
        self.current_run_id = 0
        self.cell_id = 1
        self.all_data = None
        self.current_index = 0
        self.total_timesteps = 0
        self.attack_tm = None
        self.intensity_str = "mild"
        
    def init_simulator(self, scenario_name="Urban_Low", attack_type="none", intensity="mild"):
        """Initialize the simulator with given scenario and attack type"""
        self.attack_tm = 0
        if attack_type == "tm1":
            self.attack_tm = 1
        elif attack_type == "tm2":
            self.attack_tm = 2
        elif attack_type == "tm3":
            self.attack_tm = 3
        
        self.intensity_str = intensity
        
        # Find scenario config
        scenario = None
        for sc in SCENARIOS:
            if sc['name'] == scenario_name:
                scenario = sc
                break
        if scenario is None:
            scenario = SCENARIOS[0]
        
        self.current_run_id += 1
        
        # Create simulator based on attack type
        if self.attack_tm == 0:
            # Normal traffic - use NormalKPMSimulator directly
            self.simulator = NormalKPMSimulator(
                scenario=scenario,
                run_id=self.current_run_id,
                seed=42 + self.current_run_id
            )
            df = self.simulator.simulate()
        else:
            # For attacks, we need to use the attack dataset generator
            # Create temporary attack data
            import tempfile
            import pandas as pd
            
            temp_dir = tempfile.mkdtemp()
            build_attack_dataset(
                runs_per_config=1,
                output_dir=temp_dir,
                seed_base=42 + self.current_run_id + self.attack_tm * 100
            )
            
            # Read the generated attack data
            attack_file = os.path.join(temp_dir, 'attack_dataset.csv')
            if os.path.exists(attack_file):
                df = pd.read_csv(attack_file)
                # Filter for this specific attack type and intensity
                df = df[df['label'] == self.attack_tm]
                df = df[df['attack_intensity'] == intensity]
            else:
                # Fallback: look for individual files
                csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                if csv_files:
                    df = pd.read_csv(os.path.join(temp_dir, csv_files[0]))
                else:
                    print(f"❌ No attack data generated for {attack_type}/{intensity}")
                    return -1
        
        # Extract data for one cell
        cell_data = df[df['cell_id'] == self.cell_id].sort_values('timestamp')
        
        # Get feature columns (15 KPI features)
        feature_cols = [
            "rrc_conn_req_count", "rrc_setup_success_ratio",
            "prach_attempt_count", "prach_failure_rate",
            "ho_attempt_count", "ho_failure_ratio", "rrc_reconfig_count",
            "rrc_conn_reestab_req_count", "reestab_to_release_ratio",
            "active_ue_count", "prb_utilization_ul", "avg_cqi",
            "signaling_resource_util", "burst_intensity", "access_synchrony_score",
        ]
        
        self.all_data = cell_data[feature_cols].values
        self.total_timesteps = len(self.all_data)
        self.current_index = 0
        
        print(f"✅ Simulator ready: {scenario_name}, attack={attack_type}, intensity={intensity}, steps={self.total_timesteps}")
        return 0
        
    def get_next_kpm(self):
        """Return next KPM measurement (15 features padded to 26 for compatibility)"""
        if self.current_index >= self.total_timesteps:
            self.current_index = 0
        
        # Get 15 features from simulator
        features_15 = self.all_data[self.current_index]
        self.current_index += 1
        
        # Pad to 26 values
        kpm_vector = np.zeros(26, dtype=np.float32)
        kpm_vector[:15] = features_15
        
        # Add derived features for remaining slots
        if self.current_index > 3:
            recent = self.all_data[max(0, self.current_index-5):self.current_index]
            if len(recent) > 0:
                kpm_vector[15] = np.mean(recent[:, 0])
                kpm_vector[16] = np.std(recent[:, 0])
                kpm_vector[17] = np.mean(recent[:, 4])
                kpm_vector[18] = np.std(recent[:, 4])
                kpm_vector[19] = np.mean(recent[:, 10])
                kpm_vector[20] = np.mean(recent[:, 13])
                kpm_vector[21] = np.std(recent[:, 13])
        
        return kpm_vector.tolist()
    
    def get_kpm_batch(self, count=10):
        """Return batch of KPM measurements"""
        batch = []
        for _ in range(count):
            batch.append(self.get_next_kpm())
        return batch
    
    def close(self):
        """Cleanup"""
        pass

# Global instance
_provider = None

def init_provider(scenario_name, attack_type="none", intensity="mild"):
    """Initialize the provider (called from C)"""
    global _provider
    _provider = KPMDataProvider()
    return _provider.init_simulator(scenario_name, attack_type, intensity)

def get_kpm_vector():
    """Get next KPM vector (called from C)"""
    if _provider is None:
        return [0.0] * 26
    return _provider.get_next_kpm()

def close_provider():
    """Cleanup (called from C)"""
    global _provider
    if _provider is not None:
        _provider.close()
        _provider = None

# For testing
if __name__ == "__main__":
    # Test normal traffic
    print("Testing normal traffic...")
    init_provider("Urban_Low", "none", "mild")
    for i in range(5):
        kpm = get_kpm_vector()
        print(f"  Step {i}: first 5 values = {[round(x,2) for x in kpm[:5]]}")
    close_provider()
    
    # Test TM1 attack
    print("\nTesting TM1 attack...")
    init_provider("Urban_Low", "tm1", "severe")
    for i in range(5):
        kpm = get_kpm_vector()
        print(f"  Step {i}: RRC={kpm[0]:.1f}, HO={kpm[4]:.1f}")
    close_provider()