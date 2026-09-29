"""
STAM: Signaling Traffic Anomaly Mitigation for V2X Networks
Complete pipeline: Normal KPM generation -> Attack generation -> LSTM-AE training
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import shutil
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

# -----------------------------------------------------------------------------
# SCENARIO DEFINITIONS (TR 37.885 6.1)
# -----------------------------------------------------------------------------
SCENARIOS = [
    {
        "id": 1,
        "name": "Urban_Low",
        "type": "urban",
        "speed_kmh": 60,
        "mean_ue": 30,
        "max_ue": 42,
        "cell_radius": 289,
        "n_lanes": 8,
        "occupancy": 0.25,
        "tl_period": 45,
        "tl_burst": (6, 10),
        "convoy": False,
    },
    {
        "id": 2,
        "name": "Urban_Medium",
        "type": "urban",
        "speed_kmh": 50,
        "mean_ue": 70,
        "max_ue": 88,
        "cell_radius": 289,
        "n_lanes": 8,
        "occupancy": 0.50,
        "tl_period": 40,
        "tl_burst": (12, 18),
        "convoy": False,
    },
    {
        "id": 3,
        "name": "Urban_High",
        "type": "urban",
        "speed_kmh": 40,
        "mean_ue": 100,
        "max_ue": 122,
        "cell_radius": 289,
        "n_lanes": 8,
        "occupancy": 0.50,
        "tl_period": 35,
        "tl_burst": (18, 26),
        "convoy": False,
    },
    {
        "id": 4,
        "name": "Highway_Low",
        "type": "highway",
        "speed_kmh": 120,
        "mean_ue": 45,
        "max_ue": 58,
        "cell_radius": 289,
        "n_lanes": 6,
        "occupancy": 1.00,
        "tl_period": None,
        "tl_burst": None,
        "convoy": True,
        "convoy_size": (8, 14),
        "convoy_period": 60,
    },
    {
        "id": 5,
        "name": "Highway_High",
        "type": "highway",
        "speed_kmh": 80,
        "mean_ue": 75,
        "max_ue": 92,
        "cell_radius": 289,
        "n_lanes": 6,
        "occupancy": 1.00,
        "tl_period": None,
        "tl_burst": None,
        "convoy": True,
        "convoy_size": (12, 20),
        "convoy_period": 45,
    },
]

# -----------------------------------------------------------------------------
# SIMULATION CONSTANTS
# -----------------------------------------------------------------------------
N_CELLS = 3
RUN_DURATION = 120
L_VEH = 4.5
T_GAP = 2.0
MIN_GAP = 2.0
P_HO_TRIGGER = 0.6
RLF_RATE = 0.005 / 60
TIDAL_PERIOD = 30
TIDAL_AMP = 0.25
ALPHA_RRC_MIN = 1.0
ALPHA_RRC_MAX = 2.0
ALPHA_HO_MIN = 1.5
ALPHA_HO_MAX = 3.0
ALPHA_RE_MIN = 1.0
ALPHA_RE_MAX = 2.5
CQI_MEAN = 10.5
CQI_STD = 1.8
SYNC_WINDOW = 10

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

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def mean_spacing(speed_kmh: float) -> float:
    v_ms = speed_kmh / 3.6
    mean_gap = max(MIN_GAP, v_ms * T_GAP)
    return L_VEH + mean_gap

def compute_arrival_rate(sc: dict) -> float:
    v_ms = sc["speed_kmh"] / 3.6
    road_len = 2 * sc["cell_radius"]
    t_cross = road_len / v_ms
    n_total = (road_len / mean_spacing(sc["speed_kmh"])) * sc["n_lanes"] * sc["occupancy"]
    return n_total / t_cross

def compute_crossing_time(sc: dict) -> float:
    return (2 * sc["cell_radius"] * 3.6) / sc["speed_kmh"]

def drift(t: float) -> float:
    return 1.0 + TIDAL_AMP * np.sin(2 * np.pi * t / TIDAL_PERIOD)

# -----------------------------------------------------------------------------
# PER-CELL STATE
# -----------------------------------------------------------------------------
class CellState:
    def __init__(self, cell_id: int, sc: dict, rng: np.random.Generator):
        self.cell_id = cell_id
        self.sc = sc
        self.rng = rng
        self.active_ue = max(1, int(sc["mean_ue"]))
        self.t_cross = compute_crossing_time(sc)
        self.arr_rate = compute_arrival_rate(sc)
        self.rrc_history = []

    def step(self, t: int) -> dict:
        sc = self.sc
        rng = self.rng
        d = drift(t)

        arrivals = rng.poisson(max(0.01, self.arr_rate * d))
        departures = rng.poisson(max(0.01, self.arr_rate * d))

        self.active_ue = max(1, int(np.clip(
            self.active_ue + arrivals - departures,
            1, sc["max_ue"]
        )))

        timeout_departures = rng.poisson(max(0.01, departures * 0.7 * d))
        ho_departures = departures - timeout_departures

        tl_burst_count = 0
        if sc["type"] == "urban" and sc["tl_period"] is not None:
            if t % sc["tl_period"] == 0:
                tl_burst_count = rng.integers(
                    sc["tl_burst"][0], sc["tl_burst"][1] + 1
                )
                arrivals += tl_burst_count
                self.active_ue = int(np.clip(
                    self.active_ue + tl_burst_count, 1, sc["max_ue"]
                ))

        convoy_ho = 0
        if sc["convoy"] and t % sc["convoy_period"] < 3:
            convoy_size = rng.integers(
                sc["convoy_size"][0], sc["convoy_size"][1] + 1
            )
            convoy_ho = rng.poisson(convoy_size * 0.4)
            convoy_ho = min(convoy_ho, self.active_ue // 2) if self.active_ue > 10 else 0

        rrc_req = int(rng.poisson(max(0.01, arrivals * d)))
        rrc_req = int(np.clip(rrc_req, 0, self.active_ue))

        if rrc_req > 0:
            p_suc = rng.beta(18, 2)
            rrc_suc = rng.binomial(rrc_req, p_suc)
            rrc_setup_ratio = float(rrc_suc) / rrc_req
        else:
            rrc_setup_ratio = 1.0

        ho_base = rng.poisson(max(0.001, departures * P_HO_TRIGGER * d))
        ho_att = int(np.clip(ho_base + convoy_ho, 0, self.active_ue))

        if ho_att > 0:
            p_fail = rng.beta(1, 19)
            ho_fail_count = rng.binomial(ho_att, p_fail)
            ho_fail_ratio = float(ho_fail_count) / ho_att
        else:
            ho_fail_ratio = 0.0
            ho_fail_count = 0

        rrc_reconfig = ho_att + int(rng.poisson(
            max(0.001, self.arr_rate * 0.2 * d)
        ))

        rlf_reestab = int(rng.poisson(
            max(0.0001, RLF_RATE * self.active_ue)
        ))
        ho_cascade = int(rng.binomial(ho_fail_count, 0.7)) if ho_fail_count > 0 else 0
        reestab_req = int(np.clip(rlf_reestab + ho_cascade, 0, self.active_ue))

        rrc_release = int(timeout_departures)
        denominator = rrc_release if rrc_release > 0 else 1.0
        reestab_ratio = float(np.clip(reestab_req / denominator, 0.0, 5.0))

        alpha_rrc = rng.uniform(ALPHA_RRC_MIN, ALPHA_RRC_MAX)
        alpha_ho = rng.uniform(ALPHA_HO_MIN, ALPHA_HO_MAX)
        alpha_re = rng.uniform(ALPHA_RE_MIN, ALPHA_RE_MAX)
        prach_att = int(np.clip(
            round(rrc_req * alpha_rrc + ho_att * alpha_ho + reestab_req * alpha_re),
            0, 500
        ))

        if prach_att > 0:
            p_prach_fail = rng.beta(1, 15)
            prach_fail = float(np.clip(p_prach_fail, 0.0, 1.0))
        else:
            prach_fail = 0.0

        load_factor = rng.uniform(0.85, 1.15)
        prb_ul = float(np.clip(
            (self.active_ue * 0.003 + rrc_req * 0.008) * load_factor,
            0.02, 0.75
        ))

        avg_cqi = float(np.clip(rng.normal(CQI_MEAN, CQI_STD), 1.0, 15.0))

        sig_util = float(np.clip(
            (self.active_ue / sc["max_ue"]) * 0.4 +
            prb_ul * 0.3 +
            (rrc_req / max(sc["mean_ue"] * 0.1, 1)) * 0.3,
            0.0, 1.0
        ))

        self.rrc_history.append(rrc_req)
        if len(self.rrc_history) > SYNC_WINDOW:
            self.rrc_history.pop(0)
        mean_rrc = np.mean(self.rrc_history) if self.rrc_history else 1.0
        burst_intensity = float(rrc_req / max(mean_rrc, 0.1))

        if len(self.rrc_history) >= 3:
            rolling_max = max(self.rrc_history[-SYNC_WINDOW:])
            rolling_mean = max(np.mean(self.rrc_history[-SYNC_WINDOW:]), 0.1)
            sync_score = float(np.clip(
                (rolling_max / rolling_mean - 1.0) / 5.0, 0.0, 1.0
            ))
        else:
            sync_score = 0.0

        return {
            "timestamp": float(t),
            "cell_id": self.cell_id,
            "rrc_conn_req_count": rrc_req,
            "rrc_setup_success_ratio": round(rrc_setup_ratio, 4),
            "prach_attempt_count": prach_att,
            "prach_failure_rate": round(prach_fail, 4),
            "ho_attempt_count": ho_att,
            "ho_failure_ratio": round(ho_fail_ratio, 4),
            "rrc_reconfig_count": rrc_reconfig,
            "rrc_conn_reestab_req_count": reestab_req,
            "reestab_to_release_ratio": round(reestab_ratio, 4),
            "active_ue_count": self.active_ue,
            "prb_utilization_ul": round(prb_ul, 4),
            "avg_cqi": round(avg_cqi, 2),
            "signaling_resource_util": round(sig_util, 4),
            "burst_intensity": round(burst_intensity, 4),
            "access_synchrony_score": round(sync_score, 4),
        }

# -----------------------------------------------------------------------------
# NORMAL KPM SIMULATOR
# -----------------------------------------------------------------------------
class NormalKPMSimulator:
    def __init__(self, scenario: dict, run_id: int, seed: int = None):
        self.sc = scenario
        self.run_id = run_id
        self.seed = seed if seed is not None else (run_id * 137 + scenario["id"] * 1000)
        self.rng = np.random.default_rng(self.seed)

    def simulate(self) -> pd.DataFrame:
        cells = [
            CellState(cell_id=c, sc=self.sc, rng=np.random.default_rng(self.seed + c))
            for c in range(N_CELLS)
        ]
        rows = []
        for t in range(RUN_DURATION):
            for cell in cells:
                row = cell.step(t)
                row["run_id"] = self.run_id
                row["scenario_id"] = self.sc["id"]
                row["scenario_name"] = self.sc["name"]
                row["label"] = 0
                row["split"] = "train"
                rows.append(row)
        df = pd.DataFrame(rows)
        meta_cols = ["timestamp","cell_id","run_id","scenario_id","scenario_name","label","split"]
        return df[meta_cols + FEATURE_COLS]

# -----------------------------------------------------------------------------
# ATTACK PARAMETERS (from paper)
# -----------------------------------------------------------------------------
SEED_BASE_ATK = 9999
ATTACK_ONSET = 30
RUNS_PER_CONFIG = 7
LABEL_NORMAL = 0
LABEL_TM1 = 1
LABEL_TM2 = 2
LABEL_TM3 = 3

TM1_PARAMS = {
    "mild":     {"flood_factor": 1.5, "botnet_frac": 0.05, "success_degrade": 0.10},
    "moderate": {"flood_factor": 3.0, "botnet_frac": 0.15, "success_degrade": 0.30},
    "severe":   {"flood_factor": 6.0, "botnet_frac": 0.30, "success_degrade": 0.55},
}

TM2_PARAMS = {
    "mild":     {"ho_multiplier": 3, "forced_fail_rate": 0.20, "cascade_prob": 0.40},
    "moderate": {"ho_multiplier": 8, "forced_fail_rate": 0.40, "cascade_prob": 0.65},
    "severe":   {"ho_multiplier": 15, "forced_fail_rate": 0.65, "cascade_prob": 0.80},
}

TM3_PARAMS = {
    "mild":     {"capture_frac": 0.05, "cycle_period": 25, "cqi_degrade": 0.8},
    "moderate": {"capture_frac": 0.15, "cycle_period": 15, "cqi_degrade": 1.5},
    "severe":   {"capture_frac": 0.60, "cycle_period": 8, "cqi_degrade": 2.5},
}

# -----------------------------------------------------------------------------
# ATTACK CELL STATE
# -----------------------------------------------------------------------------
class AttackCellState(CellState):
    def __init__(self, cell_id, sc, rng, attack_tm, intensity):
        super().__init__(cell_id, sc, rng)
        self.attack_tm = attack_tm
        self.intensity = intensity
        self.params = {
            LABEL_TM1: TM1_PARAMS[intensity],
            LABEL_TM2: TM2_PARAMS[intensity],
            LABEL_TM3: TM3_PARAMS[intensity],
        }[attack_tm]

    def step(self, t):
        row = super().step(t)
        if t < ATTACK_ONSET:
            row["label"] = LABEL_NORMAL
            row["attack_tm"] = 0
            row["attack_intensity"] = "none"
            return row
        if self.attack_tm == LABEL_TM1:
            row = self._inject_tm1(row)
        elif self.attack_tm == LABEL_TM2:
            row = self._inject_tm2(row)
        elif self.attack_tm == LABEL_TM3:
            row = self._inject_tm3(row, t)
        row["label"] = self.attack_tm
        row["attack_tm"] = self.attack_tm
        row["attack_intensity"] = self.intensity
        return row

    def _inject_tm1(self, row):
        p = self.params
        rng = self.rng
        ff = p["flood_factor"]
        bf = p["botnet_frac"]
        sd = p["success_degrade"]
        n_ue = max(1, row["active_ue_count"])
        n_bots = int(np.clip(rng.poisson(max(1, n_ue * bf)), 1, n_ue))
        flood_rrc = int(rng.poisson(max(1, n_bots * ff)))
        attack_rrc = int(np.clip(row["rrc_conn_req_count"] + flood_rrc, 0, 500))
        row["rrc_conn_req_count"] = attack_rrc
        row["rrc_setup_success_ratio"] = round(float(np.clip(
            row["rrc_setup_success_ratio"] * (1.0 - sd) + rng.uniform(-0.05, 0.05),
            0.0, 1.0
        )), 4)
        alpha_rrc = rng.uniform(ALPHA_RRC_MIN, ALPHA_RRC_MAX)
        flood_prach = int(rng.poisson(max(1, flood_rrc * alpha_rrc)))
        attack_prach = int(np.clip(row["prach_attempt_count"] + flood_prach, 0, 2000))
        row["prach_attempt_count"] = attack_prach
        contention = min(attack_prach / 500.0, 1.0)
        a = max(0.5, 1.0 + 8.0 * contention)
        b = max(0.5, 16.0 - 8.0 * contention)
        row["prach_failure_rate"] = round(float(np.clip(rng.beta(a, b), 0.0, 1.0)), 4)
        row["prb_utilization_ul"] = round(float(np.clip(
            row["prb_utilization_ul"] + min(flood_rrc * 0.001, 0.20), 0.02, 0.98
        )), 4)
        sig_overhead = min(flood_rrc / max(n_ue * 0.5, 1), 0.50)
        row["signaling_resource_util"] = round(float(np.clip(
            row["signaling_resource_util"] + sig_overhead, 0.0, 1.0
        )), 4)
        self.rrc_history.append(attack_rrc)
        if len(self.rrc_history) > SYNC_WINDOW:
            self.rrc_history.pop(0)
        mean_rrc = max(np.mean(self.rrc_history), 0.1)
        row["burst_intensity"] = round(float(attack_rrc / mean_rrc), 4)
        coordination = rng.uniform(0.3, 0.7) * (ff / 10.0)
        row["access_synchrony_score"] = round(float(np.clip(
            row["access_synchrony_score"] + coordination, 0.0, 1.0
        )), 4)
        return row

    def _inject_tm2(self, row):
        p = self.params
        rng = self.rng
        hm = p["ho_multiplier"]
        ffr = p["forced_fail_rate"]
        cp = p["cascade_prob"]
        n_ue = max(1, row["active_ue_count"])

        max_convoy = min(int(hm * 4), int(n_ue * 0.8))
        forced_ho = int(np.clip(rng.poisson(max(1, max_convoy)), 0, n_ue))
        attack_ho = int(np.clip(
            row["ho_attempt_count"] + forced_ho, 0, min(n_ue, 80)
        ))
        row["ho_attempt_count"] = attack_ho

        a_fail = max(0.5, ffr * 10.0)
        b_fail = max(0.5, (1.0 - ffr) * 10.0)
        p_fail = float(np.clip(rng.beta(a_fail, b_fail), 0.0, 1.0))
        ho_fail_count = int(rng.binomial(attack_ho, p_fail))
        row["ho_failure_ratio"] = round(
            float(ho_fail_count / attack_ho) if attack_ho > 0 else 0.0, 4
        )

        row["rrc_reconfig_count"] = int(np.clip(
            attack_ho + int(rng.poisson(max(0.1, attack_ho * 0.1))),
            0, min(n_ue * 2, 160)
        ))

        reestab_from_ho = int(rng.binomial(ho_fail_count, cp))
        rlf_background = int(rng.poisson(max(0.001, RLF_RATE * n_ue)))
        attack_reestab = int(np.clip(reestab_from_ho + rlf_background, 0, n_ue))
        row["rrc_conn_reestab_req_count"] = attack_reestab

        rrc_release = max(1, int(rng.poisson(max(0.01, n_ue * 0.05))))
        row["reestab_to_release_ratio"] = round(float(np.clip(
            attack_reestab / rrc_release, 0.0, 5.0
        )), 4)

        alpha_ho = rng.uniform(ALPHA_HO_MIN, ALPHA_HO_MAX)
        alpha_re = rng.uniform(ALPHA_RE_MIN, ALPHA_RE_MAX)
        ho_prach = int(rng.poisson(max(1, attack_ho * alpha_ho + attack_reestab * alpha_re)))
        row["prach_attempt_count"] = int(np.clip(
            row["prach_attempt_count"] + ho_prach, 0, 500
        ))
        row["prach_failure_rate"] = round(float(np.clip(
            row["prach_failure_rate"] + rng.uniform(0.05, 0.20), 0.0, 1.0
        )), 4)
        row["prb_utilization_ul"] = round(float(np.clip(
            row["prb_utilization_ul"] + min(attack_ho * 0.002, 0.30), 0.02, 0.98
        )), 4)
        row["avg_cqi"] = round(float(np.clip(
            row["avg_cqi"] - rng.uniform(0.5, 1.5), 1.0, 15.0
        )), 2)
        row["signaling_resource_util"] = round(float(np.clip(
            row["signaling_resource_util"] + min(attack_ho * 0.003, 0.40), 0.0, 1.0
        )), 4)
        self.rrc_history.append(row["rrc_conn_req_count"])
        if len(self.rrc_history) > SYNC_WINDOW:
            self.rrc_history.pop(0)
        mean_rrc = max(np.mean(self.rrc_history), 0.1)
        row["burst_intensity"] = round(float(row["rrc_conn_req_count"] / mean_rrc), 4)
        return row

    def _inject_tm3(self, row, t):
        p = self.params
        rng = self.rng
        cf = p["capture_frac"]
        T = p["cycle_period"]
        cqi_deg = p["cqi_degrade"]
        n_ue = max(1, row["active_ue_count"])
        n_captured = int(np.clip(rng.poisson(max(1, n_ue * cf)), 1, n_ue))
        t_atk = t - ATTACK_ONSET
        phase = t_atk % T if T > 0 else 0
        capture_end = max(1, int(T * 0.2))
        reattach_start = max(capture_end + 1, int(T * 0.8))

        if phase < capture_end:
            fbs_ho = int(rng.poisson(max(1, n_captured * 0.8)))
            row["ho_attempt_count"] = int(np.clip(
                row["ho_attempt_count"] + fbs_ho, 0, n_ue * 2
            ))
            row["ho_failure_ratio"] = round(float(np.clip(rng.beta(8, 2), 0.0, 1.0)), 4)
            row["active_ue_count"] = max(1, n_ue - n_captured)
        elif phase < reattach_start:
            reestab_wave = int(rng.poisson(max(1, n_captured * 0.9)))
            row["rrc_conn_reestab_req_count"] = int(np.clip(
                row["rrc_conn_reestab_req_count"] + reestab_wave, 0, n_ue
            ))
            rrc_release = max(1, int(rng.poisson(max(0.01, n_ue * 0.03))))
            row["reestab_to_release_ratio"] = round(float(np.clip(
                row["rrc_conn_reestab_req_count"] / rrc_release, 0.0, 5.0
            )), 4)
            alpha_re = rng.uniform(ALPHA_RE_MIN, ALPHA_RE_MAX)
            reestab_prach = int(rng.poisson(max(1, reestab_wave * alpha_re)))
            row["prach_attempt_count"] = int(np.clip(
                row["prach_attempt_count"] + reestab_prach, 0, 1000
            ))
            row["prach_failure_rate"] = round(float(np.clip(
                row["prach_failure_rate"] + rng.uniform(0.03, 0.15), 0.0, 1.0
            )), 4)
        else:
            reattach_rrc = int(rng.poisson(max(1, n_captured * 0.95)))
            row["rrc_conn_req_count"] = int(np.clip(
                row["rrc_conn_req_count"] + reattach_rrc, 0, 300
            ))
            row["rrc_setup_success_ratio"] = round(float(np.clip(
                row["rrc_setup_success_ratio"] * rng.uniform(0.6, 0.85), 0.0, 1.0
            )), 4)
            row["active_ue_count"] = n_ue

        row["avg_cqi"] = round(float(np.clip(
            row["avg_cqi"] - cqi_deg + rng.normal(0.0, 0.3), 1.0, 15.0
        )), 2)
        row["signaling_resource_util"] = round(float(np.clip(
            row["signaling_resource_util"] + cf * 0.3, 0.0, 1.0
        )), 4)
        self.rrc_history.append(row["rrc_conn_req_count"])
        if len(self.rrc_history) > SYNC_WINDOW:
            self.rrc_history.pop(0)
        mean_rrc = max(np.mean(self.rrc_history), 0.1)
        row["burst_intensity"] = round(float(row["rrc_conn_req_count"] / mean_rrc), 4)
        if len(self.rrc_history) >= 3:
            rolling_max = max(self.rrc_history[-SYNC_WINDOW:])
            rolling_mean = max(np.mean(self.rrc_history[-SYNC_WINDOW:]), 0.1)
            row["access_synchrony_score"] = round(float(np.clip(
                (rolling_max / rolling_mean - 1.0) / 5.0, 0.0, 1.0
            )), 4)
        return row

# -----------------------------------------------------------------------------
# ATTACK KPM SIMULATOR
# -----------------------------------------------------------------------------
class AttackKPMSimulator:
    def __init__(self, scenario, run_id, attack_tm, intensity, seed=None):
        self.sc = scenario
        self.run_id = run_id
        self.attack_tm = attack_tm
        self.intensity = intensity
        self.seed = seed if seed is not None else (
            run_id * 137 + scenario["id"] * 1000 + attack_tm * 100 + SEED_BASE_ATK
        )

    def simulate(self):
        cells = [
            AttackCellState(
                cell_id=c, sc=self.sc,
                rng=np.random.default_rng(self.seed + c),
                attack_tm=self.attack_tm, intensity=self.intensity,
            )
            for c in range(N_CELLS)
        ]
        rows = []
        for t in range(RUN_DURATION):
            for cell in cells:
                row = cell.step(t)
                row["run_id"] = self.run_id
                row["scenario_id"] = self.sc["id"]
                row["scenario_name"] = self.sc["name"]
                row["split"] = "test"
                rows.append(row)
        df = pd.DataFrame(rows)
        meta_cols = [
            "timestamp","cell_id","run_id","scenario_id",
            "scenario_name","label","attack_tm","attack_intensity","split",
        ]
        return df[meta_cols + FEATURE_COLS]

# -----------------------------------------------------------------------------
# DATASET BUILDERS
# -----------------------------------------------------------------------------
def build_normal_dataset(
    runs_per_scenario: int = 11,
    output_dir: str = "outputs/normal",
    val_ratio: float = 0.20,
    seed_base: int = 42,
) -> pd.DataFrame:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_dfs = []
    run_id = 0

    print(f"Building normal dataset...")
    print(f" Scenarios: {len(SCENARIOS)}")
    print(f" Runs/scenario: {runs_per_scenario}")

    for sc in SCENARIOS:
        sc_rows = 0
        for r in range(runs_per_scenario):
            seed = seed_base + run_id * 137 + sc["id"] * 1000
            sim = NormalKPMSimulator(scenario=sc, run_id=run_id, seed=seed)
            df = sim.simulate()
            csv_path = os.path.join(output_dir, f"run_{run_id:04d}_{sc['name']}.csv")
            df.to_csv(csv_path, index=False)
            all_dfs.append(df)
            sc_rows += len(df)
            run_id += 1
        print(f" Done: {sc['name']:20s} | {runs_per_scenario} runs | {sc_rows:,} rows")

    df_all = pd.concat(all_dfs, ignore_index=True)

    run_ids = df_all["run_id"].unique()
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed_base)
    train_idx, val_idx = next(gss.split(df_all, groups=df_all["run_id"]))
    val_run_ids = df_all.iloc[val_idx]["run_id"].unique()
    df_all.loc[df_all["run_id"].isin(val_run_ids), "split"] = "val"

    combined_path = os.path.join(output_dir, "normal_dataset.csv")
    df_all.to_csv(combined_path, index=False)
    print(f" Dataset saved: {combined_path}")
    return df_all

def build_attack_dataset(runs_per_config=7,
                          output_dir="outputs/attack",
                          seed_base=9999):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tm_names = {LABEL_TM1:"TM1_Flood", LABEL_TM2:"TM2_HOStorm", LABEL_TM3:"TM3_FBS"}

    print(f"Building attack dataset...")
    all_dfs = []
    run_id = 0

    for attack_tm in [LABEL_TM1, LABEL_TM2, LABEL_TM3]:
        for intensity in ["mild", "moderate", "severe"]:
            for sc in SCENARIOS:
                sc_rows = 0
                for r in range(runs_per_config):
                    seed = seed_base + run_id*137 + sc["id"]*1000 + attack_tm*100
                    sim = AttackKPMSimulator(sc, run_id, attack_tm, intensity, seed)
                    df = sim.simulate()
                    fname = (f"run_{run_id:04d}_{tm_names[attack_tm]}"
                             f"_{intensity}_{sc['name']}.csv")
                    df.to_csv(os.path.join(output_dir, fname), index=False)
                    all_dfs.append(df)
                    sc_rows += len(df)
                    run_id += 1
                print(f" Done: {tm_names[attack_tm]:<14} {intensity:<10} {sc['name']:<18} {sc_rows:,} rows")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(output_dir, "attack_dataset.csv"), index=False)
    print(f" Total rows: {len(df_all):,}")
    return df_all
