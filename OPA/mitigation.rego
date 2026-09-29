package oran.mitigation

# =============================================================================
# STAM Mitigation Policy - Proposed Work Implementation
# 
# Implements:
# - Equation 2: Severity Score S(t)
# - Equation 3: Asymmetric Rate Suppression
# - Equation 4: PRACH Backoff Duration
# - Equation 5: T300 Timer Extension
# - Equation 6: RRC Admission Cap
# - Equation 9: Recovery Condition
# =============================================================================

# ---------------------------------------------------------------------------
# DEFAULT OUTPUTS
# ---------------------------------------------------------------------------
default should_apply_prach_backoff := false
default should_extend_t300         := false
default admission_limit            := 6
default t_backoff_ms               := 0
default t300_ms                    := 600
default ttt_mit_ms                 := 40
default mitigation_factor          := 1.0
default recovered                  := false

# ============================================================================
# COMMON CALCULATIONS
# ============================================================================

# Normalized severity (Equation 2 concept)
S_norm := input.severity / input.s_max
S_norm_capped := min([S_norm, 1.0])

# ============================================================================
# THREAT MODEL 1 — Botnet RRC Connection Storm (TM1)
# ============================================================================

# --- PRACH Backoff trigger ---
should_apply_prach_backoff if {
    input.attack_type == "botnet"
    input.severity > 0
}

# --- PRACH Backoff duration (Equation 4) ---
# t_backoff(t) = 960 × min(S(t)/S_max, 1) ms
t_backoff_ms := tb if {
    input.attack_type == "botnet"
    tb := round(960 * S_norm_capped)
}

# --- T300 Timer Extension (Equation 5) ---
should_extend_t300 if {
    input.attack_type == "botnet"
    input.severity > 0
}

t300_ms := 2000 if {
    input.attack_type == "botnet"
    input.severity > 5.0
}

t300_ms := 1000 if {
    input.attack_type == "botnet"
    input.severity > 2.0
    input.severity <= 5.0
}

t300_ms := 600 if {
    input.attack_type == "botnet"
    input.severity <= 2.0
}

# --- RRC Admission Cap (Equation 6) ---
# N_cap(t) = max(1, floor(6.3 × (1 − S/S_max × 0.8)))
admission_limit := cap if {
    input.attack_type == "botnet"
    raw := 6.3 * (1 - (S_norm_capped * 0.8))
    cap := max([1, floor(raw)])
}

# ============================================================================
# THREAT MODEL 2 — Rogue Base Station / Fake BS (TM2)
# ============================================================================

# TM2 Detection: Trigger mitigation when confirmed
should_apply_prach_backoff if {
    input.attack_type == "rogue_bs"
    input.severity > 0.3
}

# For rogue BS, admission limit drops to 0 (block)
admission_limit := 0 if {
    input.attack_type == "rogue_bs"
    input.severity > 0.5
}

admission_limit := 1 if {
    input.attack_type == "rogue_bs"
    input.severity > 0.3
    input.severity <= 0.5
}

# ============================================================================
# THREAT MODEL 3 — Handover Storm (TM3)
# ============================================================================

# TM3: Handover storm mitigation
should_apply_prach_backoff if {
    input.attack_type == "handover_storm"
    input.severity > 0
}

# Admission cap based on mitigation factor
mitigation_factor := mf if {
    input.attack_type == "handover_storm"
    mf := 1.0 / (1.0 + (input.severity / input.s_max) * 5)
}

admission_limit := cap if {
    input.attack_type == "handover_storm"
    cap := max([1, floor(6.3 * mitigation_factor)])
}

# Extended TTT to reduce handovers
ttt_mit_ms := ttt if {
    input.attack_type == "handover_storm"
    ttt := 40 + round(600 * S_norm_capped)
}

should_extend_t300 if {
    input.attack_type == "handover_storm"
    input.severity > 0.5
}

# ============================================================================
# UNKNOWN ATTACK — Conservative Fallback
# ============================================================================

should_apply_prach_backoff if {
    input.attack_type == "unknown"
    input.severity > 0.4
}

should_extend_t300 if {
    input.attack_type == "unknown"
    input.severity > 0.5
}

admission_limit := cap if {
    input.attack_type == "unknown"
    raw := 6.3 * (1 - S_norm_capped * 0.6)
    cap := max([1, floor(raw)])
}

t_backoff_ms := tb if {
    input.attack_type == "unknown"
    input.severity > 0.4
    tb := round(480 * S_norm_capped)
}

# ============================================================================
# RECOVERY GATE (Equation 9)
# ============================================================================

recovered if {
    count({k | input.recent_errors[k] >= input.threshold}) == 0
}
