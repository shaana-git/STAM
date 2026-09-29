#ifndef KPM_PROVIDER_BRIDGE_H
#define KPM_PROVIDER_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the Python provider with scenario, attack type, and intensity
// attack_type: NULL or "none" for normal traffic, "botnet" (TM1), "group_ho" (TM2), "rogue_bs" (TM3)
// intensity: 0.0 to 1.0 (maps to mild/moderate/severe)
int init_kpm_provider(const char* scenario_name, const char* attack_type, double intensity);

// Get next KPM measurement (fills buffer with 26 float values)
// Returns 0 on success, -1 on error
int get_next_kpm_measurement(float* buffer, int buffer_size);

// Cleanup
void close_kpm_provider(void);

#ifdef __cplusplus
}
#endif

#endif /* KPM_PROVIDER_BRIDGE_H */