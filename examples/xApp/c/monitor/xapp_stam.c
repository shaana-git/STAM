/*
 * STAM xApp
 */

#include <sys/wait.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "../../../../src/xApp/e42_xapp_api.h"
#include "../../../../src/util/alg_ds/alg/defer.h"
#include "../../../../src/util/time_now_us.h"
#include "../../../../src/sm/kpm_sm_v2.02/kpm_sm_id.h"

/* Parameters */
#define THRESHOLD_SUCCESS_RATIO  0.7f
#define THRESHOLD_TM1_RRC_MIN    30.0f
#define THRESHOLD_TM1_PRACH_MIN  30.0f
#define THRESHOLD_BURST          2.5f

#define LSTM_THRESHOLD           1.398223f
#define MEAN_RECON_ERROR         0.188f
#define STD_RECON_ERROR          0.025f

#define TM3_WINDOW_SECONDS       30
#define TM3_MIN_REESTAB          3
#define THRESHOLD_CQI_DEGRADE    8.0f

#define LAMBDA_BOT               0.02f
#define LAMBDA_LEGIT             0.0025f
#define SEVERITY_MAX             10.0f
#define NORMAL_RRC_MEAN          6.3f
#define ADMISSION_CLAMP          0.8f
#define RECOVERY_WINDOWS         5

#define T300_NORMAL              400
#define T300_LEVEL1              600
#define T300_LEVEL2              1000
#define T300_LEVEL3              2000

#define TTT_MIN_MS               40
#define TTT_MAX_MS               640

#define KPM_CSV_PATH             "/tmp/kpm_values.csv"
#define ANOMALY_CSV_PATH         "/tmp/anomaly_results.csv"

/* Feature indices */
#define IDX_RRC_REQ              0
#define IDX_RRC_SUCCESS          1
#define IDX_PRACH_ATT            2
#define IDX_PRACH_FAIL           3
#define IDX_HO_ATT               4
#define IDX_HO_FAIL              5
#define IDX_RRC_RECONFIG         6
#define IDX_RRC_REESTAB          7
#define IDX_AVG_CQI              11
#define IDX_BURST_INTENSITY      13

typedef struct {
    double timestamp;
    float reestab_count;
    float avg_cqi;
} tm3_buffer_entry_t;

static tm3_buffer_entry_t tm3_buffer[100];
static int tm3_buffer_count = 0;
static long anomaly_last_pos = 0;
static volatile int keep_running = 1;

void int_handler(int sig) {
    printf("\n[STAM] Shutting down...\n");
    keep_running = 0;
}

static float calculate_severity(float mse) {
    float severity = (mse - MEAN_RECON_ERROR) / STD_RECON_ERROR;
    if (severity < 0) severity = 0;
    if (severity > SEVERITY_MAX) severity = SEVERITY_MAX;
    return severity;
}

static float calculate_backoff_duration(float severity) {
    float normalized = severity / SEVERITY_MAX;
    if (normalized > 1.0f) normalized = 1.0f;
    return 960.0f * normalized;
}

static int get_t300_value(float severity) {
    if (severity > 5.0f) return T300_LEVEL3;
    else if (severity > 2.0f) return T300_LEVEL2;
    return T300_LEVEL1;
}

static int calculate_admission_cap(float severity) {
    float normalized = severity / SEVERITY_MAX;
    if (normalized > 1.0f) normalized = 1.0f;
    int cap = (int)(NORMAL_RRC_MEAN * (1.0f - normalized * ADMISSION_CLAMP));
    if (cap < 1) cap = 1;
    return cap;
}

static float asymmetric_rate_suppression(float original_rate, float backoff_ms, int is_rogue) {
    float lambda = is_rogue ? LAMBDA_BOT : LAMBDA_LEGIT;
    return original_rate / (1.0f + lambda * backoff_ms);
}

static int calculate_ttt_mitigation(float severity) {
    float normalized = severity / SEVERITY_MAX;
    if (normalized > 1.0f) normalized = 1.0f;
    return TTT_MIN_MS + (int)((TTT_MAX_MS - TTT_MIN_MS) * normalized);
}

static int fast_path_detector_tm1(float rrc_req, float prach_att, float burst_intensity,
                                   float rrc_success_ratio, int callback_count) {
    if (rrc_success_ratio >= THRESHOLD_SUCCESS_RATIO) return 0;
    
    int is_tm1_flood = (rrc_req > THRESHOLD_TM1_RRC_MIN &&
                        prach_att > THRESHOLD_TM1_PRACH_MIN &&
                        burst_intensity > THRESHOLD_BURST);
    
    if (is_tm1_flood) {
        printf("\n[FAST PATH][KPM #%d]: TM1 DETECTED\n", callback_count);
        printf("   rrc_req=%.1f, prach=%.1f, burst=%.2f, succ=%.2f\n",
               rrc_req, prach_att, burst_intensity, rrc_success_ratio);
        return 1;
    }
    return 0;
}

static int tm3_confirmer(double timestamp, float reestab_count, float avg_cqi) {
    int active_count = 0;
    float sum_cqi = 0.0f;
    int cqi_count = 0;
    
    if (tm3_buffer_count < 100) {
        tm3_buffer[tm3_buffer_count].timestamp = timestamp;
        tm3_buffer[tm3_buffer_count].reestab_count = reestab_count;
        tm3_buffer[tm3_buffer_count].avg_cqi = avg_cqi;
        tm3_buffer_count++;
    } else {
        for (int i = 0; i < 99; i++) {
            tm3_buffer[i] = tm3_buffer[i+1];
        }
        tm3_buffer[99].timestamp = timestamp;
        tm3_buffer[99].reestab_count = reestab_count;
        tm3_buffer[99].avg_cqi = avg_cqi;
    }
    
    for (int i = 0; i < tm3_buffer_count; i++) {
        if (timestamp - tm3_buffer[i].timestamp <= TM3_WINDOW_SECONDS) {
            if (tm3_buffer[i].reestab_count > 0) active_count++;
            if (tm3_buffer[i].avg_cqi > 0) {
                sum_cqi += tm3_buffer[i].avg_cqi;
                cqi_count++;
            }
        }
    }
    
    float mean_cqi = (cqi_count > 0) ? (sum_cqi / cqi_count) : 15.0f;
    
    if (active_count >= TM3_MIN_REESTAB && mean_cqi < THRESHOLD_CQI_DEGRADE) {
        printf("\n[TM3 CONFIRMER] HO Storm VERIFIED\n");
        printf("   Reestablishment: %d events in %ds, Mean CQI: %.1f\n", 
               active_count, TM3_WINDOW_SECONDS, mean_cqi);
        return 1;
    }
    return 0;
}

void query_opa_and_mitigate(const char* attack_type, float severity, float duration,
                            float rrc_rate, float threshold, 
                            float prach_failure, float ho_rate) {
    
    printf("\n   [OPA QUERY] Attack: %s, Severity: %.3f\n", attack_type, severity);
    
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "curl -s -X POST http://localhost:8181/v1/data/oran/mitigation "
        "-H 'Content-Type: application/json' "
        "-d '{\"input\": {\"attack_type\": \"%s\", \"severity\": %f, "
        "\"s_max\": %f, \"duration_seconds\": %f, \"rrc_conn_req_rate\": %f, "
        "\"threshold\": %f, \"prach_failure_rate\": %f, \"ho_attempt_rate\": %f}}'",
        attack_type, severity, SEVERITY_MAX, duration, rrc_rate, 
        threshold, prach_failure, ho_rate);
    
    FILE* fp = popen(cmd, "r");
    if (!fp) {
        printf("   [ERROR] OPA query failed\n");
        return;
    }
    
    char buffer[4096];
    size_t len = fread(buffer, 1, sizeof(buffer)-1, fp);
    pclose(fp);
    buffer[len] = '\0';
    
    int prach_backoff = 0;
    int extend_t300 = 0;
    int admission_limit = 100;
    float backoff_ms = 0;
    int t300_ms = T300_NORMAL;
    
    if (strstr(buffer, "\"should_apply_prach_backoff\":true") != NULL ||
        strstr(buffer, "\"should_apply_prach_backoff\": true") != NULL) {
        prach_backoff = 1;
    }
    
    if (strstr(buffer, "\"should_extend_t300\":true") != NULL ||
        strstr(buffer, "\"should_extend_t300\": true") != NULL) {
        extend_t300 = 1;
    }
    
    char* backoff_start = strstr(buffer, "\"t_backoff_ms\":");
    if (backoff_start) {
        char* colon = strchr(backoff_start, ':');
        if (colon) {
            colon++;
            while (*colon == ' ' || *colon == '\t') colon++;
            backoff_ms = atof(colon);
        }
    }
    
    char* t300_start = strstr(buffer, "\"t300_ms\":");
    if (t300_start) {
        char* colon = strchr(t300_start, ':');
        if (colon) {
            colon++;
            while (*colon == ' ' || *colon == '\t') colon++;
            t300_ms = atoi(colon);
        }
    }
    
    char* limit_start = strstr(buffer, "\"admission_limit\":");
    if (limit_start) {
        char* colon = strchr(limit_start, ':');
        if (colon) {
            colon++;
            while (*colon == ' ' || *colon == '\t') colon++;
            admission_limit = atoi(colon);
        }
    }
    
    printf("   [OPA DECISION] Backoff: %s, T300: %s, Admission: %d\n",
           prach_backoff ? "YES" : "NO", extend_t300 ? "YES" : "NO", admission_limit);
    
    if (strcmp(attack_type, "botnet") == 0 || strstr(attack_type, "tm1") != NULL) {
        if (prach_backoff) {
            float effective_backoff = (backoff_ms > 0) ? backoff_ms : calculate_backoff_duration(severity);
            printf("\n   [TM1 MITIGATION]\n");
            printf("   1. PRACH Backoff: %.0f ms\n", effective_backoff);
            printf("   2. T300 Timer: %d ms\n", (extend_t300 ? t300_ms : T300_NORMAL));
            printf("   3. Admission Cap: %d req/s\n", admission_limit);
        }
    } else if (strcmp(attack_type, "handover_storm") == 0 || strstr(attack_type, "tm3") != NULL) {
        int ttt_ms = calculate_ttt_mitigation(severity);
        printf("\n   [TM3 MITIGATION]\n");
        printf("   1. Time-to-Trigger: %d ms\n", ttt_ms);
        printf("   2. Handover Hysteresis: +3 dB\n");
    } else if (strcmp(attack_type, "rogue_bs") == 0 || strstr(attack_type, "tm2") != NULL) {
        printf("\n   [TM2 MITIGATION]\n");
        printf("   1. Cell blocking\n");
        printf("   2. Admission limit: 0 req/s\n");
    }
}

static void sm_cb_kpm(sm_ag_if_rd_t const* rd)
{
    static int callback_count = 0;
    callback_count++;
    
    assert(rd != NULL);
    assert(rd->type == KPM_STATS_V0);
    
    printf("\n=== KPM #%d ===\n", callback_count);
    fflush(stdout);
    
    float features[15] = {0};
    
    if (rd->kpm_stats.msg.MeasData_len > 0) {
        adapter_MeasDataItem_t* meas = &rd->kpm_stats.msg.MeasData[0];
        for (int i = 0; i < 15 && i < meas->measRecord_len; i++) {
            if (meas->measRecord[i].type == MeasRecord_real) {
                features[i] = meas->measRecord[i].real_val;
            } else if (meas->measRecord[i].type == MeasRecord_int) {
                features[i] = (float)meas->measRecord[i].int_val;
            }
        }
    }
    
    float rrc_req = features[IDX_RRC_REQ];
    float rrc_success = features[IDX_RRC_SUCCESS];
    float prach_att = features[IDX_PRACH_ATT];
    float prach_fail = features[IDX_PRACH_FAIL];
    float ho_att = features[IDX_HO_ATT];
    float burst = features[IDX_BURST_INTENSITY];
    float reestab = features[IDX_RRC_REESTAB];
    float cqi = features[IDX_AVG_CQI];
    
    printf("   RRC=%.1f, Succ=%.2f, PRACH=%.1f, HO=%.1f, Burst=%.2f, CQI=%.1f, Reestab=%.1f\n",
           rrc_req, rrc_success, prach_att, ho_att, burst, cqi, reestab);
    
    int fast_path_detected = fast_path_detector_tm1(rrc_req, prach_att, burst, 
                                                     rrc_success, callback_count);
    
    if (fast_path_detected) {
        float severity = calculate_severity(1.5f);
        query_opa_and_mitigate("botnet", severity, 10.0f, rrc_req, LSTM_THRESHOLD, 
                               prach_fail, ho_att);
    }
    
    FILE* fp = fopen(KPM_CSV_PATH, "a");
    if (fp) {
        for (int i = 0; i < 26; i++) {
            if (i < 15) fprintf(fp, "%f", features[i]);
            else fprintf(fp, "0.0");
            if (i < 25) fprintf(fp, ",");
            else fprintf(fp, "\n");
        }
        fclose(fp);
    }
    
    FILE *afp = fopen(ANOMALY_CSV_PATH, "r");
    if (afp) {
        fseek(afp, 0, SEEK_END);
        long file_size = ftell(afp);
        
        if (file_size > anomaly_last_pos) {
            fseek(afp, anomaly_last_pos, SEEK_SET);
            
            char line[512];
            while (fgets(line, sizeof(line), afp)) {
                double ts, mse;
                int is_anomaly;
                char attack_type[64];
                float rrc_win, prach_win, ho_win;
                
                int parsed = sscanf(line, "%lf,%lf,%d,%63[^,],%f,%f,%f", 
                                    &ts, &mse, &is_anomaly, attack_type, 
                                    &rrc_win, &prach_win, &ho_win);
                
                if (parsed >= 4 && is_anomaly == 1) {
                    float severity = calculate_severity((float)mse);
                    
                    printf("\n+------------------------------------------------------------+\n");
                    printf("| SLOW PATH: ANOMALY DETECTED                                  |\n");
                    printf("+------------------------------------------------------------+\n");
                    printf("|   Detection: %s\n", attack_type);
                    printf("|   MSE: %.6f (threshold: %.4f)\n", mse, LSTM_THRESHOLD);
                    printf("|   Severity: %.3f\n", severity);
                    printf("|   Window: RRC=%.1f, PRACH=%.1f, HO=%.1f\n",
                           rrc_win, prach_win, ho_win);
                    printf("+------------------------------------------------------------+\n");
                    
                    if (strcmp(attack_type, "handover_storm") == 0 || 
                        strstr(attack_type, "tm3") != NULL ||
                        strcmp(attack_type, "group_ho") == 0) {
                        
                        printf("\n   Running TM3 Confirmer...\n");
                        int tm3_verified = tm3_confirmer(ts, reestab, cqi);
                        
                        if (tm3_verified) {
                            printf("\n   [TM3 CONFIRMED] Proceeding with mitigation\n");
                            query_opa_and_mitigate("handover_storm", severity, 10.0f,
                                                  rrc_win, LSTM_THRESHOLD, prach_win, ho_win);
                        } else {
                            printf("\n   [TM3 NOT CONFIRMED] Monitoring continues\n");
                        }
                    } else {
                        query_opa_and_mitigate(attack_type, severity, 10.0f,
                                              rrc_win, LSTM_THRESHOLD, prach_win, ho_win);
                    }
                }
            }
            anomaly_last_pos = ftell(afp);
        }
        fclose(afp);
    }
    
    fflush(stdout);
}

int main(int argc, char *argv[])
{
    signal(SIGINT, int_handler);
    fr_args_t args = init_fr_args(argc, argv);
    
    printf("\n+------------------------------------------------------------+\n");
    printf("|     STAM xApp                                               |\n");
    printf("+------------------------------------------------------------+\n");
    
    printf("\n[CONFIGURATION]\n");
    printf("   Fast Path: RRC_min=%.0f, PRACH_min=%.0f, Burst=%.1f\n",
           THRESHOLD_TM1_RRC_MIN, THRESHOLD_TM1_PRACH_MIN, THRESHOLD_BURST);
    printf("   LSTM Threshold: %.4f\n", LSTM_THRESHOLD);
    printf("   TM3 Confirmer: Window=%ds, min_reestab=%d, CQI<threshold=%.1f\n",
           TM3_WINDOW_SECONDS, TM3_MIN_REESTAB, THRESHOLD_CQI_DEGRADE);
    printf("   Mitigation: lambda_bot=%.4f, lambda_legit=%.4f\n", LAMBDA_BOT, LAMBDA_LEGIT);
    
    init_xapp_api(&args);
    sleep(1);
    
    e2_node_arr_t nodes = e2_nodes_xapp_api();
    defer({ free_e2_node_arr(&nodes); });
    assert(nodes.len > 0);
    printf("\n[E2 NODES] Connected: %d\n", nodes.len);
    
    inter_xapp_e i_0 = ms_5;
    sm_ans_xapp_t* kpm_handle = calloc(nodes.len, sizeof(sm_ans_xapp_t));
    assert(kpm_handle != NULL);
    
    for (int i = 0; i < nodes.len; i++) {
        kpm_handle[i] = report_sm_xapp_api(&nodes.n[i].id, SM_KPM_ID, i_0, sm_cb_kpm);
        assert(kpm_handle[i].success == true);
        printf("   Subscribed to node %d\n", i);
    }
    
    printf("\n[STATUS] xApp running. Press Ctrl+C to stop\n\n");
    
    while(keep_running) sleep(1);
    
    printf("\n[CLEANUP] Shutting down...\n");
    for(int i = 0; i < nodes.len; ++i)
        rm_report_sm_xapp_api(kpm_handle[i].u.handle);
    
    free(kpm_handle);
    
    while(try_stop_xapp_api() == false) usleep(1000);
    printf("[STATUS] STAM xApp stopped\n");
    return 0;
}
