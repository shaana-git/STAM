/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include "../../../src/agent/e2_agent_api.h"
#include "../../../test/sm/common/fill_ind_data.h"
#include <assert.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <poll.h>
#include <unistd.h>

/* KPM Provider Bridge - same folder as test_agent.c */
#include "kpm_provider_bridge.h"

/* Anomaly threshold from training (98th percentile) */
#define ANOMALY_THRESHOLD 1.271798f

static
void read_RAN(sm_ag_if_rd_t* data)
{
  assert(data->type == MAC_STATS_V0 || data->type == RLC_STATS_V0 || data->type == PDCP_STATS_V0 || data->type == SLICE_STATS_V0 || data->type == KPM_STATS_V0 || data->type == GTP_STATS_V0);

  if(data->type == MAC_STATS_V0){
    fill_mac_ind_data(&data->mac_stats);
  } else if(data->type == RLC_STATS_V0) {
    fill_rlc_ind_data(&data->rlc_stats);
  } else if(data->type == PDCP_STATS_V0){
    fill_pdcp_ind_data(&data->pdcp_stats);
  } else if(data->type == SLICE_STATS_V0){
    fill_slice_ind_data(&data->slice_stats);
  } else if(data->type == GTP_STATS_V0){
    fill_gtp_ind_data(&data->gtp_stats);
  } else if(data->type == KPM_STATS_V0){
    static int initialized = 0;
    float kpm_values[26];
    
    if (!initialized) {
        /* SELECT ATTACK TYPE HERE - Uncomment ONE of these: */
        
        /* NORMAL TRAFFIC (Default) 
        if (init_kpm_provider("Urban_Low", NULL, 0.0) == 0) {
            initialized = 1;
            printf("\n=== E2 AGENT STARTED ===\n");
            printf("Mode: NORMAL Traffic (Urban_Low)\n");
            printf("Threshold: %.4f\n", ANOMALY_THRESHOLD);
            printf("==========================\n\n");
        }
        */
        
        /*
        // TM1 - FLOOD ATTACK (Botnet/Flood)
        if (init_kpm_provider("Urban_Low", "tm1", 1.0) == 0) {
            initialized = 1;
            printf("\n=== E2 AGENT STARTED ===\n");
            printf("Mode: TM1 FLOOD ATTACK (severe, Urban_Low)\n");
            printf("==========================\n\n");
        }
        */
        
        /*
        // TM2 - FAKE BS ATTACK (Rogue BS)
        if (init_kpm_provider("Urban_Medium", "tm2", 1.0) == 0) {
            initialized = 1;
            printf("\n=== E2 AGENT STARTED ===\n");
            printf("Mode: TM2 FAKE BS ATTACK (severe, Urban_Medium)\n");
            printf("==========================\n\n");
        }
        */
        
        // TM3 - HO STORM ATTACK (Group HO)
        if (init_kpm_provider("Highway_High", "tm3", 1.0) == 0) {
            initialized = 1;
            printf("\n=== E2 AGENT STARTED ===\n");
            printf("Mode: TM3 HO STORM ATTACK (severe, Highway_High)\n");
            printf("==========================\n\n");
        }
        
        else {
            printf("[ERROR] Failed to init KPM provider\n");
            return;
        }
    }
    
    /* Get KPM measurement (26 values: first 15 real, rest padded) */
    if (get_next_kpm_measurement(kpm_values, 26) == 0) {
        /* Fill KPM data structure for E2 interface */
        data->kpm_stats.msg.MeasData_len = 1;
        data->kpm_stats.msg.MeasData = calloc(1, sizeof(adapter_MeasDataItem_t));
        data->kpm_stats.msg.MeasData[0].measRecord_len = 26;
        data->kpm_stats.msg.MeasData[0].measRecord = calloc(26, sizeof(adapter_MeasRecord_t));
        data->kpm_stats.msg.MeasData[0].incompleteFlag = 0;
        
        for (int i = 0; i < 26; i++) {
            data->kpm_stats.msg.MeasData[0].measRecord[i].type = MeasRecord_real;
            data->kpm_stats.msg.MeasData[0].measRecord[i].real_val = kpm_values[i];
        }
        
        /* Display first 5 KPI values */
        printf("\n[E2] Sending KPM data to Near-RT RIC via E2 interface\n");
        printf("[E2->RIC] RRC_req=%.0f, RRC_succ=%.2f, PRACH_att=%.0f, PRACH_fail=%.2f, HO_att=%.0f\n", 
               kpm_values[0], kpm_values[1], kpm_values[2], kpm_values[3], kpm_values[4]);
    }
  } else {
    assert(0 && "Invalid data type");
  }
}

static
sm_ag_if_ans_t write_RAN(sm_ag_if_wr_t const* data)
{
  assert(data != NULL);
  if(data->type == MAC_CTRL_REQ_V0){
    sm_ag_if_ans_t ans = {.type = MAC_AGENT_IF_CTRL_ANS_V0};
    ans.mac.ans = MAC_CTRL_OUT_OK;
    return ans;
  } else if(data->type == SLICE_CTRL_REQ_V0){
    slice_ctrl_req_data_t const* slice_req_ctrl = &data->slice_req_ctrl;
    slice_ctrl_msg_t const* msg = &slice_req_ctrl->msg;

    if(msg->type == SLICE_CTRL_SM_V0_ADD){
        printf("[E2 Agent]: SLICE CONTROL ADD rx\n");
    } else if(msg->type == SLICE_CTRL_SM_V0_DEL){
        printf("[E2 Agent]: SLICE CONTROL DEL rx\n");
    } else if(msg->type == SLICE_CTRL_SM_V0_UE_SLICE_ASSOC){
        printf("[E2 Agent]: SLICE CONTROL ASSOC rx\n");
    } else {
      assert(0 && "Unknown msg_type!");
    }
    sm_ag_if_ans_t ans = {.type = SLICE_AGENT_IF_CTRL_ANS_V0};
    return ans;
  } else if(data->type == TC_CTRL_REQ_V0){
    tc_ctrl_req_data_t const* ctrl = &data->tc_req_ctrl;
    tc_ctrl_msg_e const t = ctrl->msg.type;
    assert(t == TC_CTRL_SM_V0_CLS || t == TC_CTRL_SM_V0_PLC 
          || t == TC_CTRL_SM_V0_QUEUE || t == TC_CTRL_SM_V0_SCH 
          || t == TC_CTRL_SM_V0_SHP || t == TC_CTRL_SM_V0_PCR);
    sm_ag_if_ans_t ans = {.type = TC_AGENT_IF_CTRL_ANS_V0};
    return ans;
  } else {
    assert(0 && "Not supported function");
  }
  sm_ag_if_ans_t ans = {0};
  return ans;
}

static
void sig_handler(int sig_num)
{
  printf("\n[E2 Agent] Stopped by user signal %d\n", sig_num);
  close_kpm_provider();  /* Clean up Python bridge */
  stop_agent_api();
  exit(0);
}

int main(int argc, char *argv[])
{
  signal(SIGINT, sig_handler);

#ifdef TEST_AGENT_GNB
  const ngran_node_t ran_type = ngran_gNB;
  const int mcc = 505;
  const int mnc = 1;
  const int mnc_digit_len = 2;
  const int nb_id = 1;
  const int cu_du_id = 0;
#elif TEST_AGENT_GNB_CU
  const ngran_node_t ran_type = ngran_gNB_CU;
  const int mcc = 505;
  const int mnc = 1;
  const int mnc_digit_len = 2;
  const int nb_id = 2;
  const int cu_du_id = 21;
#elif TEST_AGENT_GNB_DU
  const ngran_node_t ran_type = ngran_gNB_DU;
  const int mcc = 505;
  const int mnc = 1;
  const int mnc_digit_len = 2;
  const int nb_id = 2;
  const int cu_du_id = 22;
#elif TEST_AGENT_ENB
  const ngran_node_t ran_type = ngran_eNB;
  const int mcc = 208;
  const int mnc = 94;
  const int mnc_digit_len = 2;
  const int nb_id = 4;
  const int cu_du_id = 0;
#else
  static_assert(0 != 0, "Unknown type");
#endif

  sm_io_ag_t io = {.read = read_RAN, .write = write_RAN};
  fr_args_t args = init_fr_args(argc, argv);

  printf("[E2 Agent] Connecting to RIC...\n");
  init_agent_api(mcc, mnc, mnc_digit_len, nb_id, cu_du_id, ran_type, io, &args);

  while(1){
    poll(NULL, 0, 1000);
  }

  return EXIT_SUCCESS;
}
