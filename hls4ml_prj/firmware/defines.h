#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 10
#define N_INPUT_2_1 14
#define N_INPUT_1_1 10
#define N_INPUT_2_1 14
#define N_OUTPUTS_3 10
#define N_FILT_3 10
#define N_OUTPUTS_3 10
#define N_FILT_3 10
#define N_OUTPUTS_6 10
#define N_FILT_6 10
#define N_OUTPUTS_6 10
#define N_FILT_6 10
#define N_FILT_9 10
#define N_LAYER_10 10
#define N_LAYER_10 10
#define N_LAYER_13 1


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<12,6,AP_TRN,AP_SAT,0> input_t;
typedef ap_fixed<12,8,AP_TRN,AP_SAT,0> layer2_t;
typedef ap_fixed<18,8> q_input_table_t;
typedef ap_fixed<14,8,AP_TRN,AP_SAT,0> q_conv1d_accum_t;
typedef ap_fixed<14,8,AP_TRN,AP_SAT,0> layer3_t;
typedef ap_fixed<10,5> weight3_t;
typedef ap_fixed<10,5> bias3_t;
typedef ap_ufixed<14,8,AP_TRN,AP_SAT,0> layer5_t;
typedef ap_fixed<18,8> q_activation_table_t;
typedef ap_fixed<16,12,AP_TRN,AP_SAT,0> q_conv1d_1_accum_t;
typedef ap_fixed<14,8,AP_TRN,AP_SAT,0> layer6_t;
typedef ap_fixed<10,5> weight6_t;
typedef ap_fixed<10,5> bias6_t;
typedef ap_ufixed<10,5,AP_TRN,AP_SAT,0> layer8_t;
typedef ap_fixed<18,8> q_activation_1_table_t;
typedef ap_fixed<16,12,AP_TRN,AP_SAT,0> global_average_pooling1d_accum_t;
typedef ap_fixed<14,8,AP_TRN,AP_SAT,0> layer9_t;
typedef ap_fixed<14,10,AP_TRN,AP_SAT,0> q_dense_accum_t;
typedef ap_fixed<14,8,AP_TRN,AP_SAT,0> layer10_t;
typedef ap_fixed<10,5> weight10_t;
typedef ap_fixed<10,5> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_ufixed<14,8,AP_TRN,AP_SAT,0> layer12_t;
typedef ap_fixed<18,8> q_activation_2_table_t;
typedef ap_fixed<14,10,AP_TRN,AP_SAT,0> q_dense_1_accum_t;
typedef ap_fixed<14,8,AP_TRN,AP_SAT,0> result_t;
typedef ap_fixed<10,5> weight13_t;
typedef ap_fixed<10,5> bias13_t;
typedef ap_uint<1> layer13_index;


#endif
