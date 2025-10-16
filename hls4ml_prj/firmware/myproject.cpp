#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_1[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer13_out[N_LAYER_13]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer13_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<weight3_t, 140>(w3, "w3.txt");
        nnet::load_weights_from_txt<bias3_t, 10>(b3, "b3.txt");
        nnet::load_weights_from_txt<weight6_t, 100>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 10>(b6, "b6.txt");
        nnet::load_weights_from_txt<weight10_t, 100>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 10>(b10, "b10.txt");
        nnet::load_weights_from_txt<weight13_t, 10>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 1>(b13, "b13.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[N_OUTPUTS_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    layer5_t layer5_out[N_OUTPUTS_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer6_t layer6_out[N_OUTPUTS_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    layer8_t layer8_out[N_OUTPUTS_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    layer9_t layer9_out[N_FILT_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    layer10_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    layer12_t layer12_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    nnet::linear<input_t, layer2_t, linear_config2>(input_1, layer2_out); // q_input
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer2_t>(layer2_out, "q_input", N_INPUT_1_1*N_INPUT_2_1);
#endif

    nnet::pointwise_conv_1d_cl<layer2_t, layer3_t, config15>(layer2_out, layer3_out, w3, b3); // q_conv1d
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer3_t>(layer3_out, "q_conv1d", N_OUTPUTS_3*N_FILT_3);
#endif

    nnet::relu<layer3_t, layer5_t, relu_config5>(layer3_out, layer5_out); // q_activation
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer5_t>(layer5_out, "q_activation", N_OUTPUTS_3*N_FILT_3);
#endif

    nnet::pointwise_conv_1d_cl<layer5_t, layer6_t, config16>(layer5_out, layer6_out, w6, b6); // q_conv1d_1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer6_t>(layer6_out, "q_conv1d_1", N_OUTPUTS_6*N_FILT_6);
#endif

    nnet::relu<layer6_t, layer8_t, relu_config8>(layer6_out, layer8_out); // q_activation_1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer8_t>(layer8_out, "q_activation_1", N_OUTPUTS_6*N_FILT_6);
#endif

    nnet::global_pooling1d_cl<layer8_t, layer9_t, config9>(layer8_out, layer9_out); // global_average_pooling1d
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer9_t>(layer9_out, "global_average_pooling1d", N_FILT_9);
#endif

    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // q_dense
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer10_t>(layer10_out, "q_dense", N_LAYER_10);
#endif

    nnet::relu<layer10_t, layer12_t, relu_config12>(layer10_out, layer12_out); // q_activation_2
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer12_t>(layer12_out, "q_activation_2", N_LAYER_10);
#endif

    nnet::dense<layer12_t, result_t, config13>(layer12_out, layer13_out, w13, b13); // q_dense_1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<result_t>(layer13_out, "q_dense_1", N_LAYER_13);
#endif

}

