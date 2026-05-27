#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

#include "nnet_conv1d_latency.h"
#include "nnet_helpers.h"

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_function_stubs.h"
#include "nnet_mult.h"

namespace hls4ml_model_emu_v3 {

template <class data_T, class res_T, typename CONFIG_T> class PointwiseConv1D {
  public:
    static void pointwise_conv(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                               res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                               typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                               typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        // To be implemented in subclasses
    }
};

// hls4ml insert code
template<class data_T, typename CONFIG_T>
class fill_buffer_15 : public nnet::FillConv1DBuffer<data_T, CONFIG_T> {
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        if (partition ==   0) {
            buffer[0][0] =    data[0]; buffer[0][1] =    data[1]; buffer[0][2] =    data[2]; buffer[0][3] =    data[3]; buffer[0][4] =    data[4]; buffer[0][5] =    data[5]; buffer[0][6] =    data[6]; buffer[0][7] =    data[7]; buffer[0][8] =    data[8]; buffer[0][9] =    data[9]; buffer[0][10] =   data[10]; buffer[0][11] =   data[11]; buffer[0][12] =   data[12];

        }
        if (partition ==   1) {
            buffer[0][0] =   data[13]; buffer[0][1] =   data[14]; buffer[0][2] =   data[15]; buffer[0][3] =   data[16]; buffer[0][4] =   data[17]; buffer[0][5] =   data[18]; buffer[0][6] =   data[19]; buffer[0][7] =   data[20]; buffer[0][8] =   data[21]; buffer[0][9] =   data[22]; buffer[0][10] =   data[23]; buffer[0][11] =   data[24]; buffer[0][12] =   data[25];

        }
        if (partition ==   2) {
            buffer[0][0] =   data[26]; buffer[0][1] =   data[27]; buffer[0][2] =   data[28]; buffer[0][3] =   data[29]; buffer[0][4] =   data[30]; buffer[0][5] =   data[31]; buffer[0][6] =   data[32]; buffer[0][7] =   data[33]; buffer[0][8] =   data[34]; buffer[0][9] =   data[35]; buffer[0][10] =   data[36]; buffer[0][11] =   data[37]; buffer[0][12] =   data[38];

        }
        if (partition ==   3) {
            buffer[0][0] =   data[39]; buffer[0][1] =   data[40]; buffer[0][2] =   data[41]; buffer[0][3] =   data[42]; buffer[0][4] =   data[43]; buffer[0][5] =   data[44]; buffer[0][6] =   data[45]; buffer[0][7] =   data[46]; buffer[0][8] =   data[47]; buffer[0][9] =   data[48]; buffer[0][10] =   data[49]; buffer[0][11] =   data[50]; buffer[0][12] =   data[51];

        }
        if (partition ==   4) {
            buffer[0][0] =   data[52]; buffer[0][1] =   data[53]; buffer[0][2] =   data[54]; buffer[0][3] =   data[55]; buffer[0][4] =   data[56]; buffer[0][5] =   data[57]; buffer[0][6] =   data[58]; buffer[0][7] =   data[59]; buffer[0][8] =   data[60]; buffer[0][9] =   data[61]; buffer[0][10] =   data[62]; buffer[0][11] =   data[63]; buffer[0][12] =   data[64];

        }
        if (partition ==   5) {
            buffer[0][0] =   data[65]; buffer[0][1] =   data[66]; buffer[0][2] =   data[67]; buffer[0][3] =   data[68]; buffer[0][4] =   data[69]; buffer[0][5] =   data[70]; buffer[0][6] =   data[71]; buffer[0][7] =   data[72]; buffer[0][8] =   data[73]; buffer[0][9] =   data[74]; buffer[0][10] =   data[75]; buffer[0][11] =   data[76]; buffer[0][12] =   data[77];

        }
        if (partition ==   6) {
            buffer[0][0] =   data[78]; buffer[0][1] =   data[79]; buffer[0][2] =   data[80]; buffer[0][3] =   data[81]; buffer[0][4] =   data[82]; buffer[0][5] =   data[83]; buffer[0][6] =   data[84]; buffer[0][7] =   data[85]; buffer[0][8] =   data[86]; buffer[0][9] =   data[87]; buffer[0][10] =   data[88]; buffer[0][11] =   data[89]; buffer[0][12] =   data[90];

        }
        if (partition ==   7) {
            buffer[0][0] =   data[91]; buffer[0][1] =   data[92]; buffer[0][2] =   data[93]; buffer[0][3] =   data[94]; buffer[0][4] =   data[95]; buffer[0][5] =   data[96]; buffer[0][6] =   data[97]; buffer[0][7] =   data[98]; buffer[0][8] =   data[99]; buffer[0][9] =  data[100]; buffer[0][10] =  data[101]; buffer[0][11] =  data[102]; buffer[0][12] =  data[103];

        }
        if (partition ==   8) {
            buffer[0][0] =  data[104]; buffer[0][1] =  data[105]; buffer[0][2] =  data[106]; buffer[0][3] =  data[107]; buffer[0][4] =  data[108]; buffer[0][5] =  data[109]; buffer[0][6] =  data[110]; buffer[0][7] =  data[111]; buffer[0][8] =  data[112]; buffer[0][9] =  data[113]; buffer[0][10] =  data[114]; buffer[0][11] =  data[115]; buffer[0][12] =  data[116];

        }
        if (partition ==   9) {
            buffer[0][0] =  data[117]; buffer[0][1] =  data[118]; buffer[0][2] =  data[119]; buffer[0][3] =  data[120]; buffer[0][4] =  data[121]; buffer[0][5] =  data[122]; buffer[0][6] =  data[123]; buffer[0][7] =  data[124]; buffer[0][8] =  data[125]; buffer[0][9] =  data[126]; buffer[0][10] =  data[127]; buffer[0][11] =  data[128]; buffer[0][12] =  data[129];

        }
    }
};
template<class data_T, class res_T, typename CONFIG_T>
class pointwise_conv_15 : public nnet::Conv1DKernel<data_T, res_T, CONFIG_T> {
  public:
    static void conv(
                     data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                     res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                     typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                     typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        data_T data_tmp[CONFIG_T::reuse_factor][CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor];
        #pragma HLS ARRAY_PARTITION variable=data_tmp complete dim=0
        res_T res_tmp[CONFIG_T::reuse_factor][CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor];
        #pragma HLS ARRAY_PARTITION variable=res_tmp complete dim=0

    RFInputLoop:
        for (int jj = 0; jj < CONFIG_T::reuse_factor; jj++) {
        #pragma HLS UNROLL
        InnerInputLoop:
            for (int ii = 0; ii < CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor; ii++) {
                #pragma HLS UNROLL
                data_tmp[jj][ii] = data[jj * CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor + ii];
            }
        }

        nnet::pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[0], res_tmp[0], weights, biases);

    RFOutputLoop:
        for (int jj = 0; jj < CONFIG_T::reuse_factor; jj++) {
        #pragma HLS UNROLL
        InnerOutputLoop:
            for (int ii = 0; ii < CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor; ii++) {
                #pragma HLS UNROLL
                res[jj * CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor + ii] = res_tmp[jj][ii];
            }
        }
    }
};
template<class data_T, typename CONFIG_T>
class fill_buffer_16 : public nnet::FillConv1DBuffer<data_T, CONFIG_T> {
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        if (partition ==   0) {
            buffer[0][0] =    data[0]; buffer[0][1] =    data[1]; buffer[0][2] =    data[2]; buffer[0][3] =    data[3]; buffer[0][4] =    data[4]; buffer[0][5] =    data[5]; buffer[0][6] =    data[6]; buffer[0][7] =    data[7]; buffer[0][8] =    data[8]; buffer[0][9] =    data[9];

        }
        if (partition ==   1) {
            buffer[0][0] =   data[10]; buffer[0][1] =   data[11]; buffer[0][2] =   data[12]; buffer[0][3] =   data[13]; buffer[0][4] =   data[14]; buffer[0][5] =   data[15]; buffer[0][6] =   data[16]; buffer[0][7] =   data[17]; buffer[0][8] =   data[18]; buffer[0][9] =   data[19];

        }
        if (partition ==   2) {
            buffer[0][0] =   data[20]; buffer[0][1] =   data[21]; buffer[0][2] =   data[22]; buffer[0][3] =   data[23]; buffer[0][4] =   data[24]; buffer[0][5] =   data[25]; buffer[0][6] =   data[26]; buffer[0][7] =   data[27]; buffer[0][8] =   data[28]; buffer[0][9] =   data[29];

        }
        if (partition ==   3) {
            buffer[0][0] =   data[30]; buffer[0][1] =   data[31]; buffer[0][2] =   data[32]; buffer[0][3] =   data[33]; buffer[0][4] =   data[34]; buffer[0][5] =   data[35]; buffer[0][6] =   data[36]; buffer[0][7] =   data[37]; buffer[0][8] =   data[38]; buffer[0][9] =   data[39];

        }
        if (partition ==   4) {
            buffer[0][0] =   data[40]; buffer[0][1] =   data[41]; buffer[0][2] =   data[42]; buffer[0][3] =   data[43]; buffer[0][4] =   data[44]; buffer[0][5] =   data[45]; buffer[0][6] =   data[46]; buffer[0][7] =   data[47]; buffer[0][8] =   data[48]; buffer[0][9] =   data[49];

        }
        if (partition ==   5) {
            buffer[0][0] =   data[50]; buffer[0][1] =   data[51]; buffer[0][2] =   data[52]; buffer[0][3] =   data[53]; buffer[0][4] =   data[54]; buffer[0][5] =   data[55]; buffer[0][6] =   data[56]; buffer[0][7] =   data[57]; buffer[0][8] =   data[58]; buffer[0][9] =   data[59];

        }
        if (partition ==   6) {
            buffer[0][0] =   data[60]; buffer[0][1] =   data[61]; buffer[0][2] =   data[62]; buffer[0][3] =   data[63]; buffer[0][4] =   data[64]; buffer[0][5] =   data[65]; buffer[0][6] =   data[66]; buffer[0][7] =   data[67]; buffer[0][8] =   data[68]; buffer[0][9] =   data[69];

        }
        if (partition ==   7) {
            buffer[0][0] =   data[70]; buffer[0][1] =   data[71]; buffer[0][2] =   data[72]; buffer[0][3] =   data[73]; buffer[0][4] =   data[74]; buffer[0][5] =   data[75]; buffer[0][6] =   data[76]; buffer[0][7] =   data[77]; buffer[0][8] =   data[78]; buffer[0][9] =   data[79];

        }
        if (partition ==   8) {
            buffer[0][0] =   data[80]; buffer[0][1] =   data[81]; buffer[0][2] =   data[82]; buffer[0][3] =   data[83]; buffer[0][4] =   data[84]; buffer[0][5] =   data[85]; buffer[0][6] =   data[86]; buffer[0][7] =   data[87]; buffer[0][8] =   data[88]; buffer[0][9] =   data[89];

        }
        if (partition ==   9) {
            buffer[0][0] =   data[90]; buffer[0][1] =   data[91]; buffer[0][2] =   data[92]; buffer[0][3] =   data[93]; buffer[0][4] =   data[94]; buffer[0][5] =   data[95]; buffer[0][6] =   data[96]; buffer[0][7] =   data[97]; buffer[0][8] =   data[98]; buffer[0][9] =   data[99];

        }
    }
};
template<class data_T, class res_T, typename CONFIG_T>
class pointwise_conv_16 : public nnet::Conv1DKernel<data_T, res_T, CONFIG_T> {
  public:
    static void conv(
                     data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                     res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                     typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                     typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        data_T data_tmp[CONFIG_T::reuse_factor][CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor];
        #pragma HLS ARRAY_PARTITION variable=data_tmp complete dim=0
        res_T res_tmp[CONFIG_T::reuse_factor][CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor];
        #pragma HLS ARRAY_PARTITION variable=res_tmp complete dim=0

    RFInputLoop:
        for (int jj = 0; jj < CONFIG_T::reuse_factor; jj++) {
        #pragma HLS UNROLL
        InnerInputLoop:
            for (int ii = 0; ii < CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor; ii++) {
                #pragma HLS UNROLL
                data_tmp[jj][ii] = data[jj * CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor + ii];
            }
        }

        nnet::pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[0], res_tmp[0], weights, biases);

    RFOutputLoop:
        for (int jj = 0; jj < CONFIG_T::reuse_factor; jj++) {
        #pragma HLS UNROLL
        InnerOutputLoop:
            for (int ii = 0; ii < CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor; ii++) {
                #pragma HLS UNROLL
                res[jj * CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor + ii] = res_tmp[jj][ii];
            }
        }
    }
};

} // namespace hls4ml_model_emu_v3

#endif
