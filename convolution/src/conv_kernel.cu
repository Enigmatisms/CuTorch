#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#include <device_functions.h>
#include "../include/conv_kernel.h"

/**
 * @brief Convolution forward function
 * kernel is organized as grid(N, H, W) block(C_i, K, K)
 * @param output 
 * @return __global__ 
 */
__global__ void convForward(const float* const data, const float* const kernel, int out_chan, float* output) {
    const int gbase2 = gridDim.z, gbase1 = gridDim.y * gbase2, gbase0 = blockDim.x * gbase1;
    const int b = blockIdx.x, y = blockIdx.y, x = blockIdx.z, k_size = blockDim.y, half_k = (k_size >> 1);
    const int pos_y = threadIdx.y, pos_x = threadIdx.z, ch_id = threadIdx.x;
    const int ofst_x = pos_x - half_k + x, ofst_y = pos_y - half_k + y;
    const int k_sqr = k_size * k_size;
    const int id_offset = ch_id * k_sqr + pos_y * k_size + pos_x;
    const float val = data[gbase0 * b + gbase1 * ch_id + ofst_y * gbase2 + ofst_x];
    const int kbase = blockDim.x * k_sqr; 
    extern __shared__ float data_block[];       // k size * k size * sizeof(float) * in_channel + sizeof(float) * in_channel
    // kernel weight data is (C_o, C_i, K, K)
    for (int i = 0; i < out_chan; i++) {
        // kernel is too big to fit into the shared memory (occupancy & latency hiding)
        float tmp = kernel[i * kbase + id_offset] * val;        
        data_block[ch_id] += tmp;
        __syncthreads();                    // wait till c_i, k, k threads are all done
        if ((ch_id | x | y) == 0)           // execute only once (warp divergence)
            output[b * out_chan * gbase1 + gbase1 * i + gbase2 * y + x];
        __syncthreads();
        // bias is added in another kernel function
    }
}

/// @note grid will be (N, H, W), block shall be 1-dim (C)
__global__ void biasForward(const float* const data, const float* const bias, float* output) {
    const int gbase2 = blockDim.x, gbase1 = gridDim.z * gbase2, gbase0 = gridDim.y * gbase1;
    const int id = threadIdx.x, base = blockIdx.x * gbase0 + id * gbase1 + blockIdx.y * gbase2 + blockIdx.z;
    output[base] = data[base] + bias[id];
}
