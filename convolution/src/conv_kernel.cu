/**
 * CUDA convolution kernel
 * @author HQY @date 2021.9.26 
 */
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
            output[b * out_chan * gbase1 + gbase1 * i + gbase2 * y + x] = tmp;
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


/// @note 输入是(N, C_i, H, W)以及(C_o, C_i, K, K)的卷积核,输出是(N, C_O, H, W)
/// @note 注意，输入的data进行了padding，而输出output没有padding，不考虑stride
/// @ 输入图像从pad开始 到x+pad结束
__global__ void convForwardV2(
    ConstFloatPtr data, ConstFloatPtr kernel, ConstFloatPtr bias, 
    FloatPtr output, const int ks, const int co_num
) {
    /// 数据复制到shared memory下 注意，这里有两部分
    /// 一部分是原始数据，另一部分是累加结果
    /// 一个函数只处理batch中的一张图片中的一个点(x, y)
    extern __shared__ float proc[];
    const int n = blockIdx.x, y = blockIdx.y, x = blockIdx.z, c = threadIdx.x, k = threadIdx.y, hks = (ks / 2);
    const int row_offset = gridDim.z + (hks << 1), chan_offset = (gridDim.y + hks << 1) * row_offset;
    const int row_base = (y + hks) * row_offset;
    const int p_row_offset = ks, p_chn_offset = ks * ks, co_offset = blockDim.x * p_chn_offset;
    const int full_base = n * chan_offset * blockDim.x + c * chan_offset + row_base + x + hks, 
              p_full_base = c * p_chn_offset + k * p_row_offset;
    FloatPtr co_output = &proc[co_num * co_offset];     // 位置需要指定
    for (int i = -hks; i <= hks; i++) {
        proc[p_full_base + i] = data[full_base + (k - hks) * row_offset + i];
    }
    FloatPtr ptr = co_output;
    for (int i = 0; i < co_num; i++) {
        ConstFloatPtr ki = &kernel[i * co_offset];
        for (int j = 0; j < ks; j++) {
            float val = ki[p_full_base + j] * proc[p_full_base + j];
            atomicAdd_system(ptr, val);
        }
        ptr++;
    }
    __syncthreads();
    // 从共享内存复制到global内存
    const int row_base_no_pad = y * gridDim.z;
    if (k == 0) {       // warp divergence
        const int obatch_base = n * co_num * chan_offset;
        for (int i = 0; i < 4; i++) {
            const int id = blockDim.x * i + c;
            if (id >= co_num) break;
            output[obatch_base + id * chan_offset + row_base_no_pad + x] = co_output[id] + bias[id];
        }
    }
}

/// 所以，设计应该是：<<<(N, C_o,  K * K), (C_i, H, W)>>>, 输入的x经过padding
__global__ void convBackwardForW(
    ConstFloatPtr grad_upstream, ConstFloatPtr x, FloatPtr grad_w, const int k
) {
    extern __shared__ float all_ci[];
    const int hk = (k >> 1);
    const int r_offset = blockDim.z, r_offset_p = r_offset + (hk << 1),
        c_offset = blockDim.y * r_offset, c_offset_p = (blockDim.y + (hk << 1) * r_offset_p),
        b_offset = blockDim.x * c_offset, b_offset_p = blockDim.x * c_offset_p;
    const int k2 = k * k, wc_offset = blockDim.x * k2;
    const int ci = threadIdx.x, n = blockIdx.x, co = blockIdx.y, w_id = blockIdx.z, k_row = w_id / k, k_col = w_id % k;
    FloatPtr this_ci = &all_ci[ci];
    float val = x[n * b_offset_p + ci * c_offset_p + (threadIdx.y + k_row) * r_offset_p + k_col + threadIdx.z] *
        grad_upstream[n * b_offset + co * c_offset + threadIdx.y * r_offset + threadIdx.z];
    atomicAdd_system(this_ci, val);
    __syncthreads();
    if (threadIdx.y == 0 && threadIdx.z == 0) {
        grad_w[co * wc_offset + ci * k2 + w_id] = all_ci[ci];
    }
}

__global__ void convBackwardForX(
    
) {
    ;
}
