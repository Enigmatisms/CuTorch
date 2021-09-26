#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief 
 * Note that a convolution layer with ci input channel and co output channel, kernel size of k
 * should be co different ci * (k * k) (weight parameters), and (co) bias parameters
 * The problem is, where should I implement the parameters, should method "paramters()" be reloaded?
 */

/**
 * @brief implement convolution method by hand 
 * @return 
 */
__global__ void weightForward(const float* const data, const float* const kernel, int out_chan, float* output);

/**
 * @brief shared memory optimization might not be so meaningful here.
 */
__global__ void biasForward(const float* const data, const float* const bias, float* output);

/**
 * @brief What is the standard input of a backward process
 * @return __global__ 
 */
__global__ void convBackward(const float* const eval, float* grad);