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
__global__ void convForward(const float* const data, float* output);

/**
 * @brief What is the standard input of a backward process
 * @return __global__ 
 */
__global__ void convBackward(const float* const eval, float* grad);