#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef unsigned char uint8_t;

/**
 * @brief SDF calculation for bubbles
 * CUDA image processing uses 1D tensor
 */

/// @brief input 2D Tensor (pointer) bubbles and calculate SDF
__global__ void fineGrainedTask(const float* const bubbles, const int x, const int y, float* shared_tmp);

__global__ void calculateSDF(const float* const bubbles, const int num, float* output);

__global__ void gridTagsCalculation(const float* const lut, uint8_t* tags);

inline __device__ float signedDistance(const float x, const float y, const float cx, const float cy, const float radius);
