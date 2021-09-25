#include "../include/conv_kernel.h"

/// @note note that input might be N, C, H, W, therefore base should be input
/**
 * My strategy: N(batches comes first), H, W
 */
__global__ void convForward(const float* const data, float* output) {

}

