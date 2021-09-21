#include <cmath>
#include <cstdio>
#include "../include/sdf_kernel.h"

/// @note note that SDF map is 600 * 450, while I want to display a 1200 * 900 result
/// threrefore we need a pixel-wise interpolation

__global__ void fineGrainedTask(const float* const bubbles, const int x, const int y, float* shared_tmp) {
    const int id = threadIdx.x, base = 3 * id;
    const float cx = bubbles[base], cy = bubbles[base + 1], radius = bubbles[base + 2];
    shared_tmp[id] = signedDistance(x, y, cx, cy, radius);
}

/// @brief input 2D Tensor (pointer) bubbles and calculate SDF
__global__ void calculateSDF(const float* const bubbles, const int num, float* output) {
    const int y = threadIdx.x, x = blockIdx.x, id = y * gridDim.x + x;
    float distance = 0.0;
    // fineGrainedTask <<< 1, num >>> (bubbles, x, y, tmp);
    // cudaDeviceSynchronize();
    for (int i = 0; i < num; i++) {
        const int base = 3 * i;
        const float cx = bubbles[base], cy = bubbles[base + 1], radius = bubbles[base + 2];
        distance += signedDistance(x, y, cx, cy, radius);
    }
    output[id] = distance - 1.0;
}

__global__ void gridTagsCalculation(const float* const sdf, uint8_t* tags) {
    const int i = threadIdx.x, j = blockIdx.x, id = i * gridDim.x + j;
    const float vals[4] = {
        sdf[id + gridDim.x + i + 1], sdf[id + gridDim.x + i + 2],
        sdf[id + i + 1], sdf[id + i],
    };
    uint8_t tag = 0x0f, and_elem = 0xfe;
    for (int k = 0; k < 4; k++) {
        if (vals[k] < 0)
            tag &= and_elem;
        and_elem <<= 1;
        and_elem |= 0x01;
    }
    tags[id] = tag;
}

__device__ float signedDistance(const float x, const float y, const float cx, const float cy, const float radius) {
    return (pow(radius, 2) / (pow(x - cx, 2) + pow(y - cy, 2) + 0.1));
}
