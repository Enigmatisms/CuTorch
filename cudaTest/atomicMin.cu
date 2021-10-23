#include <iostream>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

// __device__ uint32_t floatToOrderedInt( float floatVal ) {
//     uint32_t uintVal = __float_as_uint( floatVal );
//     uint32_t mask = -int(uintVal >> 31) | 0x80000000;
//     return uintVal ^ mask;
// }
// __device__ float orderedIntToFloat( uint32_t uintVal ) {
//     uint32_t mask = ((uintVal >> 31) - 1) | 0x80000000;
// 	return __uint_as_float(uintVal ^ mask);
// }

template<bool V>
__device__ void templateFunc(const float& v) {
    if (V == true) {
        printf("Template: true, %f\n", v);
    } else {
        printf("Template false, %f\n", -v);
    }
}

template <bool single>
__global__ void testNest(float v) {
    int id = threadIdx.x;
    if (single == true) {
        printf("%f, %d\n", v, id);
    } else {
        printf("Not single: %f, %d\n", v, id);
    }
    if (id & 1) {
        templateFunc<true>(v);
    } else {
        templateFunc<false>(v);
    }
}

__device__ __forceinline__ int floatToOrderedInt( float floatVal ) {
    int intVal = __float_as_int( floatVal );
    if (intVal & 1) {
        testNest<true><<< 1, 8 >>> (floatVal);
    } else {
        testNest<false><<< 1, 8 >>> (floatVal);
    }
    return (intVal >= 0 ) ? intVal ^ 0x80000000 : intVal ^ 0xFFFFFFFF;
}
__device__ __forceinline__ float orderedIntToFloat( int intVal ) {
    return __int_as_float( (intVal >= 0) ? intVal ^ 0xFFFFFFFF : intVal ^ 0x80000000);
}

__global__ void test(const float* const v1, float* min_vals) {
    int id = threadIdx.x;
    extern __shared__ int uvs[];
    __shared__ int min_uv[20];
    for (int i = 0; i < 20; i++) {
        int loc = i + 20 * id;
        uvs[loc] = floatToOrderedInt(v1[loc]);
        if (id == 0)
            min_uv[i] = 0xc5000000;
    }
    __syncthreads();
    for (int i = 0; i < 20; i++) {
        int loc = i + 20 * id;
        atomicMin(&min_uv[i], uvs[loc]);
        __syncthreads();
    }
    for (int i = 0; i < 20; i++) {
        if (id > 0) break;
        min_vals[i] = orderedIntToFloat(min_uv[i]);
    }
    __syncthreads();
    printf("%x,\n", floatToOrderedInt(2048.0));
}

void testOmp(float* ptr) {
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < 1024; i++) {
        ptr[i] = ptr[i] * 2;
    }
}

int main() {
    float vals[320];
    for (int i = 15; i > -1; i--) {
        int base = (15 - i) * 20;
        if (i & 1) {
            for (int j = 19; j > -1; j--) {
                vals[base + 19 - j] = float(i * 15) + float(j) + 0.54321;
            }
        } else {
            for (int j = 0; j < 20; j++) {
                vals[base + j] = float(i * 15) + float(j) + 0.54321;
            }
        }
    }
    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 20; j++) {
    //         printf("%.5f,", vals[i * 20 + j]);
    //     }
    //     printf("\n");
    // }
    float *cu_ptr, *min_vals;
    float result[20];
    size_t float_size = 320 * sizeof(float);
    cudaMalloc((void **) &cu_ptr, float_size);
    cudaMalloc((void **) &min_vals, 20 * sizeof(float));
    cudaMemcpy(cu_ptr, vals, float_size, cudaMemcpyHostToDevice);
    test <<< 1, 16, float_size >>> (cu_ptr, min_vals);
    cudaMemcpy(result, min_vals, 20 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cu_ptr);
    cudaFree(min_vals);
    for (int i = 0; i < 20; i++) {
        printf("%.5f, ", result[i]);
    }
    printf("%d\n", -2%180);
    float test_arr[1024];
    testOmp(test_arr);
    printf("Completed, %d\n", sizeof(bool));
    for (int i = 0; i < 33; i++) {
        int a = 432829 + i;
        int b = 0x03;
        int res = a + (4 - a & b);
        printf("result:%d, %d\n", res, res % 4);
    }
    return 0;
}