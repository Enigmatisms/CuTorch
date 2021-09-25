#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>

constexpr int MAT_SIZE = 16;

__global__ void MatMul(const float* const A, const float* const B, float* C) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    // __shared__ memory declared and allocated can be accessed with one block
    __shared__ float shared_a[MAT_SIZE][MAT_SIZE];
    __shared__ float shared_b[MAT_SIZE][MAT_SIZE];
    shared_a[row][col] = A[row * MAT_SIZE + col];
    shared_b[row][col] = B[row * MAT_SIZE + col];
    cudaDeviceSynchronize();
    float c_val = 0.0;
    for (int i = 0; i < MAT_SIZE; i++) {
        c_val += shared_a[row][i] * shared_b[i][col];
    }
    // register is even faster
    C[row * MAT_SIZE + col] = c_val;
    cudaDeviceSynchronize();
}

void printMat(const cv::Mat& src) {
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            printf("%.3f, ", src.at<float>(i, j));
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv) {
    cv::Mat A(MAT_SIZE, MAT_SIZE, CV_32FC1);
    cv::Mat B(MAT_SIZE, MAT_SIZE, CV_32FC1);
    cv::Mat D(MAT_SIZE, MAT_SIZE, CV_32FC1);
    cv::RNG rng(0);
    rng.fill(A, cv::RNG::NORMAL, 0, 1);
    rng.fill(B, cv::RNG::NORMAL, 0, 1);
    cv::Mat C = A * B;
    constexpr size_t mat_size = sizeof(float) * MAT_SIZE * MAT_SIZE;
    float* a_ptr, *b_ptr, *d_ptr;
    cudaMalloc((void **) &a_ptr, mat_size);
    cudaMalloc((void **) &b_ptr, mat_size);
    cudaMalloc((void **) &d_ptr, mat_size);
    cudaMemcpy(a_ptr, A.ptr<float>(), mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr, B.ptr<float>(), mat_size, cudaMemcpyHostToDevice);
    dim3 grid(MAT_SIZE, MAT_SIZE, 1);
    MatMul <<<1, grid>>> (a_ptr, b_ptr, d_ptr);
    cudaMemcpy(D.ptr<float>(), d_ptr, mat_size, cudaMemcpyDeviceToHost);
    cv::Mat res = C - D;
    printMat(A);
    printMat(B);
    printMat(C);
    printMat(D);
    printMat(res);
    cudaFree(a_ptr);
    cudaFree(b_ptr);
    cudaFree(d_ptr);
    return 0;
}