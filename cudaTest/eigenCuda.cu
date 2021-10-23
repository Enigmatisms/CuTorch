#include <chrono>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef Eigen::Matrix<double, 16, 12> MatrixXd;

__global__ void matMul(const double const* src1, const double const* src2, double* dst) {
    int y = blockIdx.x, x = threadIdx.x;
    int rows = gridDim.x, cols = blockDim.x;
    int id = y * cols + x;
    dst[id] = src1[id] * src2[id];
}

__global__ void eigenDirect(const MatrixXd* const src1, const MatrixXd* const src2, MatrixXd* dst) {
    int y = blockIdx.x, x = threadIdx.x;
    int rows = gridDim.x, cols = blockDim.x;
    dst->operator()(y, x) = src1->operator()(y, x) * src2->operator()(y, x);
}


__device__ double determinant(const Eigen::Matrix2d& mat) {
    return mat(0, 0) * mat(1, 1) - mat(1, 0) * mat(0, 1);
}

__device__ Eigen::Matrix2d matInverse(Eigen::Matrix2d mat) {
    double tmp1 = mat(0, 0), tmp2 = mat(1, 0);
    double det = determinant(mat);
    if (abs(det) > 1e-5) {
        mat(0, 0) = mat(1, 1);
        mat(1, 0) = -mat(0, 1);
        mat(0, 1) = -tmp2;
        mat(1, 1) = tmp1;
        return mat / det;
    }
    return Eigen::Matrix2d::Zero();
}

__global__ void streamEigenProcess(const Eigen::Matrix2d* const src1, const Eigen::Vector2d* const src2, Eigen::Vector2d* dst) {
    int id = threadIdx.x;
    Eigen::Matrix2d inv = matInverse(src1[id]);
    dst[id] = inv * src2[id];
}

int main() {
    // ======================= test 1 ========================
    int row = 16, col = 12;
    Eigen::Matrix<double, 16, 12> A;
    A.setRandom();
    Eigen::Matrix<double, 16, 12> B;
    B.setRandom();
    Eigen::Matrix<double, 16, 12> C;
    C.setZero();
    double *dev_a, *dev_b, *dev_c;
    size_t mat_size = sizeof(double) * row * col;
    cudaMalloc((void **) &dev_a, mat_size);
    double start_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e6;
    cudaMalloc((void **) &dev_b, mat_size);
    cudaMalloc((void **) &dev_c, mat_size);
    double end_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e6;
    printf("malloc time: %lf ms\n", end_t - start_t);
    cudaMemcpy(dev_a, A.data(), mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, B.data(), mat_size, cudaMemcpyHostToDevice);
    matMul <<< row, col >>> (dev_a, dev_b, dev_c);
    cudaMemcpy(C.data(), dev_c, mat_size, cudaMemcpyDeviceToHost);
    // std::cout << A << std::endl << std::endl;
    // std::cout << B << std::endl << std::endl;
    // std::cout << C << std::endl << std::endl;
    start_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e6;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    end_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e6;
    printf("free time: %lf ms\n", end_t - start_t);

    // ======================= test 2 ========================
    // Eigen::Matrix<double, 16, 12> *mat_a, *mat_b, *mat_c;
    // size_t eigen_size = sizeof(A);
    // cudaMalloc((void **) &mat_a, eigen_size);
    // cudaMalloc((void **) &mat_b, eigen_size);
    // cudaMalloc((void **) &mat_c, eigen_size);
    // C.setZero();
    // cudaMemcpy(mat_a->data(), A.data(), mat_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(mat_b->data(), B.data(), mat_size, cudaMemcpyHostToDevice);
    // eigenDirect <<< row, col >>> (mat_a, mat_b, mat_c);
    // cudaMemcpy(C.data(), mat_c->data(), mat_size, cudaMemcpyDeviceToHost);
    // std::cout << C << std::endl << std::endl;
    // cudaFree(mat_a);
    // cudaFree(mat_b);
    // cudaFree(mat_c);

    // std::vector<Eigen::Matrix2d> mats;
    // std::vector<Eigen::Vector2d> vecs;
    // std::vector<Eigen::Vector2d> result(128, Eigen::Vector2d::Zero());
    // for (int i = 0; i < 128; i++) {
    //     mats.push_back(Eigen::Matrix2d::Random());
    //     vecs.push_back(Eigen::Vector2d::Random());
    // }
    // Eigen::Matrix2d* mat_ptr;
    // Eigen::Vector2d* vec_ptr, *res_ptr;
    // size_t mats_size = sizeof(Eigen::Matrix2d) * 128;
    // size_t vecs_size = sizeof(Eigen::Vector2d) * 128;
    // cudaMalloc((void **) &mat_ptr, mats_size);
    // cudaMalloc((void **) &vec_ptr, vecs_size);
    // cudaMalloc((void **) &res_ptr, vecs_size);
    // cudaMemcpy(mat_ptr, mats.data(), mats_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(vec_ptr, vecs.data(), vecs_size, cudaMemcpyHostToDevice);
    // streamEigenProcess <<< 1, 128 >>> (mat_ptr, vec_ptr, res_ptr);
    // cudaMemcpy(result.data(), res_ptr, vecs_size, cudaMemcpyDeviceToHost);
    // cudaFree(mat_ptr);
    // cudaFree(vec_ptr);
    // cudaFree(res_ptr);
    // for (int i = 0; i < 128; i++) {
    //     std::cout << result[i] << std::endl << std::endl;
    // }
    return 0;
}