#include "../include/conv_cpp.h"
#include "../include/conv_kernel.h"

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line <<
			std::endl;
	exit (1);
}

/// input is already in GPU
void convolutionForward(
    const at::Tensor& img,
    const at::Tensor& weights,
    const at::Tensor& bias,
    int in_chan, int out_chan, int k_size,
    at::Tensor& out
) {
    const float* const img_data = img.data_ptr<float>();
    const float* const w_data = weights.data_ptr<float>();
    const float* const b_data = bias.data_ptr<float>();
    dim3 grid(img.size(0), img.size(2), img.size(3));
    dim3 block(img.size(2), k_size, k_size);
    convolutionForward <<<grid, block, out_chan>>>(img_data, w_data, out_chan, output.data_ptr<float>());
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    biasForward <<<grid, out_chan>>> (img_data, b_data, output.data_ptr<float>());
}