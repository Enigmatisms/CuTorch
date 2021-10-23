#include <torch/extension.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
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

at::Tensor convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
) {
     
}

at::Tensor convolution_backward_weight(
    const at::Tensor& input,
    c10::ArrayRef<int64_t> weight_size,
    const at::Tensor& grad_output,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {

    return at::cudnn_convolution_backward_weight(
        weight_size,
        grad_output,
        input,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
}

at::Tensor convolution_backward_input(
    c10::ArrayRef<int64_t> input_size,
    const at::Tensor& weight,
    const at::Tensor& grad_output,
    c10::ArrayRef<int64_t> stride,
    c10::ArrayRef<int64_t> padding,
    c10::ArrayRef<int64_t> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {

    return at::cudnn_convolution_backward_input(
        input_size,
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        allow_tf32);
}