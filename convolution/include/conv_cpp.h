#include <torch/torch.h>
#include <torch/extension.h>

void convolutionForward(
    const at::Tensor& img,
    const at::Tensor& weights,
    const at::Tensor& bias,
    int in_chan, int out_chan, int k_size,
    at::Tensor& out
);