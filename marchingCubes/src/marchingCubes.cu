#include "../include/marchingCubes.h"
#ifdef CUDA_TEST
    #include <torch/torch.h>
    #include <torch/script.h>
#else
    #include <torch/extension.h>
#endif

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line <<
			std::endl;
	exit (1);
}

const std::array<Vertex, 16> table = {
    Vertex{}, Vertex{{0, 3}}, Vertex{{1, 0}}, Vertex{{1, 3}},
    Vertex{{1, 2}}, Vertex{{2, 1}, {0, 3}}, Vertex{{2, 0}}, Vertex{{2, 3}},
    Vertex{{3, 2}}, Vertex{{0, 2}}, Vertex{{3, 2}, {1, 0}}, Vertex{{1, 2}},
    Vertex{{3, 1}}, Vertex{{0, 1}}, Vertex{{3, 0}}, Vertex{}
};

const std::array<Point2d, 4> vertices = {
    Point2d(0, 1), Point2d(1, 1), Point2d(1, 0), Point2d(0, 0)
};

constexpr int width = 400, height = 300;
constexpr size_t sdf_size = sizeof(float) * width * height, tag_size = sizeof(uint8_t) * (width - 1) * (height - 1);

/// marching square should be the complete procedure
__host__ at::Tensor marchingSquare(at::Tensor bubbles) {
    at::Tensor lut = at::zeros({height, width});
    size_t bubble_size = sizeof(float) * 3 * bubbles.size(0);
    float *bubble_data, *sdf_data;
    uint8_t tags[tag_size], *tag_data;

    cudaMalloc((void **) &bubble_data, bubble_size);
    cudaMalloc((void **) &sdf_data, sdf_size);
    cudaMemcpy(bubble_data, (void *)bubbles.data_ptr<float>(), bubble_size, cudaMemcpyHostToDevice);
    dim3 sdf_grid(width);
    calculateSDF <<< sdf_grid, height >>> (bubble_data, bubbles.size(0), sdf_data);
    cudaMalloc((void **) &tag_data, tag_size);
    dim3 tag_grid(width);
    gridTagsCalculation <<< tag_grid, height - 1 >>> (sdf_data, tag_data);
    cudaMemcpy(lut.data_ptr<float>(), sdf_data, sdf_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(tags, tag_data, tag_size, cudaMemcpyDeviceToHost);
    Edges dst;
    float* lut_ptr = lut.data_ptr<float>();
    for (int i = 0; i < height - 1; i++) {
        for (int j = 0; j < width - 1; j++) {
            const int base0 = i * width;
            const uint8_t tag = tags[base0 - i + j];
            // if (tag == 0x00 || tag == 0x0f) continue;
            const Vertex& vtx = table[tag];
            const int base1 = base0 + width;
            std::array<float, 4> vals = {
                abs(lut_ptr[base0 + j]),
                abs(lut_ptr[base0 + j + 1]),
                abs(lut_ptr[base1 + j + 1]),
                abs(lut_ptr[base1 + j])
            };
            linearInterp(vtx, vals, Point2d(j, i), dst);
        }
    }
    int size = dst.size();
    at::Tensor edges = at::zeros({size, 6});
    for (int i = 0; i < (int)dst.size(); i++) {
        const EdgeInterpret interpret(dst[i]);
        for (int j = 0; j < 6; j++) {
            edges.index_put_({i, j}, interpret.vals[j]);
        }
    }
    cudaFree(bubble_data);
    cudaFree(sdf_data);
    cudaFree(tag_data);
    return edges;
}

__host__ void linearInterp(const Vertex& vtx, const std::array<float, 4>& abs_vals, const Point2d offset, Edges& edges) {
    for (size_t i = 0; i < vtx.size(); i++) {
        const int v11 = vtx[i].first, v21 = vtx[i].second;
        const int v12 = (v11 + 1) % 4, v22 = (v21 + 1) % 4;
        const float off1 = abs_vals[v11] / (abs_vals[v11] + abs_vals[v12]), off2 = abs_vals[v21] / (abs_vals[v21] + abs_vals[v22]);
        Edge edge;
        edge.sp = (vertices[v12] - vertices[v11]) * off1;
        edge.ep = (vertices[v22] - vertices[v21]) * off2;
        edge.offset = offset;
        edges.push_back(edge);
    }
}

#ifndef CUDA_TEST
PYBIND11_MODULE (march, m)
{
    m.def ("marchingSquare", &marchingSquare, "marching square bubbles (CUDA)");
}
#endif