#pragma once
#include "Adamath.cuH"
#include <mma.h>
using namespace nvcuda;
//============================
//      RELU MATH
//============================
#define TILE_SIZE 32
__global__ void relu1d_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

void reluCUDA1D(const std::vector<float> &inputVec, std::vector<float> &outputVec)
{
    int size = inputVec.size();
    float *d_input, *d_output;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, inputVec.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    relu1d_kernel<<<numBlocks, blockSize>>>(d_input, d_output, size);

    cudaMemcpy(outputVec.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void relu2D_kernel(float *input, float *output, int totalSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalSize)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

void reluCUDA_batch(const std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &output)
{
    int rows = input.size();
    int cols = input[0].size();
    int totalSize = rows * cols;

    std::vector<float> flatInput(totalSize);
    std::vector<float> flatOutput(totalSize);
    for (int i = 0; i < rows; ++i)
        std::copy(input[i].begin(), input[i].end(), flatInput.begin() + i * cols);

    float *d_input, *d_output;
    cudaMalloc(&d_input, totalSize * sizeof(float));
    cudaMalloc(&d_output, totalSize * sizeof(float));

    cudaMemcpy(d_input, flatInput.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (totalSize + blockSize - 1) / blockSize;
    relu2D_kernel<<<numBlocks, blockSize>>>(d_input, d_output, totalSize);

    cudaMemcpy(flatOutput.data(), d_output, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    output.resize(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        std::copy(flatOutput.begin() + i * cols, flatOutput.begin() + (i + 1) * cols, output[i].begin());
}
