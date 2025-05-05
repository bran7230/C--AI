#pragma once
#include "Adamath.cuH"
#include <mma.h>
using namespace nvcuda;
//============================
//      RELU MATH
//============================

// CUDA kernel for applying ReLU activation on a 1D array
// Each thread handles one element: output[i] = max(0, input[i])
__global__ void relu1d_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Compute global thread index
    if (idx < size)
    {
        output[idx] = fmaxf(0.0f, input[idx]); // Apply ReLU: zero out negatives
    }
}

// Host function to apply ReLU using CUDA for a 1D vector
void reluCUDA1D(const std::vector<float> &inputVec, std::vector<float> &outputVec)
{
    int size = inputVec.size();
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    // Copy input from host to device
    cudaMemcpy(d_input, inputVec.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel: each thread processes 1 element
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    relu1d_kernel<<<numBlocks, blockSize>>>(d_input, d_output, size);

    // Copy result back to host
    cudaMemcpy(outputVec.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
// CUDA kernel to apply ReLU activation to a 2D matrix represented as a flattened 1D array.
// totalSize = number of elements in the matrix (rows × cols)
// Each thread processes one element: output[i] = max(0, input[i])
__global__ void relu2D_kernel(float *input, float *output, int totalSize)
{
    // Compute the global thread index for 1D launch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only apply if within bounds
    if (idx < totalSize)
    {
        // Apply ReLU activation
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Applies ReLU activation to a 2D matrix using CUDA.
// The input and output are 2D vectors [rows x cols], but internally flattened to 1D for GPU processing.
// Uses relu2D_kernel to process the entire matrix in parallel.
void reluCUDA_batch(const std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &output)
{
    int rows = input.size();     // Number of rows in the matrix
    int cols = input[0].size();  // Number of columns in the matrix
    int totalSize = rows * cols; // Total number of elements

    // Flatten the 2D input into a 1D array for GPU transfer
    std::vector<float> flatInput(totalSize);
    std::vector<float> flatOutput(totalSize);
    for (int i = 0; i < rows; ++i)
        std::copy(input[i].begin(), input[i].end(), flatInput.begin() + i * cols);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, totalSize * sizeof(float));
    cudaMalloc(&d_output, totalSize * sizeof(float));
    // Copy input data from host to device
    cudaMemcpy(d_input, flatInput.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel with enough threads to cover all elements
    int blockSize = 256;
    int numBlocks = (totalSize + blockSize - 1) / blockSize;
    relu2D_kernel<<<numBlocks, blockSize>>>(d_input, d_output, totalSize);

    // Copy the result back from device to host
    cudaMemcpy(flatOutput.data(), d_output, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    // Reconstruct the 2D output matrix from the flattened result
    output.resize(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        std::copy(flatOutput.begin() + i * cols, flatOutput.begin() + (i + 1) * cols, output[i].begin());
}
