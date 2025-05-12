#pragma once
#include "Syntari.cuH"
#include <mma.h>
using namespace nvcuda;

//=============================
//       LINEAR MATH
//=============================

// Applies a linear transformation (matrix multiplication + bias addition) followed by ReLU activation.
// input: [batch_size x input_dim]
// weights: [input_dim x output_dim]
// bias: [output_dim]
// returns: output = ReLU(input * weights + bias)
std::vector<std::vector<float>> linear(const std::vector<std::vector<float>> &input,
                                       const std::vector<std::vector<float>> &weights,
                                       const std::vector<float> &bias)
{

    // Perform matrix multiplication: output = input * weights

    auto output = matmul(input, weights);
    int batch_size = output.size();    // Number of samples in the batch
    int output_dim = output[0].size(); // Dimensionality of each output sample

    // Add bias to each output element in parallel (OpenMP)
#pragma omp parallel for
    for (int i = 0; i < batch_size; ++i)
        for (int j = 0; j < output_dim; ++j)
            output[i][j] += bias[j];
    // Apply ReLU activation using CUDA across the batch
    reluCUDA_batch(output, output);
    // Return the final activated output
    return output;
}

// CUDA kernel to efficiently add bias to a 2D output matrix stored in row-major order.
// Each thread handles one element: output[row][col] += bias[col]
// Parameters:
//   - output: flattened [batchSize x outputDim] matrix (row-major)
//   - bias:   1D array of size [outputDim]
//   - batchSize: number of rows in the output matrix
//   - outputDim: number of columns in the output matrix
__global__ void addBiasEfficient(float *output, const float *bias, int batchSize, int outputDim)
{
    // Compute the row and column this thread will process
    int row = blockIdx.y * blockDim.y + threadIdx.y; // vertical index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // horizontal index
                                                     // Ensure thread is within bounds of the output matrix
    if (row < batchSize && col < outputDim)
    {
        // Flattened index in the output array (row-major order)
        int idx = row * outputDim + col;
        // Add bias to the corresponding element
        output[idx] += bias[col];
    }
}

// Performs a linear transformation using cuBLAS for matrix multiplication,
// adds bias using a CUDA kernel, and returns the output matrix.
// input: [batchSize x inputDim]
// weights: [outputDim x inputDim] (note: transposed for cuBLAS)
// bias: [outputDim]
// handle: cuBLAS handle
// returns: output = input * weights^T + bias
std::vector<std::vector<float>> linearCUBLAS(
    const std::vector<std::vector<float>> &input,
    const std::vector<std::vector<float>> &weights,
    const std::vector<float> &bias,
    cublasHandle_t handle)
{
    // Dimensions
    const int batchSize = input.size();
    const int inputDim = input[0].size();
    const int outputDim = weights.size();
    // Flattened sizes
    const size_t inputSize = batchSize * inputDim;
    const size_t weightSize = outputDim * inputDim;
    const size_t biasSize = bias.size();
    const size_t outputSize = batchSize * outputDim;

    // Flatten input and weights into row-major 1D arrays
    std::vector<float> flatInput, flatWeights;
    flatInput.reserve(inputSize);
    flatWeights.reserve(weightSize);

    for (const auto &row : input)
        flatInput.insert(flatInput.end(), row.begin(), row.end());

    for (const auto &row : weights)
        flatWeights.insert(flatWeights.end(), row.begin(), row.end());

    std::vector<float> flatBias = bias;
    std::vector<float> flatOutput(outputSize); // Flattened output buffer

    // Allocate device memory
    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMallocAsync(&d_input, inputSize * sizeof(float), 0);
    cudaMallocAsync(&d_weights, weightSize * sizeof(float), 0);
    cudaMallocAsync(&d_bias, biasSize * sizeof(float), 0);
    cudaMallocAsync(&d_output, outputSize * sizeof(float), 0);

    // Copy host => device
    cudaMemcpyAsync(d_input, flatInput.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_weights, flatWeights.data(), weightSize * sizeof(float), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_bias, flatBias.data(), biasSize * sizeof(float), cudaMemcpyHostToDevice, 0);

    // Matrix multiplication using cuBLAS:
    // d_output = alpha * (d_weights^T * d_input) + beta * d_output
    // => outputDim x batchSize result
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N, // Transpose weights, normal input
        outputDim, batchSize, inputDim,
        &alpha,
        d_weights, inputDim, // d_weights^T: [outputDim x inputDim]
        d_input, inputDim,   // d_input:     [inputDim x batchSize]
        &beta,
        d_output, outputDim); // d_output:    [outputDim x batchSize]

    // Launch bias addition kernel: each thread adds bias[col] to output[row][col]
    dim3 threads(16, 16);
    dim3 blocks((outputDim + 15) / 16, (batchSize + 15) / 16);
    addBiasEfficient<<<blocks, threads>>>(d_output, d_bias, batchSize, outputDim);
    // Copy result back to host
    cudaMemcpyAsync(flatOutput.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0); // Wait for all GPU operations to complete

    // Convert flat output back into 2D vector format
    std::vector<std::vector<float>> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
        output[i] = std::vector<float>(flatOutput.begin() + i * outputDim, flatOutput.begin() + (i + 1) * outputDim);

    // Free GPU memory
    cudaFreeAsync(d_input, 0);
    cudaFreeAsync(d_weights, 0);
    cudaFreeAsync(d_bias, 0);
    cudaFreeAsync(d_output, 0);

    return output;
}