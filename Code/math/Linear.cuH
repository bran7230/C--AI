#pragma once
#include "Adamath.cuH"
#include <mma.h>
using namespace nvcuda;

//=============================
//       LINEAR MATH
//=============================

std::vector<std::vector<float>> linear(const std::vector<std::vector<float>> &input,
    const std::vector<std::vector<float>> &weights,
    const std::vector<float> &bias)
    {
    auto output = matmul(input, weights);
    int batch_size = output.size();
    int output_dim = output[0].size();
    
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i)
    for (int j = 0; j < output_dim; ++j)
    output[i][j] += bias[j];
    
    reluCUDA_batch(output, output);
    
    return output;
    }
    
    __global__ void addBiasEfficient(float *output, const float *bias, int batchSize, int outputDim)
    {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batchSize && col < outputDim)
    {
    int idx = row * outputDim + col;
    output[idx] += bias[col];
    }
    }
    
    std::vector<std::vector<float>> linearCUBLAS(
    const std::vector<std::vector<float>> &input,
    const std::vector<std::vector<float>> &weights,
    const std::vector<float> &bias,
    cublasHandle_t handle)
    {
    const int batchSize = input.size();
    const int inputDim = input[0].size();
    const int outputDim = weights.size();
    
    const size_t inputSize = batchSize * inputDim;
    const size_t weightSize = outputDim * inputDim;
    const size_t biasSize = bias.size();
    const size_t outputSize = batchSize * outputDim;
    
    std::vector<float> flatInput, flatWeights;
    flatInput.reserve(inputSize);
    flatWeights.reserve(weightSize);
    for (const auto &row : input) flatInput.insert(flatInput.end(), row.begin(), row.end());
    for (const auto &row : weights) flatWeights.insert(flatWeights.end(), row.begin(), row.end());
    
    std::vector<float> flatBias = bias;
    std::vector<float> flatOutput(outputSize);
    
    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMallocAsync(&d_input, inputSize * sizeof(float), 0);
    cudaMallocAsync(&d_weights, weightSize * sizeof(float), 0);
    cudaMallocAsync(&d_bias, biasSize * sizeof(float), 0);
    cudaMallocAsync(&d_output, outputSize * sizeof(float), 0);
    
    cudaMemcpyAsync(d_input, flatInput.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_weights, flatWeights.data(), weightSize * sizeof(float), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_bias, flatBias.data(), biasSize * sizeof(float), cudaMemcpyHostToDevice, 0);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    outputDim, batchSize, inputDim,
    &alpha,
    d_weights, inputDim,
    d_input, inputDim,
    &beta,
    d_output, outputDim);
    
    dim3 threads(16, 16);
    dim3 blocks((outputDim + 15) / 16, (batchSize + 15) / 16);
    addBiasEfficient<<<blocks, threads>>>(d_output, d_bias, batchSize, outputDim);
    
    cudaMemcpyAsync(flatOutput.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);
    
    std::vector<std::vector<float>> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
    output[i] = std::vector<float>(flatOutput.begin() + i * outputDim, flatOutput.begin() + (i + 1) * outputDim);
    
    cudaFreeAsync(d_input, 0);
    cudaFreeAsync(d_weights, 0);
    cudaFreeAsync(d_bias, 0);
    cudaFreeAsync(d_output, 0);
    
    return output;
    }