#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include "Adamath.cuH"
#include "AiComputations.h"
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
// DONT USE .h in COMPILE COMMAND
//  use rm Ada.exe to remove original .exe file if present, or EXAMPLEFOLDER/Ada.exe
// USE THIS SINCE IM USING CUDA:  nvcc -O3 -arch=sm_70 -lcublas -o ada Code/main.cu 
// then .\ada to run

void benchmarkMatmulCUBLAS_GPUOnly_Batched(int batchSize, int inputDim, int outputDim, int runs = 10)
{
    std::cout << "\n[Benchmark] matmulCUBLAS_GPUOnly_Batched" << std::endl;
    std::cout << "Batch: " << batchSize << ", InputDim: " << inputDim << ", OutputDim: " << outputDim << "\n";

    size_t sizeA = batchSize * inputDim;
    size_t sizeB = inputDim * outputDim;
    size_t sizeC = batchSize * outputDim;
    size_t sizeBias = outputDim;

    // Random inputs
    std::vector<half> h_A(sizeA), h_B(sizeB);
    std::vector<float> h_bias(sizeBias);

    for (auto &x : h_A) x = __float2half(static_cast<float>(rand()) / RAND_MAX);
    for (auto &x : h_B) x = __float2half(static_cast<float>(rand()) / RAND_MAX);
    for (auto &x : h_bias) x = static_cast<float>(rand()) / RAND_MAX;

    half *d_A, *d_B;
    float *d_C, *d_bias;
    cudaMalloc(&d_A, sizeA * sizeof(half));
    cudaMalloc(&d_B, sizeB * sizeof(half));
    cudaMalloc(&d_C, sizeC * sizeof(float));
    cudaMalloc(&d_bias, sizeBias * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), sizeBias * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Warm-up
    matmulCUBLAS_GPUOnly_Batched(handle, d_A, d_B, d_bias, d_C, batchSize, inputDim, outputDim);
    cudaDeviceSynchronize();

    float totalMs = 0;
    for (int i = 0; i < runs; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();

        matmulCUBLAS_GPUOnly_Batched(handle, d_A, d_B, d_bias, d_C, batchSize, inputDim, outputDim);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        float elapsedMs = std::chrono::duration<float, std::milli>(end - start).count();
        totalMs += elapsedMs;
    }

    float avgMs = totalMs / runs;
    float ops = 2.0f * batchSize * inputDim * outputDim;
    float gflops = (ops / (avgMs / 1000.0f)) / 1e9;

    std::cout << "Average Time: " << avgMs << " ms\n";
    std::cout << "Performance: " << gflops << " GFLOPS (";
    std::cout << gflops / 1000.0f << " TFLOPS)" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_bias);
    cublasDestroy(handle);
}

int main()
{
    benchmarkMatmulCUBLAS_GPUOnly_Batched(16384, 16384, 16384);

 
    return 0;
}
