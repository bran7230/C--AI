#pragma once

//========================
//       INCLUDES
//========================
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <immintrin.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cuda_fp16.h>
#include <omp.h>
#include <mma.h>

//========================
//     CUDA KERNELS
//========================
__global__ void relu1d_kernel(float *input, float *output, int size);
__global__ void relu2D_kernel(float *input, float *output, int totalSize);
__global__ void matmul_shared_float4_kernel(const float *A, const float *B, float *C, int M, int K, int N);
__global__ void matmulBiasFusedFloat4Kernel(const float *A, const float *B, const float *bias, float *C, int M, int K, int N);
__global__ void addBiasEfficient(float *output, const float *bias, int batchSize, int outputDim);
__global__ void softmaxSharedKernel(const float *input, float *output, int numCols);

//========================
//     DEVICE WRAPPERS
//========================
void reluCUDA1D(const std::vector<float> &inputVec, std::vector<float> &outputVec);
void reluCUDA_batch(const std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &output);
void matmulBiasFusedFloat4Launch(const float *d_A, const float *d_B, const float *d_bias, float *d_C, int M, int K, int N);
void softmaxCUDA_shared(const float *d_input, float *d_output, int batchSize, int numCols);
std::vector<std::vector<float>> matmulCUDA(const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B);

// TensorCore + Softmax fused kernel forward declaration
void fusedLinearSoftmaxTensorCore_half(const __half *d_input, const __half *d_weights, const __half *d_bias, __half *d_probs, int batchSize, int inputDim, int outputDim);

//========================
//     MATH FUNCTIONS
//========================
std::vector<float> sigmoid(const std::vector<float> &z);
std::vector<std::vector<float>> sigmoidBatch(const std::vector<std::vector<float>> &matrix);
std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>> &matrix);
float dotSIMD(const std::vector<float> &a, const std::vector<float> &b);
std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B);
std::vector<std::vector<float>> linear(const std::vector<std::vector<float>> &input, const std::vector<std::vector<float>> &weights, const std::vector<float> &bias);
std::vector<std::vector<float>> linearCUBLAS(const std::vector<std::vector<float>> &input, const std::vector<std::vector<float>> &weights, const std::vector<float> &bias, cublasHandle_t handle);

std::vector<float> softmax(const std::vector<float> &input);
std::vector<std::vector<float>> softmaxBatch(const std::vector<std::vector<float>> &matrix);
std::vector<std::vector<float>> softmaxCUDA_batch_half(const std::vector<std::vector<float>> &input, const std::vector<std::vector<float>> &weights, const std::vector<float> &bias);

std::vector<__half> toHalfFlat(const std::vector<std::vector<float>> &matrix);

float cross_entropy(const std::vector<float> &probs, int targetIndex);
float cross_entropy(const std::vector<std::vector<float>> &batchProb, const std::vector<int> &targetIndices);
float binary_cross_entropy_batch(const std::vector<std::vector<float>> &predictions, const std::vector<int> &targets);

std::vector<std::vector<float>> sigmoidDerivative(const std::vector<std::vector<float>> &activated);
std::vector<float> computeGradient(const std::vector<float> &probs, int targetId);
std::vector<float> oneHot(int vocabSize, int index);
std::vector<std::vector<float>> computeDW(const std::vector<float> &x, const std::vector<float> &dZ);
