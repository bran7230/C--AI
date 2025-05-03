#pragma once
#include "Adamath.cuH"
#include <mma.h>
using namespace nvcuda;
//==============================
//      SOFTMAX
//==============================

std::vector<float> softmax(const std::vector<float> &input)
{
    int size = input.size();
    std::vector<float> output(size);
    float maxVal = *std::max_element(input.begin(), input.end());

    std::vector<float> exps(size);
    for (int i = 0; i < size; ++i)
        exps[i] = std::exp(input[i] - maxVal);

    float sum = std::accumulate(exps.begin(), exps.end(), 0.0f);
    for (int i = 0; i < size; ++i)
        output[i] = exps[i] / sum;

    return output;
}

std::vector<std::vector<float>> softmaxBatch(const std::vector<std::vector<float>> &matrix)
{
    int batchSize = matrix.size();
    std::vector<std::vector<float>> output(batchSize);
#pragma omp parallel for
    for (int i = 0; i < batchSize; ++i)
        output[i] = softmax(matrix[i]);
    return output;
}

__global__ void softmaxSharedKernel(const float *input, float *output, int numCols)
{
    extern __shared__ float rowData[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (tid < numCols)
        rowData[tid] = input[row * numCols + tid];
    __syncthreads();

    float maxVal = -FLT_MAX;
    for (int i = 0; i < numCols; ++i)
        maxVal = fmaxf(maxVal, rowData[i]);
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < numCols; ++i)
    {
        rowData[i] = __expf(rowData[i] - maxVal);
        sum += rowData[i];
    }
    __syncthreads();

    for (int i = 0; i < numCols; ++i)
        output[row * numCols + i] = rowData[i] / sum;
}

void softmaxCUDA_shared(const float *d_input, float *d_output, int batchSize, int numCols)
{
    dim3 grid(batchSize);
    dim3 block(numCols);
    size_t sharedMemSize = numCols * sizeof(float);
    softmaxSharedKernel<<<grid, block, sharedMemSize>>>(d_input, d_output, numCols);
    cudaDeviceSynchronize();
}


std::vector<__half> toHalfFlat(const std::vector<std::vector<float>> &matrix)
{
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<__half> flat(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            flat[i * cols + j] = __float2half(matrix[i][j]);
    return flat;
}

std::vector<__half> toHalf1D(const std::vector<float> &vec)
{
    std::vector<__half> halfVec(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
        halfVec[i] = __float2half(vec[i]);
    return halfVec;
}

// High-performance warp-tiled Tensor Core kernel (16x16 blocks)
__global__ void turboFusedTensorCore_half_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int M, int K, int N)
{
    // Each warp computes a 16x16 tile of C
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    int row = warpM * 16;
    int col = warpN * 16;

    if (row >= M || col >= N) return;

    // Accumulator fragment (FP32)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over K tiles
    for (int t = 0; t < K; t += 16)
    {
        if (t + 15 < K)
        {
            // Load A and B tiles into WMMA fragments
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

            if (row + 15 < M && t + 15 < K)
                wmma::load_matrix_sync(a_frag, A + row * K + t, K);

            if (t + 15 < K && col + 15 < N)
                wmma::load_matrix_sync(b_frag, B + t * N + col, N);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Add bias and write result (coalesced, 1 warp stores 16x16 tile)
    #pragma unroll
    for (int i = 0; i < acc_frag.num_elements; ++i)
    {
        int r = i / 16;
        int c = i % 16;

        int globalRow = row + r;
        int globalCol = col + c;

        if (globalRow < M && globalCol < N)
        {
            float val = acc_frag.x[i];
            val += __half2float(bias[globalCol]);
            output[globalRow * N + globalCol] = __float2half(val);
        }
    }
}

void fusedLinearSoftmaxTensorCore_half(
    const __half* d_input,
    const __half* d_weights,
    const __half* d_bias,
    __half* d_output,
    int batchSize, int inputDim, int outputDim)
{
    dim3 threads(32, 4); // 4 warps per block for higher occupancy
    dim3 blocks((outputDim + 15) / 16, (batchSize + 15) / 16);
    turboFusedTensorCore_half_kernel<<<blocks, threads>>>(
        d_input, d_weights, d_bias, d_output,
        batchSize, inputDim, outputDim);
    cudaDeviceSynchronize();
}

std::vector<std::vector<float>> softmaxCUDA_batch_half(
    const std::vector<std::vector<float>> &input,
    const std::vector<std::vector<float>> &weights,
    const std::vector<float> &bias)
{
    int batchSize = input.size();
    int inputDim = input[0].size();
    int outputDim = weights.size();
    int totalOut = batchSize * outputDim;

    auto flatInput = toHalfFlat(input);
    auto flatWeights = toHalfFlat(weights);
    auto flatBias = toHalf1D(bias);

    __half *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input, flatInput.size() * sizeof(__half));
    cudaMalloc(&d_weights, flatWeights.size() * sizeof(__half));
    cudaMalloc(&d_bias, flatBias.size() * sizeof(__half));
    cudaMalloc(&d_output, totalOut * sizeof(__half));

    cudaMemcpy(d_input, flatInput.data(), flatInput.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, flatWeights.data(), flatWeights.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, flatBias.data(), flatBias.size() * sizeof(__half), cudaMemcpyHostToDevice);

    fusedLinearSoftmaxTensorCore_half(d_input, d_weights, d_bias, d_output, batchSize, inputDim, outputDim);

    std::vector<__half> flatOutput(totalOut);
    cudaMemcpy(flatOutput.data(), d_output, totalOut * sizeof(__half), cudaMemcpyDeviceToHost);

    std::vector<std::vector<float>> result(batchSize, std::vector<float>(outputDim));
    for (int i = 0; i < batchSize; ++i)
        for (int j = 0; j < outputDim; ++j)
            result[i][j] = __half2float(flatOutput[i * outputDim + j]);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);

    return result;
}




