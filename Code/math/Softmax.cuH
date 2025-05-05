#pragma once
#include "Adamath.cuH"
#include <mma.h>
using namespace nvcuda;
//==============================
//      SOFTMAX
//==============================

// Base softmax function, only i batch of inputs.
std::vector<float> softmax(const std::vector<float> &input)
{
    // pre declare size for memory saving
    int size = input.size();
    // make the output we return
    std::vector<float> output(size);
    // let the max value(safe for numbers)
    float maxVal = *std::max_element(input.begin(), input.end());
    // save all the exponents for the softmax equations(1 / 1 + e^-z)
    std::vector<float> exps(size);
    // loop thorugh the exponents, making sure to subtract the max value for number safety.
    for (int i = 0; i < size; ++i)
        exps[i] = std::exp(input[i] - maxVal);

    // make the sum
    float sum = std::accumulate(exps.begin(), exps.end(), 0.0f);
    // loop through the final values, adding the products of the exponents/sum of exponents.
    for (int i = 0; i < size; ++i)
        output[i] = exps[i] / sum;

    return output;
}
// Batch version, parallel enabled for faster ops.
std::vector<std::vector<float>> softmaxBatch(const std::vector<std::vector<float>> &matrix)
{ // pre reserve for perfomance.
    int batchSize = matrix.size();
    std::vector<std::vector<float>> output(batchSize);
#pragma omp parallel for
    for (int i = 0; i < batchSize; ++i)
        // calling orginal softmax
        output[i] = softmax(matrix[i]);
    return output;
}
// CUDA kernel to compute softmax for each row of a 2D matrix in parallel using shared memory
__global__ void softmaxSharedKernel(const float *input, float *output, int numCols)
{
    // Allocate shared memory per block to store one row of input
    extern __shared__ float rowData[];
    // Determine the current row and thread ID within the block
    int row = blockIdx.x;  // Each block processes one row of inputs
    int tid = threadIdx.x; // Each thread processes one collum

    // Load the row's data into shared memory (each thread loads one element)
    // Saftey to check if the threads are greater than the number of colums inputed.
    if (tid < numCols)
        rowData[tid] = input[row * numCols + tid];
    __syncthreads(); // Ensure all threads finish loading before proceeding

    // Compute the max value in the row for numerical stability
    float maxVal = -FLT_MAX;
    for (int i = 0; i < numCols; ++i)
        maxVal = fmaxf(maxVal, rowData[i]);
    __syncthreads(); // Ensure maxVal is available before continuing

    // Subtract max and compute exponentials, accumulate sum
    float sum = 0.0f;
    for (int i = 0; i < numCols; ++i)
    {
        rowData[i] = __expf(rowData[i] - maxVal); // Exponentiate (stabilized)
        sum += rowData[i];                        // Accumulate sum
    }
    __syncthreads(); // Wait until all exponentials and sum are ready

    // Normalize each value to compute softmax
    for (int i = 0; i < numCols; ++i)
        output[row * numCols + i] = rowData[i] / sum; // Softmax output
}

// Wrapper function to launch the softmaxSharedKernel for a batch of inputs
void softmaxCUDA_shared(const float *d_input, float *d_output, int batchSize, int numCols)
{
    // Each block will handle one row (sample) in the batch
    dim3 grid(batchSize);

    // Each thread in a block handles one column in the row
    dim3 block(numCols);

    // Allocate enough shared memory per block to store one row of data
    size_t sharedMemSize = numCols * sizeof(float);

    // Launch the kernel with shared memory to compute row-wise softmax
    softmaxSharedKernel<<<grid, block, sharedMemSize>>>(d_input, d_output, numCols);

    // Synchronize to ensure the kernel finishes before continuing
    cudaDeviceSynchronize();
}

// Converts a 2D float matrix into a 1D __half vector in row-major order
std::vector<__half> toHalfFlat(const std::vector<std::vector<float>> &matrix)
{
    int rows = matrix.size();
    int cols = matrix[0].size();
    // Allocate flat __half vector of size rows * cols
    std::vector<__half> flat(rows * cols);
    // Flatten the matrix row by row, converting each float to half
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            flat[i * cols + j] = __float2half(matrix[i][j]);
    return flat;
}

// Converts a 1D float vector to a 1D __half vector
std::vector<__half> toHalf1D(const std::vector<float> &vec)
{
    // Allocate __half vector with same size
    std::vector<__half> halfVec(vec.size());
    // Convert each float element to half-precision
    for (size_t i = 0; i < vec.size(); ++i)
        halfVec[i] = __float2half(vec[i]);
    return halfVec;
}

// High-performance warp-tiled fused Linear + Bias kernel using Tensor Cores
// Computes C = A × B + bias, where:
// A: [M x K], B: [K x N], bias: [N], output: [M x N]
__global__ void turboFusedTensorCore_half_kernel(
    const __half *__restrict__ A,
    const __half *__restrict__ B,
    const __half *__restrict__ bias,
    __half *__restrict__ output,
    int M, int K, int N)
{
    // Each warp computes a 16x16 tile of C
    int warpM = blockIdx.y; // Tile row index
    int warpN = blockIdx.x; // Tile col index

    int row = warpM * 16;
    int col = warpN * 16;

    // Early exit if out of bounds
    if (row >= M || col >= N)
        return;

    // Accumulator fragment (FP32)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f); // Initialize accumulator with 0

    // Loop over K tiles
    for (int t = 0; t < K; t += 16)
    {
        // Bounds check for valid tile
        if (t + 15 < K)
        {
            // Load A and B tiles into WMMA fragments
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

            // Load 16x16 tile from A if in bounds (row-major layout)
            if (row + 15 < M && t + 15 < K)
                wmma::load_matrix_sync(a_frag, A + row * K + t, K);
            // Load 16x16 tile from B if in bounds (column-major layout)
            if (t + 15 < K && col + 15 < N)
                wmma::load_matrix_sync(b_frag, B + t * N + col, N);

            // Perform matrix multiply: acc_frag += a_frag * b_frag
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

// Add bias and write result (coalesced, 1 warp stores 16x16 tile)
#pragma unroll
    for (int i = 0; i < acc_frag.num_elements; ++i)
    {
        int r = i / 16; // Local row inside tile
        int c = i % 16; // Local col inside tile

        int globalRow = row + r;
        int globalCol = col + c;

        // Store result only if within bounds
        if (globalRow < M && globalCol < N)
        {
            float val = acc_frag.x[i];                             // FP32 accumulator result
            val += __half2float(bias[globalCol]);                  // Add bias (converted to FP32)
            output[globalRow * N + globalCol] = __float2half(val); // Store result as FP16
        }
    }
}

// Wrapper function to launch the fused Tensor Core kernel:
// Performs: output = softmax(A × B + bias)
// Uses warp-tiled WMMA and FP16 math for speed
void fusedLinearSoftmaxTensorCore_half(
    const __half *d_input,   // [batchSize x inputDim] input activations (FP16)
    const __half *d_weights, // [inputDim x outputDim] weight matrix (FP16)
    const __half *d_bias,    // [outputDim] bias vector (FP16)
    __half *d_output,        // [batchSize x outputDim] output buffer (FP16, softmax result)
    int batchSize, int inputDim, int outputDim)
{
    // 32 threads per warp × 4 warps = 128 threads per block
    // Helps increase occupancy on large matrices
    dim3 threads(32, 4);

    // Each block processes one 16×16 tile: (output = batchSize x outputDim)
    dim3 blocks((outputDim + 15) / 16, (batchSize + 15) / 16);

    // Launch the warp-tiled fused Tensor Core kernel
    turboFusedTensorCore_half_kernel<<<blocks, threads>>>(
        d_input, d_weights, d_bias, d_output,
        batchSize, inputDim, outputDim);

    // Ensure kernel completes before moving on
    cudaDeviceSynchronize();
}
// High-performance batched softmax computation using fused Tensor Core kernel (FP16 math)
// Performs: result = softmax(input × weights + bias)
std::vector<std::vector<float>> softmaxCUDA_batch_half(
    const std::vector<std::vector<float>> &input,   // [batchSize x inputDim]
    const std::vector<std::vector<float>> &weights, // [outputDim x inputDim] (transposed layout)
    const std::vector<float> &bias)                 // [outputDim]
{
    int batchSize = input.size();         // Number of input samples
    int inputDim = input[0].size();       // Input feature size
    int outputDim = weights.size();       // Output feature size
    int totalOut = batchSize * outputDim; // Total output elements

    // === Flatten and convert to FP16 ===
    auto flatInput = toHalfFlat(input);     // [batchSize x inputDim]
    auto flatWeights = toHalfFlat(weights); // [outputDim x inputDim]
    auto flatBias = toHalf1D(bias);         // [outputDim]

    // === Allocate device memory ===
    __half *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input, flatInput.size() * sizeof(__half));
    cudaMalloc(&d_weights, flatWeights.size() * sizeof(__half));
    cudaMalloc(&d_bias, flatBias.size() * sizeof(__half));
    cudaMalloc(&d_output, totalOut * sizeof(__half));

    // === Copy host => device ===
    cudaMemcpy(d_input, flatInput.data(), flatInput.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, flatWeights.data(), flatWeights.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, flatBias.data(), flatBias.size() * sizeof(__half), cudaMemcpyHostToDevice);

    // === Launch fused Tensor Core kernel ===
    fusedLinearSoftmaxTensorCore_half(d_input, d_weights, d_bias, d_output, batchSize, inputDim, outputDim);

    // === Copy result device => host ===
    std::vector<__half> flatOutput(totalOut);
    cudaMemcpy(flatOutput.data(), d_output, totalOut * sizeof(__half), cudaMemcpyDeviceToHost);

    // === Convert FP16 output to 2D float32 matrix ===
    std::vector<std::vector<float>> result(batchSize, std::vector<float>(outputDim));
    for (int i = 0; i < batchSize; ++i)
        for (int j = 0; j < outputDim; ++j)
            result[i][j] = __half2float(flatOutput[i * outputDim + j]);

    // === Cleanup ===
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);

    return result;
}
