#pragma once
#include "Adamath.cuH"
#include <mma.h>
using namespace nvcuda;
//==============================================
//                CPU MATRIX MATH
//==============================================

// Matrix Transpose (Parallelized)
std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>> &matrix)
{
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            transposed[j][i] = matrix[i][j];

    return transposed;
}

// Computes the dot product of two vectors using SIMD intrinsics for performance.
// This version uses SSE (__m128) to compute 4 products per iteration.
// Falls back to scalar computation for the remaining elements if size is not a multiple of 4.
float dotSIMD(const std::vector<float> &a, const std::vector<float> &b)
{
    int size = a.size(); // Assume a and b are the same size
    int i = 0;
    float result = 0.0f;

    __m128 sum = _mm_setzero_ps(); // Initialize SIMD accumulator to 0
                                   // Process 4 elements at a time using SIMD
    for (; i <= size - 4; i += 4)
    {
        __m128 va = _mm_loadu_ps(&a[i]);  // Load 4 floats from a
        __m128 vb = _mm_loadu_ps(&b[i]);  // Load 4 floats from b
        __m128 prod = _mm_mul_ps(va, vb); // Multiply element-wise
        sum = _mm_add_ps(sum, prod);      // Accumulate the result
    }

    // Horizontal sum of the SIMD register
    float temp[4];
    _mm_storeu_ps(temp, sum);                       // Store SIMD sum to temp array
    result = temp[0] + temp[1] + temp[2] + temp[3]; // Sum the four partial results

    // Process any remaining elements (scalar fallback)
    for (; i < size; ++i)
        result += a[i] * b[i];

    return result;
}
// Matrix Multiplication with CPU fallback using SIMD (dotSIMD) and OpenMP for multithreading.
// A: [aRows x aCols]
// B: [aCols x bCols]
// Returns: output matrix of shape [aRows x bCols]
// Matrix Multiplication (CPU fallback, SIMD + OpenMP)
std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>> &A,
                                       const std::vector<std::vector<float>> &B)
{
    int aRows = A.size();    // Number of rows in matrix A
    int aCols = A[0].size(); // Number of columns in matrix A
    int bCols = B[0].size(); // Number of columns in matrix B
                             // Transpose B to improve cache locality during dot product access
    auto B_T = transpose(B); // B_T: [bCols x aCols]

    // Allocate result matrix with dimensions [aRows x bCols], initialized to 0
    std::vector<std::vector<float>> output(aRows, std::vector<float>(bCols, 0.0f));
    // Parallelized nested loop over output matrix using OpenMP
    // Each thread computes one output cell using SIMD-optimized dot product
#pragma omp parallel for collapse(2)
    for (int i = 0; i < aRows; ++i)
        for (int j = 0; j < bCols; ++j)
            output[i][j] = dotSIMD(A[i], B_T[j]); // Compute dot product of row A[i] and column B[:,j]

    return output;
}

///////////////////////////////////////////////////////////
//                 CUDA MATMUL TENSOR CORE
///////////////////////////////////////////////////////////

//==============================================
// CUDA Kernel: Tensor Core Matmul + Fused Bias
// (FP16 multiply, FP32 accumulate)
//==============================================
__global__ void matmulBiasTensorCoreKernel(const half *A, const half *B, const half *bias,
                                           float *C, int M, int K, int N)
{
    // Warp and thread indices
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (warpM * 16 >= M || warpN * 16 >= N)
        return; // Bounds check

    // Create fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accFrag;

    wmma::fill_fragment(accFrag, 0.0f); // Initialize accumulator to zero

    // Loop over K dimension tiles
    for (int k = 0; k < K; k += 16)
    {
        if (k + 16 <= K)
        {
            const half *tileA = A + warpM * 16 * K + k;
            const half *tileB = B + k * N + warpN * 16;

            // Load tiles into fragments
            wmma::load_matrix_sync(aFrag, tileA, K);
            wmma::load_matrix_sync(bFrag, tileB, N);

            // Perform matrix multiply-accumulate
            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }
    }

    // Define output pointer for this warp's tile
    float *tileC = C + warpM * 16 * N + warpN * 16;

    // Store accumulator results with fused bias addition
#pragma unroll
    for (int i = 0; i < accFrag.num_elements; ++i)
    {
        int row = (i / 16);
        int col = (i % 16);
        int globalRow = warpM * 16 + row;
        int globalCol = warpN * 16 + col;

        if (globalRow < M && globalCol < N)
        {
            float biasVal = __half2float(bias[globalCol]);
            tileC[row * N + col] = accFrag.x[i] + biasVal;
        }
    }
}

__global__ void addBiasBatchKernel(float *output, const float *bias, int batchSize, int outputDim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Output dimension

    if (row < batchSize && col < outputDim)
    {
        int idx = row * outputDim + col;
        output[idx] += bias[col];
    }
}

// GPU-Only Batched cuBLAS MatMul + Bias (No memcpy, No malloc inside)
void matmulCUBLAS_GPUOnly_Batched(cublasHandle_t handle,
                                  const half *d_A,     // (batchSize x inputDim)
                                  const half *d_B,     // (inputDim x outputDim)
                                  const float *d_bias, // (outputDim)
                                  float *d_C,          // (batchSize x outputDim)
                                  int batchSize, int inputDim, int outputDim)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Matrix multiply: C = A x B
    // d_A: (batchSize x inputDim)
    // d_B: (inputDim x outputDim)
    // d_C: (batchSize x outputDim)
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 outputDim, batchSize, inputDim,
                 &alpha,
                 d_B, CUDA_R_16F, outputDim,
                 d_A, CUDA_R_16F, inputDim,
                 &beta,
                 d_C, CUDA_R_32F, outputDim,
                 CUBLAS_COMPUTE_32F_FAST_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Add bias
    dim3 blockDim(16, 16);
    dim3 gridDim((outputDim + 15) / 16, (batchSize + 15) / 16);
    addBiasBatchKernel<<<gridDim, blockDim>>>(d_C, d_bias, batchSize, outputDim);
}
// Performs matrix multiplication using a custom CUDA kernel with shared memory and float4 optimization.
// A: [M x K], B: [K x N], returns C = A * B as [M x N]
std::vector<std::vector<float>> matmulCUDA(const std::vector<std::vector<float>> &A,
                                           const std::vector<std::vector<float>> &B)
{
    int M = A.size();    // Number of rows in matrix A
    int K = A[0].size(); // Number of columns in A, also rows in B
    int N = B[0].size(); // Number of columns in matrix B

    // Output matrix C [M x N], initialized with 0s
    std::vector<std::vector<float>> C(M, std::vector<float>(N, 0.0f));

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float)); // Matrix A on device
    cudaMalloc(&d_B, K * N * sizeof(float)); // Matrix B on device
    cudaMalloc(&d_C, M * N * sizeof(float)); // Output matrix C on device

    // Flatten input matrices A and B to 1D row-major arrays
    std::vector<float> flatA, flatB;
    for (const auto &row : A)
        flatA.insert(flatA.end(), row.begin(), row.end());
    for (const auto &row : B)
        flatB.insert(flatB.end(), row.begin(), row.end());

    // Copy A and B to GPU memory
    cudaMemcpy(d_A, flatA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions for 2D kernel launch
    dim3 block(16, 16);                      // 16x16 threads per block
    dim3 grid((N + 15) / 16, (M + 15) / 16); // Enough blocks to cover MxN output

    // Launch shared memory optimized CUDA kernel (e.g., using float4 and tiling)
    matmul_shared_float4_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize(); // Wait for GPU to finish computation

    // Retrieve output from GPU
    std::vector<float> flatC(M * N);
    cudaMemcpy(flatC.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert flat result into 2D matrix format
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] = flatC[i * N + j];

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

// Shared memory optimized CUDA kernel for matrix multiplication (C = A * B)
// Each thread block computes one TILE x TILE tile of the output matrix C
// A: [M x K], B: [K x N], C: [M x N]
// TILE = 16 by default(FOR PERFOMANCE USE TILE = 32 INSTEAD OF 16)
__global__ void matmul_shared_float4_kernel(const float *A, const float *B, float *C, int M, int K, int N)
{
    const int TILE = 16; // INCREASE TO 32 IF YOUR GPU CAN HANDLE IT

    // Declare shared memory tiles for a block of A and B
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    // Compute global row and column index this thread will compute
    int row = blockIdx.y * TILE + threadIdx.y; // Row in C
    int col = blockIdx.x * TILE + threadIdx.x; // Column in C

    float value = 0.0f; // Accumulator for the output element

    // Loop over all tiles of A and B needed to compute C[row][col]
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        // Load one element from A into shared memory (if in bounds)
        if (row < M && t * TILE + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load one element from B into shared memory (if in bounds)
        if (col < N && t * TILE + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Wait for all threads to finish loading into shared memory
        __syncthreads();

        // Compute partial dot product for this tile
        for (int i = 0; i < TILE; ++i)
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        // Wait again before loading the next tile
        __syncthreads();
    }

    // Write the final value to output matrix C if within bounds
    if (row < M && col < N)
        C[row * N + col] = value;
}
