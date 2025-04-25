#pragma once
#include <cmath>
#include <vector>
#include <immintrin.h> // For SIMD intrinsics like _mm_max_ps, _mm_loadu_ps
#include <numeric>     // for std::accumulate
#include <algorithm>   // This is required for std::max_element
#include <cublas_v2.h>
#include <cuda_runtime.h>
//============================
//      RELU MATH
//============================

/*
    WHEN YOU SEE printMatrix() IN EXAMPLE CODE ITS REFERING TO THIS BELOW:
*/
// Print for matrixes
// Its easier to do this than loop every time i test, just a simple printer.
void printMatrix(const std::vector<std::vector<float>> &matrix)
{
    // loops through rows
    for (const auto &row : matrix)
    {
        // goes through the values, and prints them
        for (float val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}
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

// Kernel that applies ReLU to each element in a flattened 2D matrix
__global__ void relu2D_kernel(float *input, float *output, int totalSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    if (idx < totalSize)
    {
        output[idx] = fmaxf(0.0f, input[idx]); // ReLU: max(0, val)
    }
}

void reluCUDA_batch(const std::vector<std::vector<float>> &input,
                    std::vector<std::vector<float>> &output)
{
    int rows = input.size();
    int cols = input[0].size();
    int totalSize = rows * cols;

    // Step 1: Flatten the 2D input matrix to a 1D array
    std::vector<float> flatInput(totalSize);
    std::vector<float> flatOutput(totalSize);
    for (int i = 0; i < rows; ++i)
        std::copy(input[i].begin(), input[i].end(), flatInput.begin() + i * cols);

    // Step 2: Allocate GPU memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, totalSize * sizeof(float));
    cudaMalloc(&d_output, totalSize * sizeof(float));

    // Step 3: Copy input data from host (CPU) to device (GPU)
    cudaMemcpy(d_input, flatInput.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice);

    // Step 4: Launch CUDA kernel with 1 thread per matrix element
    int blockSize = 256;
    int numBlocks = (totalSize + blockSize - 1) / blockSize;
    relu2D_kernel<<<numBlocks, blockSize>>>(d_input, d_output, totalSize);

    // Step 5: Copy the result back from device (GPU) to host (CPU)
    cudaMemcpy(flatOutput.data(), d_output, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 6: Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Step 7: Reshape the flat 1D output into a 2D matrix structure
    output.resize(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        std::copy(flatOutput.begin() + i * cols, flatOutput.begin() + (i + 1) * cols, output[i].begin());
}

// example running for relu:
/*
 std::vector<std::vector<float>> input = {
    { -1.0f, 0.0f, 2.0f },
    { 3.5f, -0.5f, 1.0f }
};

std::vector<std::vector<float>> output;
reluCUDA_batch(input, output);

printMatrix(output);

*/

//==============================
//      SIGMOID MATH
//==============================

// Confidence or sigmoid function
// Added vectors
std::vector<float> sigmoid(const std::vector<float> &z)
{

    // Return Sigmoid function ( 1 / 1 + E ^-z)
    std::vector<float> output;
    output.reserve(z.size());

    // Loop through values and add Sigmoid vals
    for (float val : z)
    {
        output.push_back(1 / (1 + (std::exp(-val))));
    }
    return output;
}
// sigmoid matrix, same thing as relu, loop through rows, apply sigmoid per row.
std::vector<std::vector<float>> sigmoidBatch(const std::vector<std::vector<float>> &matrix)
{
    // declare a vector and save its size
    std::vector<std::vector<float>> output(matrix.size());
#pragma omp parallel for
    for (int i = 0; i < matrix.size(); ++i)
    {
        output[i] = sigmoid(matrix[i]);
    }

    return output;
}
// example running for sigmoid:
/*
 std::vector<std::vector<float>> testMatrix = {
        { -1.0f, 0.0f, 2.0f },
        { 3.5f, -0.5f, 1.0f }
    };

    auto res = sigmoid(testMatrix);
    std::cout << "Sigmoid test:\n ";
    printMatrix(res);
*/
//==============================
//      MATRIX MATH
//==============================
// Matrix multiplication:
// Transposes a 2D matrix (rows become columns)
// Used to make matrix B cache-friendly for SIMD row-wise access
std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>> &matrix)
{
    int rows = matrix.size();
    int cols = matrix[0].size();

    // Create transposed matrix: [cols x rows]
    std::vector<std::vector<float>> transposed(cols, std::vector<float>(rows));

    // Parallel transpose using OpenMP
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            transposed[j][i] = matrix[i][j];

    return transposed;
}

// Computes the dot product between two float vectors using SIMD acceleration
float dotSIMD(const std::vector<float> &a, const std::vector<float> &b)
{
    int size = a.size();
    int i = 0;
    float result = 0.0f;

    __m128 sum = _mm_setzero_ps(); // Initialize SIMD register to [0, 0, 0, 0]

    // Process 4 elements at a time using SSE instructions
    for (; i <= size - 4; i += 4)
    {
        __m128 va = _mm_loadu_ps(&a[i]);  // Load 4 floats from vector a
        __m128 vb = _mm_loadu_ps(&b[i]);  // Load 4 floats from vector b
        __m128 prod = _mm_mul_ps(va, vb); // Multiply element-wise
        sum = _mm_add_ps(sum, prod);      // Accumulate the sum
    }

    // Horizontally add the 4 floats from the SIMD register
    float temp[4];
    _mm_storeu_ps(temp, sum);
    result = temp[0] + temp[1] + temp[2] + temp[3];

    // Handle leftover elements (if size isn't divisible by 4)
    for (; i < size; ++i)
        result += a[i] * b[i];

    return result;
}

// Performs matrix multiplication: output = A × B
// Uses transposed B for better memory access, and SIMD for faster inner loop
std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>> &A,
                                       const std::vector<std::vector<float>> &B)
{
    int aRows = A.size();    // Rows in A
    int aCols = A[0].size(); // Columns in A
    int bCols = B[0].size(); // Columns in B

    // Transpose B once so we can SIMD dot with its "columns"
    auto B_T = transpose(B);

    // Allocate result matrix: [aRows x bCols]
    std::vector<std::vector<float>> output(aRows, std::vector<float>(bCols, 0.0f));

// Parallel multiply: one thread per output[i][j]
#pragma omp parallel for collapse(2)
    for (int i = 0; i < aRows; ++i)
    {
        for (int j = 0; j < bCols; ++j)
        {
            // Compute dot product between A[i] and transposed B[j]
            output[i][j] = dotSIMD(A[i], B_T[j]);
        }
    }

    return output;
}

//============================
//      MATMUL CUDA VERSION
//============================
#define TILE_SIZE 32

// CUDA kernel for matrix multiplication using shared memory tiles
// Computes: C = A × B where
// A is (M x K), B is (K x N), C is (M x N)
__global__ void matmul_shared_float4_kernel(const float *__restrict__ A,
                                            const float *__restrict__ B,
                                            float *__restrict__ C,
                                            int M, int K, int N)
{
    // Shared memory tiles for block-sized chunks of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Calculate the row and column this thread is responsible for
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f; // Accumulator for the dot product

    // Loop over tiles in K-dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        // Global indices for A and B tiles
        int tiledCol = t * TILE_SIZE + threadIdx.x; // For A
        int tiledRow = t * TILE_SIZE + threadIdx.y; // For B

        // Load A tile element into shared memory (if within bounds)
        if (row < M && tiledCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tiledCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile element into shared memory (if within bounds)
        if (tiledRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Wait for all threads to finish loading before compute
        __syncthreads();

// Dot product of the row of A and column of B using TILE_SIZE steps
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        // Wait before loading next tile
        __syncthreads();
    }

    // Write the computed value to the output matrix (C)
    if (row < M && col < N)
        C[row * N + col] = value;
}

std::vector<std::vector<float>> matmulCUDA(const std::vector<std::vector<float>> &A,
                                           const std::vector<std::vector<float>> &B)
{
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();

    std::vector<float> A_flat(M * K);
    std::vector<float> B_flat(K * N);
    std::vector<float> C_flat(M * N);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            A_flat[i * K + j] = A[i][j];

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            B_flat[i * N + j] = B[i][j];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_flat.size() * sizeof(float));
    cudaMalloc(&d_B, B_flat.size() * sizeof(float));
    cudaMalloc(&d_C, C_flat.size() * sizeof(float));

    cudaMemcpy(d_A, A_flat.data(), A_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), B_flat.size() * sizeof(float), cudaMemcpyHostToDevice);

    // NEW (shared memory optimized)
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_shared_float4_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C_flat.data(), d_C, C_flat.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<std::vector<float>> C(M, std::vector<float>(N));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] = C_flat[i * N + j];

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

#define TILE_SIZE 32

__global__ void matmulBiasFusedFloat4Kernel(const float *__restrict__ A,
                                            const float *__restrict__ B,
                                            const float *__restrict__ bias,
                                            float *__restrict__ C,
                                            int M, int K, int N)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;

        // --- SAFE float4 LOAD for A ---
        if (K % 4 == 0 && (tiledCol + 3 < K) && (row < M))
        {
            uintptr_t addrA = reinterpret_cast<uintptr_t>(&A[row * K + tiledCol]);
            if (addrA % 16 == 0)
            {
                const float4 *vecA = reinterpret_cast<const float4 *>(&A[row * K + tiledCol]);
                float4 loadA = *vecA;
                tileA[threadIdx.y][threadIdx.x + 0] = loadA.x;
                tileA[threadIdx.y][threadIdx.x + 1] = loadA.y;
                tileA[threadIdx.y][threadIdx.x + 2] = loadA.z;
                tileA[threadIdx.y][threadIdx.x + 3] = loadA.w;
            }
            else
            {
                tileA[threadIdx.y][threadIdx.x] = (tiledCol < K) ? A[row * K + tiledCol] : 0.0f;
            }
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] = (tiledCol < K && row < M) ? A[row * K + tiledCol] : 0.0f;
        }

        // --- SAFE float4 LOAD for B ---
        if (K % 4 == 0 && (tiledRow + 3 < K) && (col < N))
        {
            uintptr_t addrB = reinterpret_cast<uintptr_t>(&B[tiledRow * N + col]);
            if (addrB % 16 == 0)
            {
                const float4 *vecB = reinterpret_cast<const float4 *>(&B[tiledRow * N + col]);
                float4 loadB = *vecB;
                tileB[threadIdx.y + 0][threadIdx.x] = loadB.x;
                tileB[threadIdx.y + 1][threadIdx.x] = loadB.y;
                tileB[threadIdx.y + 2][threadIdx.x] = loadB.z;
                tileB[threadIdx.y + 3][threadIdx.x] = loadB.w;
            }
            else
            {
                tileB[threadIdx.y][threadIdx.x] = (tiledRow < K) ? B[tiledRow * N + col] : 0.0f;
            }
        }
        else
        {
            tileB[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0f;
        }

        __syncthreads();

// Dot product of row A and column B
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    // Write result with fused bias
    if (row < M && col < N)
        C[row * N + col] = value + bias[col];
}

void matmulBiasFusedFloat4Launch(const float *d_A, const float *d_B, const float *d_bias,
                                 float *d_C, int M, int K, int N)
{
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmulBiasFusedFloat4Kernel<<<blocks, threads>>>(d_A, d_B, d_bias, d_C, M, K, N);
    cudaDeviceSynchronize();
}

// example running of matrix multiply code:
/*
     std::vector<std::vector<float>> A = {
        {0.5, 0.3, 0.2},
        {0.2, 0.6,0.2},
        {0.1, 0.4, 0.5}
    };

    std::vector<std::vector<float>> B = {
        {2, 1},
        {0, 3},
        {1, 2}
    };

    auto result = matmul(A, B);

    printMatrix(result);

*/

//=============================
//       LINEAR MATH
//=============================
// Applies a linear layer: output = ReLU(input * weights + bias)

std::vector<std::vector<float>> linear(const std::vector<std::vector<float>> &input,
                                       const std::vector<std::vector<float>> &weights,
                                       const std::vector<float> &bias)
{
    auto output = matmul(input, weights); // [batch x output_dim]
    int batch_size = output.size();
    int output_dim = output[0].size();

// Add bias
#pragma omp parallel for
    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = 0; j < output_dim; ++j)
        {
            output[i][j] += bias[j];
        }
    }

    // Use CUDA ReLU
    reluCUDA_batch(output, output);

    return output;
}

// Example running:
/*
std::vector<std::vector<float>> input = {
        {1.0, 2.0, 3.0}
    };

    std::vector<std::vector<float>> weights = {
        {0.1, 0.2},
        {0.3, 0.4},
        {0.5, 0.6}
    };

    std::vector<float> bias = {0.5, 1.0};

    auto result = linear(input, weights, bias);

    // Print result
    printMatrix(result);

*/



//======================
//  LINEAR CUBLAS
//======================

__global__ void addBiasEfficient(float *output, const float *bias, int batchSize, int outputDim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output column

    if (row < batchSize && col < outputDim)
    {
        int idx = row * outputDim + col;
        output[idx] += bias[col]; // bias is broadcasted across rows
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
    const int outputDim = weights.size(); // weights = [outputDim][inputDim]

    const size_t inputSize = batchSize * inputDim;
    const size_t weightSize = outputDim * inputDim;
    const size_t biasSize = bias.size();
    const size_t outputSize = batchSize * outputDim;

    // Flatten input in one pass (avoid nested loops)
    std::vector<float> flatInput;
    flatInput.reserve(inputSize);
    for (const auto &row : input)
        flatInput.insert(flatInput.end(), row.begin(), row.end());

    std::vector<float> flatWeights;
    flatWeights.reserve(weightSize);
    for (const auto &row : weights)
        flatWeights.insert(flatWeights.end(), row.begin(), row.end());

    std::vector<float> flatBias = bias;
    std::vector<float> flatOutput(outputSize);

    // Allocate GPU memory
    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMallocAsync(&d_input, inputSize * sizeof(float), 0);
    cudaMallocAsync(&d_weights, weightSize * sizeof(float), 0);
    cudaMallocAsync(&d_bias, biasSize * sizeof(float), 0);
    cudaMallocAsync(&d_output, outputSize * sizeof(float), 0);

    // Async memory transfers
    cudaMemcpyAsync(d_input, flatInput.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_weights, flatWeights.data(), weightSize * sizeof(float), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_bias, flatBias.data(), biasSize * sizeof(float), cudaMemcpyHostToDevice, 0);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // SGEMM: C = A * Bᵗ (A = d_input, B = d_weights)
    cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        outputDim, batchSize, inputDim,
        &alpha,
        d_weights, inputDim,
        d_input, inputDim,
        &beta,
        d_output, outputDim);

    // Bias add kernel
    dim3 threads(16, 16);
    dim3 blocks((outputDim + 15) / 16, (batchSize + 15) / 16);
    addBiasEfficient<<<blocks, threads>>>(d_output, d_bias, batchSize, outputDim);

    // Copy output back
    cudaMemcpyAsync(flatOutput.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0); // wait for everything to finish

    // Reconstruct 2D output efficiently
    std::vector<std::vector<float>> output(batchSize);
    for (int i = 0; i < batchSize; ++i)
        output[i] = std::vector<float>(flatOutput.begin() + i * outputDim,
                                       flatOutput.begin() + (i + 1) * outputDim);

    // Free GPU memory (asynchronous)
    cudaFreeAsync(d_input, 0);
    cudaFreeAsync(d_weights, 0);
    cudaFreeAsync(d_bias, 0);
    cudaFreeAsync(d_output, 0);

    return output;
}

//================================
//     SOFTMAX MATH/SOFTMAX BATCH
//================================

// Converts a vector into a probability distribution using softmax (numerically stable)
std::vector<float> softmax(const std::vector<float> &input)
{
    int size = input.size();
    std::vector<float> output(size);

    // Find the max value for numerical stability
    float maxVal = *std::max_element(input.begin(), input.end());

    // Compute exponentials of (x - maxVal)
    std::vector<float> exps(size);
    for (int i = 0; i < size; ++i)
    {
        exps[i] = std::exp(input[i] - maxVal);
    }

    // Sum of exponentials
    float sum = std::accumulate(exps.begin(), exps.end(), 0.0f);

    // Normalize
    for (int i = 0; i < size; ++i)
    {
        output[i] = exps[i] / sum;
    }

    return output;
}

// Applies softmax row-wise on a batch matrix using OpenMP
std::vector<std::vector<float>> softmaxBatch(const std::vector<std::vector<float>> &matrix)
{
    int batchSize = matrix.size();
    std::vector<std::vector<float>> output(batchSize);

#pragma omp parallel for
    for (int i = 0; i < batchSize; ++i)
    {
        output[i] = softmax(matrix[i]);
    }

    return output;
}
// example running:
/*
    //example rows
    std::vector<std::vector<float>> inputs = {
        {0.0f, 1.2f},
        {3.5f, 2.0f}
    };
    //loop through the test inputs, softmaxing each
    auto result = softmaxBatch(inputs);
    //loop through the rows, prints results, then ending line after each row
    for(const auto& val:result)
    {
        for(const auto& i : val)
        {
            std::cout<<i << "\t";
        }
        std::cout<<std::endl;
    }
*/

//=================================
// CROSS ENTROPY MATH
//=================================

// Computes the cross-entropy loss between predicted probabilities and the correct class index
float cross_entropy(const std::vector<float> &probs, int targetIndex)
{
    // Get the probability the model assigned to the correct class
    float correctProb = probs[targetIndex];

    // Compute the negative log of that probability
    float loss = -std::log(correctProb);

    return loss;
}

float cross_entropy(const std::vector<std::vector<float>> &batchProb, const std::vector<int> &targetIndices)
{
    float totalLoss = 0.0f;
#pragma omp parallel for reduction(+ : totalLoss)
    for (int i = 0; i < batchProb.size(); i++)
    {
        totalLoss += cross_entropy(batchProb[i], targetIndices[i]);
    }

    return totalLoss / batchProb.size();
}
// example code for crossEntropy calc or Foward pass + loss system:
/*

    std::vector<std::vector<float>> batch = {
        softmax({2.0f, 1.0f, 0.1f}),  // Should be confident in class 0
        softmax({0.5f, 2.5f, 0.3f}),  // Should be confident in class 1
        softmax({0.1f, 0.2f, 3.0f})   // Should be confident in class 2
    };

    std::vector<int> targets = {0, 1, 2};

    float loss = cross_entropy(batch, targets);
    std::cout << "Batch Cross-Entropy Loss: " << loss << std::endl;
*/

//=====================================
//    SIGMOID DERIVATIVE MATH(BATCH)
//=====================================
std::vector<std::vector<float>> sigmoidDerivative(const std::vector<std::vector<float>> &activated)
{
    std::vector<std::vector<float>> output;

    output.resize(activated.size());

#pragma omp parallel for
    for (int i = 0; i < activated.size(); ++i)
    {
        const auto &row = activated[i];
        std::vector<float> derivedRow(row.size());

        for (int j = 0; j < row.size(); ++j)
        {
            float val = row[j];
            derivedRow[j] = val * (1.0f - val);
        }

        output[i] = derivedRow;
    }

    return output;
}
// Example running:
/*
 //test set, 2 rows and cols
     std::vector<std::vector<float>> test = {
        {0.8f, 0.1f},
        {0.5f, 0.9f}};
    //result is passing through sigmoid function
    auto result = sigmoidDerivative(test);
        //through rows
    for (auto &row : result)
    {   //results from rows
        for (float val : row)
        {   //print results
            std::cout << val << " ";
        }

    }
*/

//==================================
// BINARY CROSS ENTROPY MATH
//==================================

// Binary cross entropy loss:
// Useful for 0s and 1s
float binary_cross_entropy_batch(const std::vector<std::vector<float>> &predictions, const std::vector<int> &targets)
{
    // total loss
    float totalLoss = 0.0f;
    // go through predictions vectors, add predict as p[i] first val, then targets through the target vectors.
#pragma omp parallel for reduction(+ : totalLoss)
    for (int i = 0; i < predictions.size(); ++i)
    {
        float p = predictions[i][0];
        float y = targets[i];
        totalLoss -= (y * std::log(p) + (1 - y) * std::log(1 - p));
    }

    // finally return the loss
    return totalLoss / predictions.size();
}

// Example code run:
/*
 std::vector<std::vector<float>> predictions = {
        {0.1f}, {0.9f}, {0.8f}, {0.2f}
    };

    std::vector<int> targets = {0, 1, 1, 0};

    float loss = binary_cross_entropy_batch(predictions, targets);
    std::cout << loss << std::endl;
*/

//=====================================
//      GRADIENT COMPUTATIONS
//=====================================
/*

Gradient computations:

A vector that tells how to change each logit

Positive = increase prediction

Negative = decrease

Example:

probs:      [0.1, 0.7, 0.2]
target_id:  1

dZ:         [0.1, -0.3, 0.2]

It was too confident in 0.7, so it lowered the vectors number.
*/
std::vector<float> computeGradient(const std::vector<float> &probs, int targetId)
{
    std::vector<float> dZ = probs; // copy the predicted prbabilities
    dZ[targetId] -= 1.0f;          // Subtract 1 from the correct class
    return dZ;
}
/*
//Future me, heres an example running:

    std::vector<float> probs = {0.1, 0.7, 0.2};
    int target = 1;
    auto dz = computeGradient(probs, target);
    for(float val : dz)
    {
        std::cout<<val <<std::endl;
    }

*/

//============================================
//      ONE-HOT INPUT TOKEN MATH
//============================================
std::vector<float> oneHot(int vocabSize, int index)
{
    /*
        We map the values, and only have the letter I decide to activate active.
        then I set the actives letters index to 1 instead of default 0.
        Then return it
    */
    std::vector<float> vec(vocabSize, 0.0f);
    vec[index] = 1.0f;
    return vec;
}

//=========================================
//  COMPUTING DELTA WEIGHTS OR DW
//=========================================

std::vector<std::vector<float>> computeDW(const std::vector<float> &x, const std::vector<float> &dZ)
{
    int input_size = x.size();
    int output_size = dZ.size();

    std::vector<std::vector<float>> dW(input_size, std::vector<float>(output_size));

    for (int i = 0; i < input_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            dW[i][j] = x[i] * dZ[j];
        }
    }

    return dW;
}
/*
   //Example testing code for the matrixes

    std::vector<float> input = {0.0f, 0.0f, 1.0f, 0.0f};

     // Fake softmax gradient (output error dZ)
     std::vector<float> dZ = {0.1f, -0.3f, 0.2f, 0.0f};

     // Compute dW
     auto dW = computeDW(input, dZ);

     //loop through rows
     for(const auto& row: dW)
     {
     //display values in matrix format ie: 0 0 0 0
     //                                    1 1 1 1
         for(float val :row)
         {
             std::cout<<val<<"\t";
         }
         std::cout<<"\n";
     }
*/