#include "Adamath.cuH"
#include <cuda_fp16.h>
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

//==============================
//      SIGMOID MATH
//==============================

std::vector<float> sigmoid(const std::vector<float> &z)
{
    std::vector<float> output;
    output.reserve(z.size());
    for (float val : z)
        output.push_back(1 / (1 + std::exp(-val)));
    return output;
}

std::vector<std::vector<float>> sigmoidBatch(const std::vector<std::vector<float>> &matrix)
{
    std::vector<std::vector<float>> output(matrix.size());
#pragma omp parallel for
    for (int i = 0; i < matrix.size(); ++i)
        output[i] = sigmoid(matrix[i]);
    return output;
}


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

// SIMD-optimized Dot Product
float dotSIMD(const std::vector<float> &a, const std::vector<float> &b)
{
    int size = a.size();
    int i = 0;
    float result = 0.0f;

    __m128 sum = _mm_setzero_ps();
    for (; i <= size - 4; i += 4)
    {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 prod = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, prod);
    }

    float temp[4];
    _mm_storeu_ps(temp, sum);
    result = temp[0] + temp[1] + temp[2] + temp[3];

    for (; i < size; ++i)
        result += a[i] * b[i];

    return result;
}

// Matrix Multiplication (CPU fallback, SIMD + OpenMP)
std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>> &A,
                                       const std::vector<std::vector<float>> &B)
{
    int aRows = A.size();
    int aCols = A[0].size();
    int bCols = B[0].size();

    auto B_T = transpose(B);

    std::vector<std::vector<float>> output(aRows, std::vector<float>(bCols, 0.0f));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < aRows; ++i)
        for (int j = 0; j < bCols; ++j)
            output[i][j] = dotSIMD(A[i], B_T[j]);

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

    if (warpM * 16 >= M || warpN * 16 >= N) return; // Bounds check

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
                                  const half *d_A,    // (batchSize x inputDim)
                                  const half *d_B,    // (inputDim x outputDim)
                                  const float *d_bias,// (outputDim)
                                  float *d_C,         // (batchSize x outputDim)
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

//============================
//   LOSS / GRADIENT UTILS
//============================

float cross_entropy(const std::vector<float> &probs, int targetIndex)
{
    return -std::log(probs[targetIndex]);
}

float cross_entropy(const std::vector<std::vector<float>> &batchProb, const std::vector<int> &targetIndices)
{
    float totalLoss = 0.0f;
#pragma omp parallel for reduction(+ : totalLoss)
    for (int i = 0; i < batchProb.size(); i++)
        totalLoss += cross_entropy(batchProb[i], targetIndices[i]);
    return totalLoss / batchProb.size();
}

float binary_cross_entropy_batch(const std::vector<std::vector<float>> &predictions, const std::vector<int> &targets)
{
    float totalLoss = 0.0f;
#pragma omp parallel for reduction(+ : totalLoss)
    for (int i = 0; i < predictions.size(); ++i)
    {
        float p = predictions[i][0];
        float y = targets[i];
        totalLoss -= (y * std::log(p) + (1 - y) * std::log(1 - p));
    }
    return totalLoss / predictions.size();
}

std::vector<std::vector<float>> sigmoidDerivative(const std::vector<std::vector<float>> &activated)
{
    std::vector<std::vector<float>> output(activated.size());
#pragma omp parallel for
    for (int i = 0; i < activated.size(); ++i)
    {
        const auto &row = activated[i];
        std::vector<float> derivedRow(row.size());
        for (int j = 0; j < row.size(); ++j)
            derivedRow[j] = row[j] * (1.0f - row[j]);
        output[i] = derivedRow;
    }
    return output;
}

std::vector<float> computeGradient(const std::vector<float> &probs, int targetId)
{
    std::vector<float> dZ = probs;
    dZ[targetId] -= 1.0f;
    return dZ;
}

std::vector<float> oneHot(int vocabSize, int index)
{
    std::vector<float> vec(vocabSize, 0.0f);
    vec[index] = 1.0f;
    return vec;
}

std::vector<std::vector<float>> computeDW(const std::vector<float> &x, const std::vector<float> &dZ)
{
    int input_size = x.size();
    int output_size = dZ.size();
    std::vector<std::vector<float>> dW(input_size, std::vector<float>(output_size));
    for (int i = 0; i < input_size; ++i)
        for (int j = 0; j < output_size; ++j)
            dW[i][j] = x[i] * dZ[j];
    return dW;
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
