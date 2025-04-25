#include "Adamath.h"

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

//==============================
//      MATRIX MATH
//==============================

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

//============================
//      CUDA MATMUL
//============================

__global__ void matmul_shared_float4_kernel(const float *A, const float *B, float *C, int M, int K, int N)
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
        tileA[threadIdx.y][threadIdx.x] = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
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
__global__ void matmulBiasFusedFloat4Kernel(const float *A, const float *B, const float *bias,
    float *C, int M, int K, int N)
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
#pragma unroll
for (int k = 0; k < TILE_SIZE; ++k)
value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
__syncthreads();
}

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

__global__ void dummyKernel() {}

void fusedLinearSoftmaxTensorCore_half(
    const __half* d_input,
    const __half* d_weights,
    const __half* d_bias,
    __half* d_probs,
    int batchSize, int inputDim, int outputDim)
{
    // TEMP: just zero memory for now or use cudaMemset
    cudaMemset(d_probs, 0, batchSize * outputDim * sizeof(__half));
}
