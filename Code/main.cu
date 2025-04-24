#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include "Adamath.h"
#include "AiComputations.h"
#include <chrono>
#include <random>

// DONT USE .h in COMPILE COMMAND
//  use rm Ada.exe to remove original .exe file if present, or EXAMPLEFOLDER/Ada.exe
// USE THIS SINCE IM USING CUDA: nvcc -O3 Code/main.cu -o ada
// then .\ada to run


// Utility to generate a random matrix of given dimensions
std::vector<std::vector<float>> generateMatrix(int rows, int cols) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = dist(gen);

    return matrix;
}


int main()
{
    int size = 16384;  // You can scale this up for stress testing

    std::vector<std::vector<float>> A = generateMatrix(size, size);
    std::vector<std::vector<float>> B = generateMatrix(size, size);
    std::vector<std::vector<float>> output;

    std::cout << "Benchmarking CUDA matmul (" << size << "x" << size << ")..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    output = matmulCUDA(A, B);  // Run CUDA accelerated version
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> duration = end - start;
    std::cout << "CUDA matmul time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
