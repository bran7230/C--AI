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
    const int size = 16384;

    std::cout << "Benchmarking CPU matmul (" << size << "x" << size << ")...\n";

    // Create large input matrices
    std::vector<std::vector<float>> A(size, std::vector<float>(size, 1.0f));
    std::vector<std::vector<float>> B(size, std::vector<float>(size, 1.0f));

    // Time the matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    auto result = matmul(A, B);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> duration = end - start;
    std::cout << "CPU matmul time: " << duration.count() << " seconds\n";

    return 0;
}
