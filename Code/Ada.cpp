#include <iostream>
#include <cmath>
#include <vector>
#include "Adamath.h"
#include "AiComputations.h"
#include <chrono>
#include <random>
// DONT USE .h in COMPILE COMMAND
//  use rm Ada.exe to remove original .exe file if present, or EXAMPLEFOLDER/Ada.exe
//  then g++ -O3 Code/Ada.cpp -o Ada.exe
// BUT FOR MATIRX OPS USE THIS SINCE IM USING OPENMP: g++ -fopenmp -O3 Code/Ada.cpp -o Ada.exe
std::vector<std::vector<float>> generateRandomMatrix(int rows, int cols) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = dist(rng);

    return matrix;
}

void benchmarkReLU(int rows, int cols) {
    auto matrix = generateRandomMatrix(rows, cols);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = relu(matrix);  // your OpenMP relu
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "ReLU (" << rows << "x" << cols << ") took " << duration.count() << " seconds.\n";
}
void benchmarkSoftmaxBatch(int rows, int cols) {
    auto matrix = generateRandomMatrix(rows, cols);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = softmaxBatch(matrix);  // your OpenMP softmaxBatch
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "SoftmaxBatch (" << rows << "x" << cols << ") took " << duration.count() << " seconds.\n";
}
int main()
{
    benchmarkReLU(100000, 512);
    benchmarkSoftmaxBatch(100000, 512);
    benchmarkReLU(1000000, 512);  // Try larger sizes too
    benchmarkSoftmaxBatch(1000000, 512);
  
    return 0;
}
