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
// USE THIS SINCE IM USING OPENMP: g++ -fopenmp -O3 -march=native Code/main.cpp -o Ada.exe


std::vector<std::vector<float>> generateMatrix(int rows, int cols) {
    std::vector<std::vector<float>> mat(rows, std::vector<float>(cols));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& row : mat)
        for (auto& val : row)
            val = dist(rng);
    return mat;
}
int main()
{
    auto A = generateMatrix(2048, 2048);
    auto B = generateMatrix(2048, 2048);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = matmul(A, B);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "C++ matmul time: " << elapsed.count() << " seconds\n";

    return 0;
}
