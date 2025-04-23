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


std::vector<std::vector<float>> generateLargeInput(int rows = 10000, int cols = 512) {
    std::vector<std::vector<float>> data(rows, std::vector<float>(cols));
    std::mt19937 rng(42);  // Seeded RNG
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &row : data)
        for (auto &val : row)
            val = dist(rng);
    return data;
}
int main()
{
    auto input = generateLargeInput();

    auto start = std::chrono::high_resolution_clock::now();
    auto result = softmaxBatch(input);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "C++ softmaxBatch time: " << elapsed.count() << " seconds\n";

    return 0;
}
