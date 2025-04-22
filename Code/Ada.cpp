#include <iostream>
#include <cmath>
#include <vector>
#include "Adamath.h"
#include "AiComputations.h"
// DONT USE .h in COMPILE COMMAND
//  use rm Ada.exe to remove original .exe file if present, or EXAMPLEFOLDER/Ada.exe
// USE THIS SINCE IM USING OPENMP: g++ -fopenmp -O3 Code/Ada.cpp -o Ada.exe

int main()
{

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
  
    return 0;
}
