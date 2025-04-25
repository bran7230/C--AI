#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include "Adamath.h"
#include "AiComputations.h"
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
// DONT USE .h in COMPILE COMMAND
//  use rm Ada.exe to remove original .exe file if present, or EXAMPLEFOLDER/Ada.exe
// USE THIS SINCE IM USING CUDA: nvcc -O3 -lcublas Code/main.cu -o ada
// then .\ada to run


int main()
{
    std::vector<float> input = {1.0f, -2.0f, 3.5f, -4.0f};
    std::vector<float> output(4);

    reluCUDA1D(input, output);

    std::cout << "ReLU Output: ";
    for (float v : output)
        std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}
