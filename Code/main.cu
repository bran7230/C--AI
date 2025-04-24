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


__global__ void helloCUDA() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main()
{
    helloCUDA<<<1, 5>>>();
    cudaDeviceSynchronize();

    return 0;
}
