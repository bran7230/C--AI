#include <iostream>
#include <cmath>
#include <vector>
#include "Adamath.h"
#include "AiComputations.h"
// DONT USE .h in COMPILE COMMAND
//  use rm Ada.exe to remove original .exe file if present, or EXAMPLEFOLDER/Ada.exe
// USE THIS SINCE IM USING OPENMP: g++ -fopenmp -O3 Code/main.cpp -o Ada.exe

int main() {


//example rows
std::vector<std::vector<float>> inputs = {
    {0.0f, 1.2f},
    {3.5f, 2.0f}
};
//loop through the test inputs, softmaxing each
auto result = softmaxBatch(inputs);
//loop through the rows, prints results, then ending line after each row
for(const auto& val:result)
{
    for(const auto& i : val)
    {
        std::cout<<i << "\t";
    }
    std::cout<<std::endl;
}
   
    
  

    return 0;
}
