#include <iostream>
#include <cmath>
#include <vector>
#include "Adamath.h"
#include "AiComputations.h"
#include <sstream>
#include <fstream>
#include <string>
#include <unordered_map>
// DONT USE .h in COMPILE COMMAND
//  use rm Ada.exe to remove original .exe file if present, or EXAMPLEFOLDER/Ada.exe
//  then g++ -O3 Code/Ada.cpp -o Ada.exe
int main()
{
    std::vector<std::vector<float>> output= {
        {0.707, 0.707, 1.414}
    };
    auto result = softmaxBatch(output);

    for(const auto& val : result)
    {
        for(auto& i : val)
        {
            std::cout<< i <<" ";
        }
    }
    return 0;
}
