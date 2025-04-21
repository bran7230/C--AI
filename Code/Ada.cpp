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
//  use rm Code/Ada.exe to recompile
//  then g++ -O3 Code/Ada.cpp -o Ada.exe
int main()
{
    // example rows
    std::vector<std::vector<float>> inputs = {
        {0.0f, 1.2f},
        {3.5f, 2.0f}};
    // loop through the test inputs, softmaxing each
    auto result = softmaxBatch(inputs);
    // loop through the rows, prints results, then ending line after each row
    for (const auto &val : result)
    {
        for (const auto &i : val)
        {
            std::cout << i << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}
