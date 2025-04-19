#include <iostream>
#include <cmath>
#include <vector>
#include "Adamath.h"
#include "AiComputations.h"
#include <sstream>
#include <fstream>
#include <string>
#include <unordered_map>

int main()
{

    std::vector<std::vector<float>> testMatrix = {
        {-1.0f, 0.0f, 2.0f},
        {3.5f, -0.5f, 1.0f}};

    std::vector<std::vector<float>> result = relu(testMatrix);

    std::vector<std::vector<float>> inputs = sigmoid(result);
    std::vector<std::vector<float>> outputs;
    std::vector<float> rowints;

    for (const auto &vec : result)
    {
        rowints.clear(); 
        for (float x : vec)
        {
            if (x > 0)
                rowints.push_back(x);
        }
        outputs.push_back(rowints);
    }

    std::vector<std::vector<float>> sigmoidOut; 
    for (const auto &in : outputs)
    {
            sigmoidOut.push_back(sigmoid(in));   
    }

    for(const auto& val : sigmoidOut)
    {
        for(float res : val)
        {
            std::cout<<res << "\t";
        }
    }
    return 0;
}
