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
    // Test weights
    std::vector<std::vector<float>> weights = {
        {0.5f, 0.3f, -0.2f},
        {0.1f, -0.4f, 0.7f}
    };

    // Fake gradients (same shape as weights)
    std::vector<std::vector<float>> dW = {
        {0.1f, -0.2f, 0.05f},
        {-0.3f, 0.1f, -0.15f}
    };
    // Default learning rate
    float learningRate = 0.1f;
   
    std::cout << "Weights before update: " << std::endl;
    for(const auto& row : weights)
    {
        for(float val : row)
        {
            std::cout << val << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << "\nAfter: " << std::endl;
    updateWeights(weights, dW, learningRate);
    for(const auto& row : weights)
    {
        for(float val : row)
        {
            std::cout << val << "\t";
        }
        std::cout << std::endl;
    }

    

  

    
    return 0;
}
