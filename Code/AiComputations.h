#pragma once
#include <vector>
#include <cmath>

//====================================
//  UPDATING WEIGHTS, IN HERE FOR NOW
//====================================

// Applies gradient descent to update the weight matrix W
// Parameters:
// - W: The current weight matrix to be updated (modified in-place)
// - dW: The gradient of the loss with respect to the weights (same shape as W)
// - learningRate: Scalar multiplier that controls how much we adjust each weight
// This function subtracts (learningRate × gradient) from each weight,
// nudging the weights in the direction that reduces the loss.


void updateWeights(std::vector<std::vector<float>> &W, const std::vector<std::vector<float>> &dW, float learningRate)
{
    for (int i = 0; i < W.size(); ++i)
    {
        for (int j = 0; j < W[0].size(); ++j)
        {
            W[i][j] -= learningRate * dW[i][j]; // Apply the weight update
        }
    }
}
/*
    Example code:

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
*/
