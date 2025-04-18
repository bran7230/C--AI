#pragma once
#include <iostream>
#include <cmath>
#include <vector>

//------------------------------------------
// RELU MATH
//------------------------------------------

// original relu to test and loop through rows.
std::vector<float> relu(const std::vector<float> &input)
{
    std::vector<float> output;
    output.reserve(input.size());

    for (float val : input)
    {
        output.push_back(val < 0.0f ? 0.0f : val);
    }

    return output;
}
// Matrix ReLU: Applies relu() to each row and returns the activated matrix
std::vector<std::vector<float>> relu(const std::vector<std::vector<float>> &value)
{
    // vectorize input or value:
    std::vector<std::vector<float>> output;
    // reserve memory or space for perfomance
    output.reserve(value.size());

    // loop through values(for matrixes ex 128 would be quicker than manual check)
    for (auto &row : value)
    {
        // If value is greater than 0, add to vector, else, return 0
        output.push_back(relu(row));
    }

    return output;
}
// example running for relu:
/*
  std::vector<std::vector<float>> testMatrix = {
    { -1.0f, 0.0f, 2.0f },
    { 3.5f, -0.5f, 1.0f }
};

std::vector<std::vector<float>> result = relu(testMatrix);

std::cout << "ReLU Output:\n";
for (const auto& row : result) {
    for (float val : row) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
*/

//------------------------------------------
// SIGMOID MATH
//------------------------------------------

// Confidence or sigmoid function
// Added vectors
std::vector<float> sigmoid(const std::vector<float> &z)
{

    // Return Sigmoid function ( 1 / 1 + E ^-z)
    std::vector<float> output;
    output.reserve(z.size());

    // Loop through values and add Sigmoid vals
    for (float val : z)
    {
        output.push_back(1 / (1 + (std::exp(-val))));
    }
    return output;
}
// sigmoid matrix, same thing as relu, loop through rows, apply sigmoid per row.
std::vector<std::vector<float>> sigmoid(const std::vector<std::vector<float>> &matrix)
{
    std::vector<std::vector<float>> output;
    output.reserve(matrix.size());

    for (auto &row : matrix)
    {
        output.push_back(sigmoid(row));
    }
    return output;
}
// example running for sigmoid:
/*
  std::vector<std::vector<float>> testMatrix = {
    { -1.0f, 0.0f, 2.0f },
    { 3.5f, -0.5f, 1.0f }
};

std::vector<std::vector<float>> res = sigmoid(testMatrix);
std::cout << "Sigmoid test:\n ";
for(auto &row : res){
    for(float val : row){
        std::cout << val <<" ";
    }
    std::cout<<std::endl;
}
*/

//-------------------------------------------------------
// MATRIX MATH
//-------------------------------------------------------
// Matrix multiplication:
std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B)
{
    int aRows = A.size();    // number of rows in A
    int aCols = A[0].size(); // number of columns in A
    int bRows = B.size();    // number of rows in B
    int bCols = B[0].size(); // number of columns in B

    std::vector<std::vector<float>> output;
    if (aCols == bRows)
    {
        // Initialize output matrix with correct size (aRows x bCols), filled with 0s
        output = std::vector<std::vector<float>>(aRows, std::vector<float>(bCols, 0.0f));

        // Loop through rows of A and columns of B
        for (int i = 0; i < aRows; ++i)
        {
            for (int j = 0; j < bCols; ++j)
            {
                for (int k = 0; k < aCols; ++k)
                {
                    output[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
    // if its not compatable, output error msg
    else
    {
        std::cerr << "Matrixes are not compatable for multiplication." << std::endl;
    }
    return output;
}
// Print for matrixes
void printMatrix(const std::vector<std::vector<float>> &matrix)
{
    // loops through rows
    for (const auto &row : matrix)
    {
        // goes through the values, and prints them
        for (float val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}
// example running of matrix multiply code:
/*
     std::vector<std::vector<float>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };

    std::vector<std::vector<float>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };

    std::vector<std::vector<float>> result = matmul(A, B);

    printMatrix(result);

*/

//-------------------------------------------------
// LINEAR MATH
//-------------------------------------------------
// Applies a linear layer: output = ReLU(input * weights + bias)
std::vector<std::vector<float>> linear(const std::vector<std::vector<float>> &input, const std::vector<std::vector<float>> &weights, const std::vector<float> &bias)
{
    // Multiply input matrix by weights matrix
    auto output = matmul(input, weights);

    // Add bias to each element in the output's first row
    for (int i = 0; i < output[0].size(); ++i)
    { // Add bias to each element in the output's first row
        output[0][i] += bias[i];
    }
    // Apply ReLU activation: set negative values to 0
    for (int i = 0; i < output[0].size(); ++i)
    {
        if (output[0][i] < 0)
            output[0][i] = 0;
    }
    return output;
}

// Example running:
/*
 std::vector<std::vector<float>> input = {
        {1.0, 2.0, 3.0}
    };

    std::vector<std::vector<float>> weights = {
        {0.1, 0.2},
        {0.3, 0.4},
        {0.5, 0.6}
    };

    std::vector<float> bias = {0.5, 1.0};

    std::vector<std::vector<float>> result = linear(input, weights, bias);

    // Print result
    for (const auto& row : result) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
*/

//---------------------------------------
// SOFTMAX MATH
//---------------------------------------

// Applies the Softmax function to a 1D vector of scores.
// Converts raw values into a probability distribution that sums to 1.
// Uses max-subtraction for numerical stability.
std::vector<float> softmax(const std::vector<float> &input)
{
    // Find the maximum value in the input vector
    //  (Used to stabilize the exponentials and prevent overflow)
    float maxVal = input[0];
    for (float val : input)
    {
        if (val > maxVal)
            maxVal = val;
    }

    // Compute the exponentials of each input after subtracting the max
    std::vector<float> exps(input.size());
    for (int i = 0; i < input.size(); i++)
    {
        exps[i] = std::exp(input[i] - maxVal);
    }

    // Sum all exponentials
    float sum = 0.0f;
    for (float val : exps)
    {
        sum += val;
    }

    // Divide each exponential by the sum to get probabilities
    std::vector<float> output(input.size());
    for (int i = 0; i < input.size(); i++)
    {
        output[i] = exps[i] / sum;
    }

    return output;
}

// example running:
/*
std::vector<float> input = {2.0f, 1.0f, 0.1f};
std::vector<float> output = softmax(input);

for (float val : output) {
    std::cout << val << " ";
}
std::cout << std::endl;

*/

//---------------------------------
// CROSS ENTROPY MATH
//---------------------------------

// Computes the cross-entropy loss between predicted probabilities and the correct class index
float cross_entropy(const std::vector<float> &probs, int targetIndex)
{
    // Get the probability the model assigned to the correct class
    float correctProb = probs[targetIndex];

    // Compute the negative log of that probability
    float loss = -std::log(correctProb);

    return loss;
}

float cross_entropy(const std::vector<std::vector<float>> &batchProb, const std::vector<int> &targetIndices)
{
    float totalLoss = 0.0f;
    for (int i = 0; i < batchProb.size(); i++)
    {
        totalLoss += cross_entropy(batchProb[i], targetIndices[i]);
    }

    return totalLoss / batchProb.size();
}
// example code for crossEntropy calc or Foward pass + loss system:
/*

    std::vector<std::vector<float>> batch = {
        softmax({2.0f, 1.0f, 0.1f}),  // Should be confident in class 0
        softmax({0.5f, 2.5f, 0.3f}),  // Should be confident in class 1
        softmax({0.1f, 0.2f, 3.0f})   // Should be confident in class 2
    };

    std::vector<int> targets = {0, 1, 2};

    float loss = cross_entropy(batch, targets);
    std::cout << "Batch Cross-Entropy Loss: " << loss << std::endl;
*/

//-------------------------------------
//      SIGMOID DERIVATIVE MATH(BATCH)
//-------------------------------------
std::vector<std::vector<float>> sigmoidDerivative(const std::vector<std::vector<float>> &activated)
{
    std::vector<std::vector<float>> output;

    for (auto &row : activated)
    {
        std::vector<float> derivedrow;

        for (float val : row)
        {
            derivedrow.push_back(val * (1.0f - val));
        }
        output.push_back(derivedrow);
    }
    return output;
}
// Example running:
/*
 //test set, 2 rows and cols
    std::vector<std::vector<float>> test = {
        {0.8f, 0.1f},
        {0.5f, 0.9f}};
    //result is passing through sigmoid function
    std::vector<std::vector<float>> result = sigmoidDerivative(test);
        //through rows
    for (auto &row : result)
    {   //results from rows
        for (float val : row)
        {   //print results
            std::cout << val << " ";
        }

    }
*/

//----------------------------------
// BINARY CROSS ENTROPY MATH
//----------------------------------

// Binary cross entropy loss:
// Useful for 0s and 1s
float binary_cross_entropy_batch(const std::vector<std::vector<float>> &predictions, const std::vector<int> &targets)
{ // total loss
    float totalLoss = 0.0f;
    // go through predictions vectors, add predict as p[i] first val, then targets through the target vectors.
    for (int i = 0; i < predictions.size(); i++)
    {
        float p = predictions[i][0];
        float y = targets[i];
        // add to total loss using loss formula : loss = -[y * log(p) + (1 - y) * log(1 - p)]
        totalLoss -= (y * std::log(p) + (1 - y) * std::log(1 - p));
    }
    // finally return the loss
    return totalLoss / predictions.size();
}

// Example code run:
/*
 std::vector<std::vector<float>> predictions = {
        {0.1f}, {0.9f}, {0.8f}, {0.2f}
    };

    std::vector<int> targets = {0, 1, 1, 0};

    float loss = binary_cross_entropy_batch(predictions, targets);
    std::cout << loss << std::endl;
*/

//=====================================
//      GRADIENT COMPUTATIONS
//=====================================
/*

Gradient computations:

A vector that tells how to change each logit

Positive = increase prediction

Negative = decrease

Example:

probs:      [0.1, 0.7, 0.2]
target_id:  1

dZ:         [0.1, -0.3, 0.2]

It was too confident in 0.7, so it lowered the vectors number.
*/
std::vector<float> computeGradient(const std::vector<float> &probs, int targetId)
{
    std::vector<float> dZ = probs; // copy the predicted prbabilities
    dZ[targetId] -= 1.0f;          // Subtract 1 from the correct class
    return dZ;
}
/*
//Future me, heres an example running:

    std::vector<float> probs = {0.1, 0.7, 0.2};
    int target = 1;
    std::vector<float> dz = computeGradient(probs, target);
    for(float val : dz)
    {
        std::cout<<val <<std::endl;
    }

*/

//============================================
//      ONE-HOT INPUT TOKEN MATH
//============================================
std::vector<float> oneHot(int vocabSize, int index)
{
    /*
        We map the values, and only have the letter I decide to activate active.
        then I set the actives letters index to 1 instead of default 0.
        Then return it
    */
    std::vector<float> vec(vocabSize, 0.0f);
    vec[index] = 1.0f;
    return vec;
}

//=========================================
//  COMPUTING DELTA WEIGHTS OR DW
//=========================================

std::vector<std::vector<float>> computeDW(const std::vector<float> &x, const std::vector<float> &dZ)
{
    int input_size = x.size();
    int output_size = dZ.size();

    std::vector<std::vector<float>> dW(input_size, std::vector<float>(output_size));

    for (int i = 0; i < input_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            dW[i][j] = x[i] * dZ[j];
        }
    }

    return dW;
}
/*
    Example testing code for the matrixes

    std::vector<float> input = {0.0f, 0.0f, 1.0f, 0.0f};

    // Fake softmax gradient (output error dZ)
    std::vector<float> dZ = {0.1f, -0.3f, 0.2f, 0.0f};

    // Compute dW
    std::vector<std::vector<float>> dW = computeDW(input, dZ);

    //loop through rows
    for(const auto& row: dW)
    {
    //display values in matrix format ie: 0 0 0 0
                                          1 1 1 1
        for(float val :row)
        {
            std::cout<<val<<"\t";
        }
        std::cout<<"\n";
    }
*/
