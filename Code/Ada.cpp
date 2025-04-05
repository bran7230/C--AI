#include <iostream>
#include <cmath>
#include <vector>

// =====================================MATH FUNCTIONS=============================================================
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

//===================================================END OF MATH=======================================================

int main()
{

    return 0;
}
