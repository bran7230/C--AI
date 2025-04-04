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
// mad a matrix, loop through, apply relu and return final matrix
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
//===================================================END OF MATH=======================================================

int main()
{

    return 0;
}
