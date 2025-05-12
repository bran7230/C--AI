#pragma once
#include "Syntari.cuH"
#include <mma.h>
using namespace nvcuda;
//==============================
//      SIGMOID MATH
//==============================

//====================================================================
//  DO NOT USE THIS FOR LLMS, ONLY USE FOR SIMPLER NEURAL NETS IE RRNS
//====================================================================
std::vector<float> sigmoid(const std::vector<float> &z)
{
    std::vector<float> output;
    output.reserve(z.size());
    for (float val : z)
        output.push_back(1 / (1 + std::exp(-val)));
    return output;
}

std::vector<std::vector<float>> sigmoidBatch(const std::vector<std::vector<float>> &matrix)
{
    std::vector<std::vector<float>> output(matrix.size());
#pragma omp parallel for
    for (int i = 0; i < matrix.size(); ++i)
        output[i] = sigmoid(matrix[i]);
    return output;
}
