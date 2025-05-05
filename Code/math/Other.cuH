#pragma once
#include "Adamath.cuH"
#include <mma.h>
using namespace nvcuda;
//============================
//   LOSS / GRADIENT UTILS
//============================

float cross_entropy(const std::vector<float> &probs, int targetIndex)
{
    return -std::log(probs[targetIndex]);
}

float cross_entropy(const std::vector<std::vector<float>> &batchProb, const std::vector<int> &targetIndices)
{
    float totalLoss = 0.0f;
#pragma omp parallel for reduction(+ : totalLoss)
    for (int i = 0; i < batchProb.size(); i++)
        totalLoss += cross_entropy(batchProb[i], targetIndices[i]);
    return totalLoss / batchProb.size();
}

float binary_cross_entropy_batch(const std::vector<std::vector<float>> &predictions, const std::vector<int> &targets)
{
    float totalLoss = 0.0f;
#pragma omp parallel for reduction(+ : totalLoss)
    for (int i = 0; i < predictions.size(); ++i)
    {
        float p = predictions[i][0];
        float y = targets[i];
        totalLoss -= (y * std::log(p) + (1 - y) * std::log(1 - p));
    }
    return totalLoss / predictions.size();
}

std::vector<std::vector<float>> sigmoidDerivative(const std::vector<std::vector<float>> &activated)
{
    std::vector<std::vector<float>> output(activated.size());
#pragma omp parallel for
    for (int i = 0; i < activated.size(); ++i)
    {
        const auto &row = activated[i];
        std::vector<float> derivedRow(row.size());
        for (int j = 0; j < row.size(); ++j)
            derivedRow[j] = row[j] * (1.0f - row[j]);
        output[i] = derivedRow;
    }
    return output;
}

std::vector<float> computeGradient(const std::vector<float> &probs, int targetId)
{
    std::vector<float> dZ = probs;
    dZ[targetId] -= 1.0f;
    return dZ;
}

std::vector<float> oneHot(int vocabSize, int index)
{
    std::vector<float> vec(vocabSize, 0.0f);
    vec[index] = 1.0f;
    return vec;
}

std::vector<std::vector<float>> computeDW(const std::vector<float> &x, const std::vector<float> &dZ)
{
    int input_size = x.size();
    int output_size = dZ.size();
    std::vector<std::vector<float>> dW(input_size, std::vector<float>(output_size));
    for (int i = 0; i < input_size; ++i)
        for (int j = 0; j < output_size; ++j)
            dW[i][j] = x[i] * dZ[j];
    return dW;
}
