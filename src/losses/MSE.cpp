#include "losses/MSE.h"
#include "core/Tensor.h"
#include <cassert>
#include <cmath>

double MSE::calculateTotalLoss(
    const vector<double>& targets, 
    const Tensor& activations
) const {
    const vector<double> &actFlat = activations.getFlat();
    assert(actFlat.size() == targets.size());

    size_t size = actFlat.size();
    double totalLoss = 0.0;

    #pragma omp parallel for reduction(+:totalLoss)
    for (size_t i = 0; i < size; i++) {
        double diff = targets[i] - actFlat[i];
        totalLoss += (diff * diff);
    }

    return totalLoss;
}

Tensor MSE::calculateGradient(
    const vector<double> &targets, 
    const Tensor &activations
) const {
    const vector<double> &actFlat = activations.getFlat();
    assert(actFlat.size() == targets.size());

    size_t size = actFlat.size();
    Tensor gradients({size, 1});

    vector<double> &gradientsFlat = gradients.getFlat();
    const vector<double> &activationsFlat = activations.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        gradientsFlat[i] = 2*(activationsFlat[i] - targets[i]);
    }
    
    return gradients;
}

double MSE::formatLoss(double avgLoss) const {
    return sqrt(avgLoss);
}

uint32_t MSE::getEncoding() const {
    return Loss::Encodings::MSE;
}

