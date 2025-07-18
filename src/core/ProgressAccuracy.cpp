#include "core/ProgressAccuracy.h"
#include "core/Tensor.h"
#include "core/Matrix.h"
#include "core/Batch.h"
#include "utils/TrainingUtils.h"

const string ProgressAccuracy::NAME = "Accuracy";

ProgressAccuracy::ProgressAccuracy(size_t numSamples) : 
    ProgressMetric(numSamples), predictions(numSamples) {}

void ProgressAccuracy::init() {
    ProgressMetric::init();
    correctPredictions = 0;
}

string ProgressAccuracy::getName() const {
    return NAME;
}

void ProgressAccuracy::update(
    const Batch &batch,
    const Loss *loss,
    const Tensor &outputActivations,
    double batchTotalLoss
) {
    ProgressMetric::update(batch, loss, outputActivations, batchTotalLoss);

    size_t batchSize = batch.getSize();
    Matrix probsMat = outputActivations.M();
    size_t numCols = probsMat.getNumCols();
    const vector<double> &probsFlat = outputActivations.getFlat();
    const vector<size_t> &indices = batch.getIndices();
    const vector<double> &batchTargets = batch.getTargets();
    size_t localCorrect = 0;

    #pragma omp parallel for reduction(+:localCorrect)
    for (size_t i = 0; i < batchSize; i++) {
        double prediction = TrainingUtils::getPrediction(probsFlat, i, numCols);
        predictions[indices[i]] = prediction;

        if (prediction == batchTargets[i]) {
            localCorrect++;
        }
    }
    
    correctPredictions += localCorrect;
}

double ProgressAccuracy::calculate() const {
    if (getSamplesProcessed() == 0) {
        return 0.0;
    }

    return 100 * (double) correctPredictions/getSamplesProcessed();
}