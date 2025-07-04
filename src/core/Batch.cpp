#include "core/Batch.h"
#include <omp.h>
#include "core/DenseLayer.h"
#include "utils/TrainingUtils.h"
#include "activations/Activation.h"
#include "core/Matrix.h"

Batch::Batch(size_t numLayers, size_t batchSize) :
    batchSize(batchSize),
    indices(batchSize),
    targets(batchSize)
{}

void Batch::setBatchIndices(
    size_t start,
    size_t end,
    const vector<size_t> &shuffledIndices
) {
    #pragma omp parallel for
    for (size_t i = start; i < end; i++) {
        indices[i - start] = shuffledIndices[i];
    }
}

void Batch::setBatch(
    const Tensor &train,
    const vector<double> &trainLabels
) {
    const vector<size_t> &trainShape = train.getShape();
    vector<size_t> batchShape = trainShape;
    batchShape[0] = batchSize;
    data = Tensor(batchShape);
    size_t batchDims = batchShape.size();
    size_t elementSize = 1;
    for (size_t i = 1; i < batchDims; i++) {
        elementSize *= batchShape[i];
    }

    vector<double> &batchFlat = data.getFlat();
    const vector<double> &trainFlat = train.getFlat();
    #pragma omp parallel for
    for (size_t i = 0; i < batchSize; i++) {
        size_t rdIdx = indices[i];
        for (size_t j = 0; j < elementSize; j++) {
            batchFlat[i*elementSize + j] = trainFlat[rdIdx * elementSize + j];
        }

        targets[i] = trainLabels[rdIdx];
    }
}

const Tensor& Batch::getData() const {
    return data;
}

const vector<double>& Batch::getTargets() const {
    return targets;
}

size_t Batch::getSize() const {
    return data.getShape()[0];
}

const vector<size_t>& Batch::getIndices() const {
    return indices;
}