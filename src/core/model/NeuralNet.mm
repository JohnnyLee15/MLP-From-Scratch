#include "core/model/NeuralNet.h"
#include "core/data/Batch.h"
#include "core/losses/Loss.h"
#include "core/gpu/GpuEngine.h"

void NeuralNet::fitBatchGpu(Batch &batch, float learningRate) {
    id<MTLCommandBuffer> cmdBuf = [GpuEngine::getCmdQueue() commandBuffer];
    forwardPassGpu(batch, (GpuCommandBuffer) cmdBuf);
    backpropGpu(batch, learningRate, (GpuCommandBuffer) cmdBuf);
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    layers.back()->downloadOutputFromGpu();
}

void NeuralNet::forwardPassGpu(Batch &batch, GpuCommandBuffer cmdBuf) {
    const Tensor *prevActivations = &batch.getData();
    size_t numLayers = layers.size();

    for (size_t j = 0; j < numLayers; j++) {
        layers[j]->forwardGpu(*prevActivations, cmdBuf);
        prevActivations = &layers[j]->getOutput();
    }
}

void NeuralNet::backpropGpu(Batch &batch, float learningRate, GpuCommandBuffer cmdBuf) {
    if (batch.getSize() != dL.getShape()[0]) {
        reShapeDL(batch.getSize());
    }

    loss->calculateGradientGpu(batch.getTargets(),layers.back()->getOutput(), dL, cmdBuf);
    size_t numLayers = (int) layers.size();
    
    Tensor *grad = &dL;
    for (int i = numLayers - 1; i >= 0; i--) {
        bool isFirstLayer = (i == 0);
        const Tensor &prevActivations = ((i == 0) 
            ? batch.getData() 
            : layers[i-1]->getOutput());

        layers[i]->backpropGpu(prevActivations, learningRate, *grad, isFirstLayer, cmdBuf);

        grad = &layers[i]->getOutputGradient();
    }
}
