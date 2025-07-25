#pragma once

#include <vector>
#include "core/tensor/Tensor.h"
#include "core/layers/Layer.h"

class Activation;

using namespace std;

class Dense : public Layer {
    private:

        // Constants
        static const float HE_INT_GAIN;

        // Instance Variables
        size_t numNeurons;

        Tensor activations;
        Tensor preActivations;
        Tensor weights;
        Tensor dB;
        Tensor dW;
        Tensor dX;
        Tensor dA;
        Tensor biases;

        Activation *activation;
        bool isLoadedDense;
        
        // Methods
        void initWeights();
        void initParams(size_t);
        void checkBuildSize(const vector<size_t>&) const;
        
        void ensureGpu();
        void ensureCpu();

        vector<uint32_t> generateThreadSeeds() const;
        void loadActivation(ifstream&);

        void reShapeBatch(size_t);
        
    public:
        // Constructors
        Dense(size_t, Activation*);
        Dense();

        // Destructor
        ~Dense();

        // Methods
        void build(const vector<size_t>&) override;

        void forward(const Tensor&) override;
        void backprop(const Tensor&, float, Tensor&, bool) override;

        const Tensor& getOutput() const override;
        Tensor& getOutputGradient() override;
        
        Layer::Encodings getEncoding() const override;
        vector<size_t> getBuildOutShape(const vector<size_t>&) const override;

        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;

        // GPU Interface
        #ifdef __APPLE__
            void forwardGpu(const Tensor&, GpuCommandBuffer) override;
            void backpropGpu(const Tensor&, float, Tensor&, bool, GpuCommandBuffer) override;
            void downloadOutputFromGpu() override;
        #endif
};