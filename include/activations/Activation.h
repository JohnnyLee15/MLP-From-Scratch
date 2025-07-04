#pragma once
#include <cstdint>
#include <vector>

class Tensor;

using namespace std;

class Activation {
    public:
        // Methods
        virtual Tensor activate(const Tensor&) const = 0;
        virtual Tensor calculateGradient(const Tensor&) const = 0;
        virtual vector<double> initBias(size_t) const = 0;
        virtual ~Activation() = default;
        virtual bool isFused() const;
        virtual uint32_t getEncoding() const = 0;

        // Enums
        enum Encodings : uint32_t {
            Linear,
            ReLU,
            Softmax
        };
};