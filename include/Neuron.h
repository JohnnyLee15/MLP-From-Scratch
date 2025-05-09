#pragma once
#include <vector>
#include <random>
class Layer;

using namespace std;

class Neuron {
    private:
        // Instance Variables
        vector<double> weights;
        double bias;

        // Constant
        static const double RELU_BIAS;

        // Static Variables
        static random_device rd;
        static mt19937 generator;

        // Methods
        void initWeights(int);
        void initBias();
        bool isOutputNeuron;

    public:
        Neuron(int, bool);
        double calActivation(const vector<double>&);
        void updateBias(double);
        int getNumWeights() const;
        void updateWeight(int, double);
        double getWeight(int) const;
};