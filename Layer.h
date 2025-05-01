#include <vector>
#include "Neuron.h"
using namespace std;

class Layer {
    private:
        vector<Neuron> neurons;
        vector<double> activations;

    public:
        Layer(int, int, bool = false);
        void calActivations(const vector<double>&);
        vector<double> getActivations() const;
        int getNumNeurons() const;
        void updateNeuronBias(double, int);
        int getNumNeuronWeights(int) const;
        void updateNeuronWeight(int, int, double);
        double getNeuronActivation(int) const;
        double getNeuronWeight(int, int) const;
        double getMaxActivation() const;
};