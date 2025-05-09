#include "Layer.h"
#include "Neuron.h"
#include <cmath>

Layer::Layer(int numNeurons, int numWeights, bool isOutputLayer): 
    neurons(numNeurons, Neuron(numWeights, isOutputLayer)),
    activations(numNeurons, 0) {}

void Layer::calActivations(const vector<double> &prevActivations) {
    for (int i = 0; i < neurons.size(); i++) {
        activations[i] = neurons[i].calActivation(prevActivations);
    }
}

vector<double> Layer::getActivations() const {
    return activations;
}

int Layer::getNumNeurons() const {
    return neurons.size();
}

void Layer::updateNeuronBias(double delta, int idx) {
    neurons[idx].updateBias(delta);
}

int Layer::getNumNeuronWeights(int idx) const{
    return neurons[idx].getNumWeights();
}

void Layer::updateNeuronWeight(int neuronIdx, int weightIdx, double delta) {
    neurons[neuronIdx].updateWeight(weightIdx, delta);
}

double Layer::getNeuronActivation(int idx) const{
    return activations[idx];
}

double Layer::getNeuronWeight(int neuronIdx, int weightIdx) const{
    return neurons[neuronIdx].getWeight(weightIdx);
}

double Layer::getMaxActivation() const {
    double max = -INFINITY;
    for (int i = 0; i < activations.size(); i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    return max;
}