#pragma once

#include "core/Data.h"
#include "core/Tensor.h"
#include <unordered_map>
#include <vector>
#include <fstream>

class ImageData2D : public Data {
    private:
        Tensor trainFeatures;
        vector<double> trainTargets;

        Tensor testFeatures;
        vector<double> testTargets;

        unordered_map<string, int> labelMap;

    public:
        void setTrainFeatures(const Tensor&);
        void setTestFeatures(const Tensor&);
        void setTrainTargets(const vector<string>&);
        void setTestTargets(const vector<string>&);

        const Tensor& getTrainFeatures() const override;
        const Tensor& getTestFeatures() const override;;
        const vector<double>& getTrainTargets() const override;;
        const vector<double>& getTestTargets() const override;;

        size_t getNumTrainSamples() const override;
        uint32_t getEncoding() const override;
        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;
};