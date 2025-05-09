#include "Data.h"
#include <fstream>
#include <iostream>
#include <sstream>

void Data::readData(string filename, bool isTrainData, int targetIdx) {
    ifstream file(filename);
    string line;
    vector<vector<double> > features;
    vector<double> target;

    // Read Header
    getline(file, line);
    while(getline(file, line)) {
        stringstream lineParser(line);
        string token;

        int i = 0;
        vector<double> sample;
        while(getline(lineParser, token, ',')) {
            double value = stod(token);
            if (i == targetIdx) {
                target.push_back(value); 
            } else {
                sample.push_back(value/255.0);
            }
            i++;
        }

        features.push_back(sample);
    }

    if (isTrainData) {
        trainFeatures = features;
        trainTarget = target;
    } else {
        testFeatures = features;
        testTarget = target;
    }
}

void Data::readTrain(string filename, int targetIdx) {
    readData(filename, true, targetIdx);
}

void Data::readTest(string filename, int targetIdx) {
    readData(filename, false, targetIdx);
}

vector<vector<double> > Data::getTrainFeatures() const {
    return trainFeatures;
}

vector<vector<double> > Data::getTestFeatures() const {
    return testFeatures;
}

vector<double> Data::getTrainTarget() const {
    return trainTarget;
}

vector<double> Data::getTestTarget() const {
    return testTarget;
}