#include <utils/FeatureEncoder.h>
#include <utils/ConsoleUtils.h>
#include "core/Matrix.h"
#include <stdexcept>

bool FeatureEncoder::getValueType(const string &value) {
    try {
        stod(value);
        return false;
    } catch(const invalid_argument &ia) {
        return true;
    } catch(const out_of_range &oof) {
        return true;
    }
}

vector<bool> FeatureEncoder::getCategoricalCols(
    const vector<vector<string> > &featuresRaw
) {
    size_t numRows = featuresRaw.size();
    size_t numCols = featuresRaw[0].size();
    vector<bool> isCategorical(numCols, false);

    for (size_t j = 0; j < numCols; j++)  {
        for (size_t i = 0; i < numRows && !isCategorical[j]; i++) {
            isCategorical[j] = getValueType(featuresRaw[i][j]);
        }
    }
    
    return isCategorical;
}

unordered_map<string, double> FeatureEncoder::encodeFeature(
    const vector<vector<string> > &featuresRaw,
    size_t colIdx
) {
    size_t numRows = featuresRaw.size();
    unordered_map<string, double> encoding;
    int nextIdx = 0;

    for (size_t i = 0; i < numRows; i++) {
        const string &val = featuresRaw[i][colIdx];
        if (encoding.find(val) == encoding.end()) {
            encoding[val] = (double) nextIdx++;
        }
    }

    return encoding;
}

vector<unordered_map<string, double> > FeatureEncoder::encodeFeatures(
    const vector<vector<string> > &featuresRaw,
    const vector<bool> &isCategorical
) {
    size_t numCols = featuresRaw[0].size();
    vector<unordered_map<string, double> > encodings(numCols);

    #pragma omp parallel for
    for (size_t j = 0; j < numCols; j++)  {
        if (isCategorical[j]) {
            encodings[j] = encodeFeature(featuresRaw, j);
        }
    }

    return encodings;
}

vector<size_t> FeatureEncoder::getOffsets(
    const vector<bool> &isCategorical,
    const vector<unordered_map<string, double> > &encodings,
    size_t numCols,
    int &totalCols
) {
    vector<size_t> offsets(numCols);
    for (size_t i = 0; i < numCols; i++) {
        offsets[i] = totalCols;
        if (isCategorical[i]) {
            totalCols += encodings[i].size();
        } else {
            totalCols++;
        }
    }

    return offsets;
}

Matrix FeatureEncoder::getFeatures(
    const vector<vector<string> > &featuresRaw
) {
    ConsoleUtils::loadMessage("Extracting Features.");
    vector<bool> isCategorical = getCategoricalCols(featuresRaw);
    size_t numRows = featuresRaw.size();
    size_t numCols = featuresRaw[0].size();
    vector<unordered_map<string, double> > encodings = encodeFeatures(featuresRaw, isCategorical);

    int totalCols = 0;
    vector<size_t> offsets = getOffsets(isCategorical, encodings, numCols, totalCols);
    Matrix features(numRows, totalCols);
    vector<double> &featuresFlat = features.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++)  {
        for (size_t j = 0; j < numCols; j++) {
            size_t offset = offsets[j];
            if (isCategorical[j]) {
                size_t catIdx = encodings[j].at(featuresRaw[i][j]);
                featuresFlat[i * totalCols + offset + catIdx] = 1;
            } else {
                featuresFlat[i * totalCols + offset] = stod(featuresRaw[i][j]);
            }
        }
    }
    ConsoleUtils::completeMessage();

    return features;
}