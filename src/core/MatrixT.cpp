#include "core/MatrixT.h"
#include "core/Matrix.h"

MatrixT::MatrixT(const Matrix &matrix) :
    numRows(matrix.getNumCols()), numCols(matrix.getNumRows()), matrix(matrix) {}

size_t MatrixT::getNumRows() const {
    return numRows;
}

size_t MatrixT::getNumCols() const {
    return numCols;
}

const vector<double>& MatrixT::getFlat() const {
    return matrix.getFlat();
}

Tensor MatrixT::operator *(const Matrix &mat2) const {
    Matrix::checkSizeMatch(numCols, mat2.getNumRows());

    size_t mat2Rows = mat2.getNumRows();
    size_t mat2Cols = mat2.getNumCols();

    Tensor product({numRows, mat2Cols});
    vector<double> &productFlat = product.getFlat();
    const vector<double> &mat2Flat = mat2.getFlat();
    const vector<double> &flat = matrix.getFlat();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < mat2Cols; j++) {
            size_t offsetProd = i * mat2Cols;
            double value = 0.0;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += flat[k*  numRows + i] * mat2Flat[k*mat2Cols + j];
            }
            productFlat[offsetProd + j] = value;
        }
    }

    return product;
}

Tensor MatrixT::operator *(const MatrixT &mat2) const {
    Matrix::checkSizeMatch(numCols, mat2.numRows);

    size_t mat2Rows = mat2.numRows;
    size_t mat2Cols = mat2.numCols;

    Tensor product({numRows, mat2Cols});
    vector<double> &productFlat = product.getFlat();
    const vector<double> &mat2Flat = mat2.getFlat();
    const vector<double> &flat = matrix.getFlat();
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < mat2Cols; j++) {
            double offset = j*mat2Rows;
            size_t offsetProd = i * mat2Cols;
            double value = 0.0;
            for (size_t k = 0; k < mat2Rows; k++) {
                value += flat[k*numRows + i] * mat2Flat[offset + k];
            }
            productFlat[offsetProd + j] = value;
        }
    }

    return product;
}
