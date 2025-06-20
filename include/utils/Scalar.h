#pragma once
#include <vector>

class Matrix;

using namespace std;

class Scalar {
    private:
        // Instance Variables
        bool fitted;
        bool transformed;

        // Methods
        void checkFitted();
        void checkTransformed() const;

    public:
        // Methods
        virtual void fit(const Matrix&);
        virtual void transform(Matrix&) = 0;
        virtual void reverseTransform(Matrix&) const = 0;

        virtual void fit(const vector<double>&);
        virtual void transform(vector<double>&) = 0;
        virtual void reverseTransform(vector<double>&) const = 0;

        virtual void resetToRaw();
        bool isTransformed() const;
        virtual ~Scalar() = default;
};