#include <string>
#include <vector>

using namespace std;

class Data {
    private:
        vector<vector<double> > trainFeatures;
        vector<double> trainTarget;
        vector<vector<double> > testFeatures;
        vector<double> testTarget;
        void readData(string, bool, int);
    
    public:
        void readTrain(string, int);
        void readTest(string, int);
        // void readAllData(string, int, float);
        vector<vector<double> > getTrainFeatures() const;
        vector<vector<double> > getTestFeatures() const;
        vector<double> getTrainTarget() const;
        vector<double> getTestTarget() const;
};