#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <string>
#include <fstream>
#include <vector>
#include <utility>
#include <iostream>
#include <sstream>
#include <cmath>

// DataInstance class stores an instance of data
struct DataInstance {
    DataInstance(std::vector<double> features, int label = 0);
    std::vector<double> x;
    int y;

    friend std::ostream& operator<<(std::ostream& o, const DataInstance& d);
};

// Functions to calculate the mean and standard deviation of each feature
std::vector<double> calculateMean(const std::vector<DataInstance>& data);
std::vector<double> calculateStdDev(const std::vector<DataInstance>& data, const std::vector<double>& mean);

// DataLoader class loads data and stores the data as DataInstances
class DataLoader {
    public:
        DataLoader(std::string filename);
        DataLoader(std::istream& fin);

        std::vector<DataInstance> getData() const;

    private:

        std::vector<std::string> split(std::string s, std::string delimiter);
        void loadData(std::istream& in);
        std::vector<DataInstance> data;

        void normalizeDataSet();

};

#endif