#include <iostream>
#include "NeuralNetwork.hpp"
#include "utility.hpp"
#include "DataLoader.hpp"
using namespace std;

void testTrain(string networkFile, string trainFile, string testFile);

int main(int argc, char* argv[]) {
    testTrain("./models/diabetes.init", "./data/diabetes_train.csv", "./data/diabetes_test.csv");
    return 0;
}

// train loop similar to pytorch
void testTrain(string networkFile, string trainFile, string testFile) {
    NeuralNetwork nn(networkFile);
    nn.setLearningRate(0.001);

    // initialize a dataloader
    DataLoader dl(trainFile);

    vector<double> output;
    int numEpochs = 4;

    // put the neural network in train mode
    nn.train();

    for (int i = 0; i < numEpochs; i++) {
        // cout << nn << endl;
        for (size_t j = 0; j < dl.getData().size(); j++) {
            DataInstance di = dl.getData().at(j);
            output = nn.predict(di);
        }
        cout << "epoch: " << i << " accuracy: " << nn.assess(testFile) << endl;
        nn.update();
    }

    // cout << nn << endl;
    cout << "accuracy: " << nn.assess(testFile) << endl;
}
