#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include "Graph.hpp"
#include "DataLoader.hpp"

// NeuralNetwork class inherits from the Graph class
class NeuralNetwork : public Graph {

    public:
        NeuralNetwork();
        NeuralNetwork(int size);
        NeuralNetwork(std::string filename);
        NeuralNetwork(std::istream& in);

        void eval(); // puts the neural network in eval mode, no gradients accumulated
        void train(); // puts the neural network in train mode, gradients accumulated
        void setLearningRate(double lr);
        void setInputNodeIds(std::vector<int> inputNodeIds);
        void setOutputNodeIds(std::vector<int> outputNodeIds);
        std::vector<int> getInputNodeIds() const;
        std::vector<int> getOutputNodeIds() const;

        std::vector<double> predict(DataInstance instance); // computes predicted values
        bool update(); // apply accumumated gradients and update weights and biases

        double assess(DataLoader dl); // calculates neural networks accuracy
        double assess(std::string filename); // calculates neural networks accuracy
        void saveModel(std::string filename); // saves the model
        friend std::ostream& operator<<(std::ostream& out, const NeuralNetwork& nn);

    private:
        bool contribute(double y, double p);
        double contribute(int nodeId, const double& y, const double& p); // contribute helper function

        void visitPredictNode(int vId); // visits the node during evaluation
        void visitPredictNeighbor(Connection c); // visits node's neighbor during evaluation
        void visitContributeNode(int vId, double& outgoingContribution);
        void visitContributeNeighbor(Connection& c, double& incomingContribution, double& outgoingContribution);

        double batchSize; // number of training instances
        bool evaluating; // eval or train mode
        double learningRate;
        std::vector<std::vector<int> > layers; // stores each layer as a vector of nodes
        std::unordered_map<int, double> contributions; // keeps track of which contributions have already been made
        std::vector<int> inputNodeIds;
        std::vector<int> outputNodeIds;
        
        void loadNetwork(std::istream& in); // loads neural network structure from the input file stream
        void flush(); // refreshes node values for the next computation
};

#endif