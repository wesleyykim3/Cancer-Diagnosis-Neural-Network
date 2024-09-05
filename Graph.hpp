#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "utility.hpp"
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <queue>
#include <cmath>
#include <set>

// NodeInfo is the class which describes the qualities of a Node
struct NodeInfo {
    public:
        NodeInfo();
        NodeInfo(std::string activationFunction, double value, double bias);

        bool operator==(const NodeInfo& other);
        friend std::ostream& operator<<(std::ostream& out, const NodeInfo& n);

        // evaluate the activation function at the preActivation node value and store it in postActivationValue
        double activate();
        // evaluate the activation function's derivative at the preActivation node value
        double derive();

        FuncSig activationFunction;
        FuncSig activationDerivative;
        
        double preActivationValue;
        double postActivationValue;
        double bias;
        double delta; // accumulated derivative for this bias
};

// Connection Class is responsible for describing connections between Nodes in the graph
struct Connection {

    public:
        Connection();
        Connection(int source, int dest, double weight);

        bool operator<(const Connection& other);
        bool operator==(const Connection& other);
        friend std::ostream& operator<<(std::ostream& out, const Connection& c);

        int source;
        int dest;
        double weight;
        double delta; // accumulated derivative for this weight

};

// AdjList becomes an alias for std::vector<std::unordered_map<int, Connection>>
// The adjacency list maps a node's id to a Connection object
typedef std::vector<std::unordered_map<int, Connection> > AdjList;

// Graph class describes a general graph structure
// The graph contains a collection of NodeInfo 
class Graph {

    public:
        Graph();
        Graph(int size);
        Graph(const Graph& other);
        Graph& operator=(const Graph& other);
        ~Graph();

        void updateNode(int id, NodeInfo n);
        NodeInfo* getNode(int id) const;
        void updateConnection(int v, int u, double w);

        AdjList& getAdjacencyList();

        friend std::ostream& operator<<(std::ostream& out, const Graph& g);
        void resize(int size);

    protected:
        // protected to give NeuralNetwork access

        AdjList adjacencyList; // adjacency list containing weights for edges
        std::vector<NodeInfo*> nodes; // vector of all the nodes in the graph
        int size; // number of nodes

        std::vector<NodeInfo*> getNodes() const;
        void clear();
};

#endif