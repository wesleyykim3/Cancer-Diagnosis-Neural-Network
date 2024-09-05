CXX=g++
CXX_FLAGS=-std=c++17

targets=neuralnet

all: $(targets)

neuralnet: main.o NeuralNetwork.o Graph.o DataLoader.o utility.o
	$(CXX) $(CXX_FLAGS) $^ -o $@

main.o: main.cpp
	$(CXX) $(CXX_FLAGS) $^ -c

NeuralNetwork.o: NeuralNetwork.cpp NeuralNetwork.hpp 
	$(CXX) $(CXX_FLAGS) NeuralNetwork.cpp -c 

Graph.o: Graph.cpp Graph.hpp 
	$(CXX) $(CXX_FLAGS) Graph.cpp -c

DataLoader.o: DataLoader.cpp DataLoader.hpp
	$(CXX) $(CXX_FLAGS) DataLoader.cpp -c

utility.o: utility.cpp utility.hpp
	$(CXX) $(CXX_FLAGS) utility.cpp -c 

clean:
	rm -f $(targets) *.o *.gch a.out *.exe