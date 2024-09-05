#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <cmath>
#include <string>
#include <random>
#include <iostream>
typedef double(* FuncSig)(double param);

// Activation functions
double identity(double x);
double ReLU(double x);
double sigmoid(double x);

// Activation function derivatives
double step(double x);
double sigmoid_prime(double x);
double identity_prime(double x);

FuncSig getActivationFunction(std::string identifier);
FuncSig getActivationDerivative(std::string identifier);
std::string getActivationIdentifier(FuncSig f);

double sample();

std::ostream& operator<<(std::ostream& out, std::vector<double> v);

#endif