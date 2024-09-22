#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <iostream>
#include <GPUmatrix.h>
#include <nn_exception.h>
#include <utility.h>
#include <nn_layer.h>

class SigmoidActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix dZ;

public:
	SigmoidActivation(std::string name);
	~SigmoidActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};

class ReLUActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix dZ;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};

class LeakyReLUActivation : public NNLayer {
public:
    LeakyReLUActivation(std::string name, float alpha = 0.01f);
    ~LeakyReLUActivation();

    Matrix& forward(Matrix& Z);
    Matrix& backprop(Matrix& dA, float learning_rate = 0.01);

private:
    std::string name;
    Matrix Z;
    Matrix A;
    Matrix dZ;
    float alpha;  // The slope for the negative part of Leaky ReLU
};

#endif