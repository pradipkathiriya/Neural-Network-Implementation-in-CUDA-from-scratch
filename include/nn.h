#ifndef NN_H
#define NN_H

#include <BCE_loss.h>
#include <nn_layer.h>
#include <vector>

using namespace std;

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	BCELoss bce_cost;

	Matrix Y;
	Matrix dY;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.001);
	~NeuralNetwork();

	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);

	void add_layer(NNLayer *layer);
	std::vector<NNLayer*> get_layers() const;

};

#endif