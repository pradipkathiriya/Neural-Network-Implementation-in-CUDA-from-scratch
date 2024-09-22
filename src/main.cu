#include <iostream>
#include <nn.h>
#include <linear_layer.h>
#include <activation.h>
#include <nn_exception.h>
#include <BCE_loss.h>
#include <nn_layer.h>
#include <utility.h>
#include <dataset.h>
#include <logger.h>

// void printMatrix(Matrix& matrix, const std::string& name) {
//     matrix.copy_device_to_host();
//     std::cout << name << ":" << std::endl;
//     for (int i = 0; i < matrix.shape.x * matrix.shape.y; ++i) {
//         std::cout << matrix[i] << " ";
//     }
//     std::cout << std::endl;
// }

// int main() {
//     // Define input dimensions and initialize the layer
//     Shape input_shape(1, 3); // (1 rows, 3 columns, transposed vector)
//     Shape weight_shape(3, 1); // shape of weights, resulting in a 1x1 output

//     LinearLayer layer("test_layer", weight_shape);

//     // Allocate memory for input and output
//     Matrix input(input_shape);
//     input.allocate_memory();
//     input[0] = 0.1f; input[1] = 0.2f; input[2] = 0.3f;
//     input.copy_host_to_device();

//     // Allocate memory for target
//     Matrix target(Shape(1, 1)); // 1x1 target matrix
//     target.allocate_memory();
//     target[0] = 0.0f;
//     target.copy_host_to_device();

//     // Print initial weights and biases
//     printMatrix(layer.get_weight_matrix(), "Initial Weights");
//     printMatrix(layer.get_bias_vector(), "Initial Biases");

//     // Perform forward pass
//     Matrix& output = layer.forward(input);

//     output.copy_device_to_host();

//     // Print forward pass output
//     std::cout << "Forward pass output:" << std::endl;
//     for (int i = 0; i < output.shape.x * output.shape.y; ++i) {
//         std::cout << output[i] << " ";
//     }
//     std::cout << std::endl;

//     // Calculate BCE loss
//     BCELoss bce;
//     float loss = bce.cost(output, target);
//     std::cout << "Binary Cross Entropy Loss: " << loss << std::endl;

//     // Calculate gradient of BCE loss
//     Matrix dZ(output.shape);
//     dZ.allocate_memory();
//     bce.d_cost(output, target, dZ);

//     // Perform backpropagation
//     float learning_rate = 0.01f;
//     Matrix& dA = layer.backprop(dZ, learning_rate);
//     dA.copy_device_to_host();

//     // Print backpropagation output (dA)
//     std::cout << "Backpropagation output (dA):" << std::endl;
//     for (int i = 0; i < dA.shape.x * dA.shape.y; ++i) {
//         std::cout << dA[i] << " ";
//     }
//     std::cout << std::endl;

//     // Print updated weights and biases
//     printMatrix(layer.get_weight_matrix(), "Updated Weights");
//     printMatrix(layer.get_bias_vector(), "Updated Biases");

//     return 0;
// }


float compute_accuracy(const Matrix& predictions, const Matrix& targets);

int main(int argc, char *argv[])
{
    srand( time(NULL) );

    Logger logger("logfile.txt");
    logger.set_log_level(LogLevel::INFO);

	CoordinatesDataset dataset(32, 128);
	BCELoss bce_cost;

	NeuralNetwork nn;
	nn.add_layer(new LinearLayer("linear_1", Shape(2, 32)));
	nn.add_layer(new SigmoidActivation("relu_1"));
    nn.add_layer(new LinearLayer("linear_1", Shape(32, 64)));
	nn.add_layer(new SigmoidActivation("relu_2"));
    nn.add_layer(new LinearLayer("linear_1", Shape(64, 128)));
	nn.add_layer(new SigmoidActivation("relu_3"));
    nn.add_layer(new LinearLayer("linear_1", Shape(128, 64)));
	nn.add_layer(new SigmoidActivation("relu_3"));
    nn.add_layer(new LinearLayer("linear_1", Shape(64, 32)));
	nn.add_layer(new SigmoidActivation("relu_3"));
	nn.add_layer(new LinearLayer("linear_2", Shape(32, 1)));
	nn.add_layer(new SigmoidActivation("sigmoid_output"));

	// network training
	Matrix Y;
	for (int epoch = 0; epoch < 10000; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.get_num_of_batches() - 1; batch++) {
			Y = nn.forward(dataset.get_batches().at(batch));
			nn.backprop(Y, dataset.get_targets().at(batch));
			cost += bce_cost.cost(Y, dataset.get_targets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch << ", Cost: " << cost / dataset.get_num_of_batches()
				    << std::endl;
		}
	}

	// compute accuracy
	Y = nn.forward(dataset.get_batches().at(dataset.get_num_of_batches() - 1));
	Y.copy_device_to_host();

	float accuracy = compute_accuracy( Y, dataset.get_targets().at(dataset.get_num_of_batches() - 1));
	std::cout 	<< "Accuracy: " << accuracy << std::endl;

	return 0;
}

float compute_accuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}

	return static_cast<double>(correct_predictions) / m;
}
