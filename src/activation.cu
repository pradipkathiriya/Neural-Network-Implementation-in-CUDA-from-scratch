#include <activation.h>

ReLUActivation::ReLUActivation(std::string name) {
	this->name = name;
}

ReLUActivation::~ReLUActivation() { }

Matrix& ReLUActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocate_memory_if_not_allocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	relu_activation_forward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														 Z.shape.x, Z.shape.y);
	NNException::throw_if_device_error_occured("Cannot perform ReLU forward propagation.");

	return A;
}

Matrix& ReLUActivation::backprop(Matrix& dA, float learning_rate) {
	dZ.allocate_memory_if_not_allocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	relu_activation_backprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
			dZ.data_device.get(), Z.shape.x, Z.shape.y);
	NNException::throw_if_device_error_occured("Cannot perform ReLU back propagation");

	return dZ;
}

SigmoidActivation::SigmoidActivation(std::string name) {
	this->name = name;
}

SigmoidActivation::~SigmoidActivation()
{ }

Matrix& SigmoidActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocate_memory_if_not_allocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	sigmoid_activation_forward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														   	Z.shape.x, Z.shape.y);
	NNException::throw_if_device_error_occured("Cannot perform sigmoid forward propagation.");


	return A;
}

Matrix& SigmoidActivation::backprop(Matrix& dA, float learning_rate) {
	dZ.allocate_memory_if_not_allocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	sigmoid_activation_backprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
			dZ.data_device.get(), Z.shape.x, Z.shape.y);
	NNException::throw_if_device_error_occured("Cannot perform sigmoid back propagation");


	return dZ;
}


LeakyReLUActivation::LeakyReLUActivation(std::string name, float alpha) : name(name), alpha(alpha) { }

LeakyReLUActivation::~LeakyReLUActivation() 
{

}

Matrix& LeakyReLUActivation::forward(Matrix& Z) {
    this->Z = Z;
    A.allocate_memory_if_not_allocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    leaky_relu_activation_forward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
            Z.shape.x, Z.shape.y, alpha);
    NNException::throw_if_device_error_occured("Cannot perform Leaky ReLU forward propagation.");

    return A;
}

Matrix& LeakyReLUActivation::backprop(Matrix& dA, float learning_rate) {
    dZ.allocate_memory_if_not_allocated(Z.shape);

    dim3 block_size(256);
    dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

    leaky_relu_activation_backprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
            dZ.data_device.get(), Z.shape.x, Z.shape.y, alpha);
    NNException::throw_if_device_error_occured("Cannot perform Leaky ReLU back propagation");

    return dZ;
}