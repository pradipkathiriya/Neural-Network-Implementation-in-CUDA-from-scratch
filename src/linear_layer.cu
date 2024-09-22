#include <linear_layer.h>

using namespace std;

LinearLayer::LinearLayer(std::string name, Shape W_shape) :
	W(W_shape), b(W_shape.y, 1)
{
	this->name = name;
	b.allocate_memory();
	W.allocate_memory();
	initialize_bias_with_zeros();
	initialize_weights_randomly();
}

LinearLayer::~LinearLayer()
{ }

void LinearLayer::initialize_weights_randomly() {
	std::default_random_engine generator;
	std::normal_distribution<double> normal_distribution(0.0, 1.0);

	for (int x = 0; x < W.shape.x; x++) {
		for (int y = 0; y < W.shape.y; y++) {
			W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
		}
	}

	W.copy_host_to_device();
}

void LinearLayer::initialize_bias_with_zeros() {
	for (int x = 0; x < b.shape.x; x++) {
		b[x] = 0;
	}

	b.copy_host_to_device();
}

Matrix& LinearLayer::forward(Matrix& A) {
	assert(W.shape.x == A.shape.y);

	this->A = A;
	Shape Z_shape(A.shape.x, W.shape.y);
	Z.allocate_memory_if_not_allocated(Z_shape);

	compute_and_store_layer_output(A);
	NNException::throw_if_device_error_occured("Cannot perform linear layer forward propagation.");

	return Z;
}

void LinearLayer::compute_and_store_layer_output(Matrix& A) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(Z.shape.x + block_size.x - 1) / block_size.x,
						(Z.shape.y + block_size.y - 1) / block_size.y);
	linear_layer_forward<<<num_of_blocks, block_size>>>( W.data_device.get(), A.data_device.get(), Z.data_device.get(), 
			b.data_device.get(), W.shape.x, W.shape.y, A.shape.x, A.shape.y);
}

Matrix& LinearLayer::backprop(Matrix& dZ, float learning_rate) {
	dA.allocate_memory_if_not_allocated(A.shape);

	compute_and_store_back_prop_error(dZ);
	NNException::throw_if_device_error_occured("Cannot perform back propagation.");

	update_bias(dZ, learning_rate);
	NNException::throw_if_device_error_occured("Cannot perform bias update.");

	update_weights(dZ, learning_rate);
	NNException::throw_if_device_error_occured("Cannot perform weights update.");

	return dA;
}

void LinearLayer::compute_and_store_back_prop_error(Matrix& dZ) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(A.shape.x + block_size.x - 1) / block_size.x,
						(A.shape.y + block_size.y - 1) / block_size.y);
	linear_layer_back_propogation<<<num_of_blocks, block_size>>>( W.data_device.get(), dZ.data_device.get(),
			dA.data_device.get(), W.shape.x, W.shape.y, dZ.shape.x, dZ.shape.y);
}

void LinearLayer::update_weights(Matrix& dZ, float learning_rate) {
	dim3 block_size(8, 8);
	dim3 num_of_blocks(	(W.shape.x + block_size.x - 1) / block_size.x,
						(W.shape.y + block_size.y - 1) / block_size.y);
	linear_layer_update_weight<<<num_of_blocks, block_size>>>(dZ.data_device.get(), A.data_device.get(),
			W.data_device.get(), dZ.shape.x, dZ.shape.y,A.shape.x, A.shape.y, learning_rate);
}

void LinearLayer::update_bias(Matrix& dZ, float learning_rate) {
	dim3 block_size(256);
	dim3 num_of_blocks( (dZ.shape.y * dZ.shape.x + block_size.x - 1) / block_size.x);
	linear_layer_update_bias<<<num_of_blocks, block_size>>>(dZ.data_device.get(), b.data_device.get(), dZ.shape.x, dZ.shape.y,
			b.shape.x, learning_rate);
}

int LinearLayer::get_x_dim() const {
	return W.shape.x;
}

int LinearLayer::get_y_dim() const {
	return W.shape.y;
}

Matrix& LinearLayer::get_weight_matrix() {
    return W;
}

Matrix& LinearLayer::get_bias_vector() {
    return b;
}