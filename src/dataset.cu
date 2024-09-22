#include <dataset.h> 

CoordinatesDataset::CoordinatesDataset(size_t batch_size, size_t number_of_batches) :
	batch_size(batch_size), number_of_batches(number_of_batches)
{
	for (int i = 0; i < number_of_batches; i++) {
		batches.push_back(Matrix(Shape(batch_size, 2)));
		targets.push_back(Matrix(Shape(batch_size, 1)));

		batches[i].allocate_memory();
		targets[i].allocate_memory();

		for (int k = 0; k < batch_size; k++) {
			batches[i][k] = static_cast<double>(rand()) / RAND_MAX - 0.5;
			batches[i][batches[i].shape.x + k] = static_cast<double>(rand()) / RAND_MAX - 0.5;;

			if ( (batches[i][k] > 0 && batches[i][batches[i].shape.x + k] > 0) ||
				 ((batches[i][k] < 0 && batches[i][batches[i].shape.x + k] < 0)) ) {
				targets[i][k] = 1;
			}
			else {
				targets[i][k] = 0;
			}
		}

		batches[i].copy_host_to_device();
		targets[i].copy_host_to_device();
	}
}

int CoordinatesDataset::get_num_of_batches() {
	return number_of_batches;
}

std::vector<Matrix>& CoordinatesDataset::get_batches() {
	return batches;
}

std::vector<Matrix>& CoordinatesDataset::get_targets() {
	return targets;
}