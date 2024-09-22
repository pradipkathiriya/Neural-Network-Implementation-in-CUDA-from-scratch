#ifndef DATASET_H
#define DATASET_H

#include <GPUmatrix.h>
#include <vector>

class CoordinatesDataset {
private:
	size_t batch_size;
	size_t number_of_batches;

	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

public:

	CoordinatesDataset(size_t batch_size, size_t number_of_batches);

	int get_num_of_batches();
	std::vector<Matrix>& get_batches();
	std::vector<Matrix>& get_targets();

};  

#endif