#include <BCE_loss.h>
#include <utility.h>

using namespace std;

float BCELoss::cost(Matrix prediction, Matrix target) {
    assert(prediction.shape.x == target.shape.x);

    float* cost;

    cudaMallocManaged(&cost, sizeof(float));

    dim3 block_size(256);
    dim3 num_of_block((prediction.shape.x + block_size.x - 1) / block_size.x);

    binary_cross_entropy_loss<<<num_of_block, block_size>>>(prediction.data_device.get(),
            target.data_device.get(), prediction.shape.x, cost);

    cudaDeviceSynchronize();
    NNException::throw_if_device_error_occured("can not compute binary cross entropy cose");

    float cost_value = *cost;
    cudaFree(cost);

    return cost_value;

}

Matrix BCELoss::d_cost(Matrix prediction, Matrix target, Matrix dY) {
    assert(prediction.shape.x == target.shape.x);

    dim3 block_size(256);
    dim3 num_of_blocks((prediction.shape.x + block_size.x - 1) / block_size.x);
    d_binary_cross_entropy_loss<<<num_of_blocks, block_size>>>(prediction.data_device.get(),
                 target.data_device.get(), dY.data_device.get(), prediction.shape.x);
    NNException::throw_if_device_error_occured("Cannot compute derivative for binary cross entropy.");

 return dY;
}

