#include <GPUmatrix.h>
#include <nn_exception.h>

using namespace std;

Matrix::Matrix(size_t x_dim, size_t y_dim) :
    shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
    device_allocated(false), host_allocated(false)
    {

    }

Matrix::Matrix(Shape shape):
    Matrix(shape.x, shape.y)
    {

    }

void Matrix::allocate_cuda_memory() {
    if (!device_allocated) {
        float* device_memory = nullptr;
        cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
        NNException::throw_if_device_error_occured("can not allcoate memory for tensor");
        data_device = std::shared_ptr<float>(device_memory,
        [&](float* ptr){cudaFree(ptr);});
        device_allocated = true;
    }
}

void Matrix::allocate_host_memory() {
    if (!host_allocated) {
        data_host = std::shared_ptr<float>(new float[shape.x * shape.y],
                [&](float* ptr){delete[] ptr;});
        host_allocated = true;
    }
}

void Matrix::allocate_memory() {
    allocate_cuda_memory();
    allocate_host_memory();
}

void Matrix::allocate_memory_if_not_allocated(Shape shape) {
    if (!device_allocated && !host_allocated) {
        this->shape = shape;
        allocate_memory();
    }
}

void Matrix::copy_host_to_device() {
    if (device_allocated && host_allocated) {
        cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice );
        NNException::throw_if_device_error_occured("Cannot copy host data to CUDA device.");
    } else {
        throw NNException("Cannot copy device data to not allocated memory on host.");
    }
}

void Matrix::copy_device_to_host() {

    if (device_allocated && host_allocated) {
        cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost );
        NNException::throw_if_device_error_occured("Cannot copy device data to host.");
    } else {
        throw NNException("Cannot copy device data to not allocated memory on host.");
 }
}

float& Matrix::operator[](const int index) {
    return data_host.get()[index];
}

const float& Matrix::operator[](const int index) const {
    return data_host.get()[index];
}
