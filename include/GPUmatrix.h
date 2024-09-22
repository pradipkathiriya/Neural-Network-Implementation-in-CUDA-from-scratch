#pragma once

#include <shape.h>
#include <memory>
#include <cuda.h>

class Matrix {

private:
    
    void allocate_cuda_memory();
    void allocate_host_memory();

public:
    Shape shape;
    bool device_allocated;
    bool host_allocated;

    std::shared_ptr<float> data_device;
    std::shared_ptr<float> data_host;

    Matrix(size_t x_dim = 1, size_t y_dim = 1);
    Matrix(Shape Shape);

    void allocate_memory();
    void allocate_memory_if_not_allocated(Shape Shape);
    void copy_host_to_device();
    void copy_device_to_host();

    float& operator[](const int index);
    const float& operator[](const int index) const;

};


