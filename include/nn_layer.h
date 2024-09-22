#pragma once

#include <iostream>
#include <GPUmatrix.h>

class NNLayer {

public:
    std::string name;

    virtual ~NNLayer() = 0;

    virtual Matrix& forward(Matrix& A) = 0;
    virtual Matrix& backprop(Matrix& dZ, float learning_rate) = 0;

    std::string get_name()
    {
        return this->name;
    }

};

inline NNLayer::~NNLayer() {}