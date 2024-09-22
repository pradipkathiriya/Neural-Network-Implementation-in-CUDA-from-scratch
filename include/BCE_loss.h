#pragma once
#include <GPUmatrix.h>
#include <nn_exception.h>
#include <assert.h>
#include <utility.h>

class BCELoss {
    public:
        float cost(Matrix prediction, Matrix target);
        Matrix d_cost(Matrix prection, Matrix target, Matrix dY);
};