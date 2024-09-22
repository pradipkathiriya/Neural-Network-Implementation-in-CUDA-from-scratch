#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <nn_layer.h>
#include <iostream>
#include <nn_exception.h>
#include <cassert>
#include <utility.h>

class LinearLayer : public NNLayer {

    private:
        const float weights_init_threshold = 0.01;

        Matrix W;
        Matrix b;

        Matrix Z;
        Matrix A;
        Matrix dA;

        void initialize_bias_with_zeros();
        void initialize_weights_randomly();

        void compute_and_store_back_prop_error(Matrix& dZ);
        void compute_and_store_layer_output(Matrix& A);

        void update_weights(Matrix& dZ, float learning_rate);
        void update_bias(Matrix& dZ, float learning_rate);

    public:
        LinearLayer(std::string name, Shape W_shape);
        ~LinearLayer();

        Matrix& forward(Matrix& A);
        Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

        int get_x_dim() const;
        int get_y_dim() const;

        Matrix& get_weight_matrix();
        Matrix& get_bias_vector();

};

#endif