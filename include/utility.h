#pragma once

#include <cuda.h>

 __global__ void binary_cross_entropy_loss(float* prediction, float*target, int size, float* cost);

 __global__ void d_binary_cross_entropy_loss(float* prediction, float* target, float* dY, int size);

 __global__ void linear_layer_forward(float* W, float* A, float*Z, float* b, int W_x_dim, 
        int W_y_dim, int A_x_dim, int A_y_dim);

__global__ void linear_layer_back_propogation(float* W, float*dZ, float* dA, int W_x_dim,
        int W_y_dim, int dZ_x_dim, int dZ_y_dim);

__global__ void linear_layer_update_weight(float* dZ, float* A, float* W,int dZ_x_dim, 
        int dZ_y_dim, int A_x_dim, int A_y_dim, float learning_rate);

__global__ void linear_layer_update_bias(  float* dZ, float* b, int dZ_x_dim, int dZ_y_dim,
            int b_x_dim, float learning_rate) ;

__device__ float sigmoid(float x);

__global__ void sigmoid_activation_forward(float* Z, float* A, int Z_x_dim, int Z_y_dim);

__global__ void sigmoid_activation_backprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim);

__global__ void relu_activation_forward(float* Z, float* A, int Z_x_dim, int Z_y_dim);

__global__ void relu_activation_backprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim);

__global__ void leaky_relu_activation_forward(float* Z, float* A, int width, int height, float alpha);

__global__ void leaky_relu_activation_backprop(float* Z, float* dA, float* dZ, int width, int height, float alpha);