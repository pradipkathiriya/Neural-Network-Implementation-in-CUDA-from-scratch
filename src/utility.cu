#include <utility.h>

__global__ void d_binary_cross_entropy_loss(float* prediction, float* target, float* dY, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        float pred = prediction[index];
        pred = fmaxf(fminf(pred, 1.0f - 1e-7), 1e-7);
        dY[index] = -1.0 * (target[index] / pred - (1 - target[index]) / (1 - pred));
    }

}

__global__ void binary_cross_entropy_loss(float* prediction, float*target, int size, float* cost)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {

        float pred = prediction[index];
        pred = fmaxf(fminf(pred, 1.0f - 1e-7), 1e-7);

        float partial_cost = target[index] * logf(pred)
            + (1.0f - target[index]) * logf(1.0f - pred);
        atomicAdd(cost, -partial_cost / size);
    }

}

__global__ void linear_layer_forward(float* W, float* A, float* Z, float* b,
        int W_x_dim, int W_y_dim, int A_x_dim, int A_y_dim) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_y_dim;

    float Z_value = 0;

    if (row < Z_y_dim && col < Z_x_dim) {
        for (int i = 0; i < W_x_dim; i++) {
            Z_value = W[row * W_x_dim + i] * A[i * A_x_dim + col];
        }
        Z[row * Z_x_dim + col] = Z_value + b[row];
    }

}

__global__ void linear_layer_back_propogation(float* W, float* dZ, float *dA, int W_x_dim, 
        int W_y_dim, int dZ_x_dim, int dZ_y_dim) 

{

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int dA_x_dim = dZ_x_dim;
    int dA_y_dim = W_x_dim;

    float dA_value = 0.0f;

    if (row < dA_y_dim && col < dA_x_dim) {
        for (int i = 0; i < W_y_dim; i++) {
            dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
        }
        dA[row * dA_x_dim + col] = dA_value;
    }
}


__global__ void linear_layer_update_weight(  float* dZ, float* A, float* W,int dZ_x_dim, 
        int dZ_y_dim, int A_x_dim, int A_y_dim, float learning_rate) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // A is treated as transposed
    int W_x_dim = A_y_dim;
    int W_y_dim = dZ_y_dim;

    float dW_value = 0.0f;

    if (row < W_y_dim && col < W_x_dim) {
        for (int i = 0; i < dZ_x_dim; i++) {
            dW_value += dZ[row * dZ_x_dim + i] * A[col * A_x_dim + i];
        }
        W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (dW_value / A_x_dim);
    }
}

__global__ void linear_layer_update_bias(  float* dZ, float* b, int dZ_x_dim, int dZ_y_dim,
            int b_x_dim, float learning_rate) 
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dZ_x_dim * dZ_y_dim) {
        int dZ_x = index % dZ_x_dim;
        int dZ_y = index / dZ_x_dim;
        atomicAdd(&b[dZ_y], - learning_rate * (dZ[dZ_y * dZ_x_dim + dZ_x] / dZ_x_dim));
    }
}

__device__ float sigmoid(float x) {
    return 1.0f / (1 + exp(-x));
}

__global__ void sigmoid_activation_forward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        A[index] = sigmoid(Z[index]);
    }
}

__global__ void sigmoid_activation_backprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        dZ[index] = dA[index] * sigmoid(Z[index]) * (1 - sigmoid(Z[index]));
    }
}

__global__ void relu_activation_forward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        A[index] = fmaxf(Z[index], 0);
    }
}

__global__ void relu_activation_backprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        if (Z[index] > 0) {
            dZ[index] = dA[index];
        }
        else {
            dZ[index] = 0;
        }
    }
} 

__global__ void leaky_relu_activation_forward(float* Z, float* A, int width, int height, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        A[idx] = Z[idx] > 0 ? Z[idx] : alpha * Z[idx];
    }
}

__global__ void leaky_relu_activation_backprop(float* Z, float* dA, float* dZ, int width, int height, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        dZ[idx] = Z[idx] > 0 ? dA[idx] : alpha * dA[idx];
    }
}