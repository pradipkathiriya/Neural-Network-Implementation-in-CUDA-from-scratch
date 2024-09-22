# pragma once

#include <exception>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>



class NNException : std::exception {
private:
    const char* exception_msg;
    bool device_allocated;
    bool host_allocated;


public:
    NNException(const char* exception_msg) :
    exception_msg(exception_msg)
    {
    }

    virtual const char* what() const noexcept {
        return exception_msg;
    }

    static void throw_if_device_error_occured(const char* exception_msg) {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << error << ": " << exception_msg;
            throw NNException(exception_msg);
        }
    }
};


