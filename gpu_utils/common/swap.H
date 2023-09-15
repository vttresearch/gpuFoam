#pragma once

#include "cuda_host_dev.H"
#include <utility>

template <typename T> CUDA_HOSTDEV void swap(T& t1, T& t2) {
    T temp = std::move(t1); // or T temp(std::move(t1));
    t1     = std::move(t2);
    t2     = std::move(temp);
}