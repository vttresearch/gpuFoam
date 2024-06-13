#pragma once

#include <cstddef>

#include "cuda_host_dev.H"

template<class T, size_t N>
struct device_array{


    T data_[N];


    using size_type = size_t;
    using value_type = T;
    using pointer = value_type*;
    using iterator = pointer;
    using reference = value_type&;
    using const_reference = const value_type&;

    inline constexpr CUDA_HOSTDEV size_t size() const {return N;}


    inline constexpr CUDA_HOSTDEV pointer data() {return data_;}
    inline constexpr CUDA_HOSTDEV auto data() const {return data_;}


    inline constexpr CUDA_HOSTDEV iterator begin()       {return data_;}
    inline constexpr CUDA_HOSTDEV iterator begin() const {return data_;}

    inline constexpr CUDA_HOSTDEV iterator end()       {return begin() + N;}
    inline constexpr CUDA_HOSTDEV iterator end() const {return begin() + N;}


    inline constexpr CUDA_HOSTDEV reference operator[](size_t i) {return data_[i];}
    inline constexpr CUDA_HOSTDEV const_reference operator[](size_t i) const {return data_[i];}


};

