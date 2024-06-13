#pragma once

#include <vector>
#include <iterator>



#include "device_allocate.H"
#include "device_free.H"
#include "error_handling.H"
#include "host_device_transfers.H"
#include "cuda_host_dev.H"

/*
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
template<class T>
using device_vector = thrust::device_vector<T>;


template<class T>
static inline std::vector<T> toStdVector(const device_vector<T>& v){

    thrust::host_vector<T> tmp(v.begin(), v.end());
    return std::vector<T>(tmp.begin(), tmp.end());
}
*/


template<class T>
struct device_vector{

    using size_type = size_t;
    using value_type = T;
    using pointer = value_type*;
    using iterator = pointer;
    using reference = value_type&;
    using const_reference = const value_type&;


    device_vector() : size_(0), data_(nullptr)
    {}

    explicit device_vector(size_type size) : size_(size), data_(device_allocate<T>(size))
    {}


    explicit device_vector(const std::vector<T>& host )
    : device_vector(host.size())
    {
        host_to_device(host.begin(), host.end(), this->begin());
    }



    template<class Iter>
    device_vector(Iter h_begin, Iter h_end) :
    device_vector(static_cast<size_t>(std::distance(h_begin, h_end)))
    {
        host_to_device(h_begin, h_end, this->begin());
    }


    ~device_vector(){
        if (data_ != nullptr) {
            device_free(data_);
            size_ = 0;
            data_ = nullptr;
        }
    }

    inline size_type size()       { return size_; }
    inline size_type size() const { return size_; }


    pointer data() const { return data_; }
    pointer data()       { return data_; }

    iterator begin()       {return data_;}
    iterator begin() const {return data_;}

    iterator end()       {return data_ + size_;}
    iterator end() const {return data_ + size_;}


private:

    size_type size_;
    pointer data_;


};


template<class T>
static inline std::vector<T> toStdVector(const device_vector<T>& v){

    std::vector<T> ret(v.size());
    device_to_host(v.begin(), v.end(), ret.begin());
    return ret;

}