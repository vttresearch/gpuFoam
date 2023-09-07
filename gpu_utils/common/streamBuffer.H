#pragma once

#include <vector>
#include <thrust/device_vector.h>


struct streamBuffer{

    streamBuffer() = default;

    streamBuffer(size_t n) : streams_(n){
        for (auto& s : streams_){
            cudaStreamCreate(&s);
        }
    }

    ~streamBuffer() {
        for (auto& s : streams_){
            cudaStreamDestroy(s);
        }
    }

    void syncAll() {
        for (auto& s : streams_){
            cudaStreamSynchronize(s);
        }
    }

    void sync(size_t i){
        cudaStreamSynchronize(this->operator()(i));
    }


    cudaStream_t& operator()(size_t i) {
        return streams_[i % streams_.size()];
    }
    const cudaStream_t& operator()(size_t i) const {
        return streams_[i % streams_.size()];
    }


private:
    std::vector<cudaStream_t> streams_;

};
