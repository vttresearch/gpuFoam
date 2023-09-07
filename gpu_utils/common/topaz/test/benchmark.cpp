#define CATCH_CONFIG_ENABLE_BENCHMARKING
#define CATCH_CONFIG_MAIN // This tells the catch header to generate a main
#include "catch.hpp"

#include "topaz.hpp"

#ifdef __NVIDIA_COMPILER__
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
template<class T>
using vector_t = thrust::device_vector<T>;

template<class T>
using NVec_t = topaz::NumericArray<T, thrust::device_malloc_allocator<T>>;

#else
#include <vector>
template<class T>
using vector_t = std::vector<T>;

template<class T>
using NVec_t = topaz::NumericArray<T, std::allocator<T>>;
#endif


template<class Scalar_t, class Vector_t>
Vector_t nxpy(const Scalar_t& s, const Vector_t& x, const Vector_t& y){
    return s*x + y;
}

template<class Vector_t>
Vector_t arithmetic1(const Vector_t& v1, const Vector_t& v2, const Vector_t& v3){
    using T = typename Vector_t::value_type;
    return v1 * v2 + T(43) / v1 * v3 - v1 - T(32);

}

TEST_CASE("Benchmark NumericArray"){

    using namespace topaz;

    SECTION("Saxpy"){


        auto do_benchmark = [] (size_t n){
            float s = float(43.213123);
            NVec_t<float> x(n, float(3234.32));
            NVec_t<float> y(n, float(-31131.444444));
            std::string name = "Saxpy n = " + std::to_string(n);
            BENCHMARK(name.c_str()) {
                return nxpy(s, x, y);
            };
        };

        do_benchmark(10);
        do_benchmark(100);
        do_benchmark(1000);
        //do_benchmark(1E5);
    }

    SECTION("Arithmetic1"){


        auto do_benchmark = [] (size_t n){
            NVec_t<float> x(n, float(3234.32));
            NVec_t<float> y(n, float(-31131.444444));
            NVec_t<float> z(n, float(-31131.444444));
            std::string name = "Arithmetic1 n = " + std::to_string(n);
            BENCHMARK(name.c_str()) {
                return arithmetic1(x,y,z);
            };
        };

        do_benchmark(10);
        do_benchmark(100);
        do_benchmark(1000);
        //do_benchmark(1E5);
    }

}
