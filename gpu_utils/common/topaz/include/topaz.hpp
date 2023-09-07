#pragma once


#if defined(__NVCOMPILER) || defined(__NVCC__)
    #define __NVIDIA_COMPILER__
#endif

#ifdef __NVIDIA_COMPILER__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif
#include "begin_end.hpp"
#include "traits.hpp"
#include "range.hpp"
#include "zip_range.hpp"
#include "zip.hpp"
#include "transform.hpp"
#include "numeric_array.hpp"
#include "arithmetic_ops.hpp"
#include "parallel_force_evaluate.hpp"
#include "copy.hpp"
#include "runtime_assert.hpp"

#ifdef __NVIDIA_COMPILER__
#include "device_host_copy.hpp"
#endif
