#pragma once

#include "cuda_host_dev.H"
#include <type_traits>

#ifdef __NVIDIA_COMPILER__
#include <thrust/device_ptr.h>
#endif

#ifdef __NVIDIA_COMPILER__

template <class T> static constexpr inline auto make_raw_pointer(T p) {
    return thrust::raw_pointer_cast(p);
}

template <class T> static constexpr inline auto make_device_pointer(T p) {
    return thrust::device_pointer_cast(p);
}

#else

template <class T> static constexpr inline auto make_raw_pointer(T p) {
    static_assert(std::is_pointer_v<T>, "Not a pointer type.");
    return p;
}

template <class T> static constexpr inline auto make_device_pointer(T p) {
    static_assert(std::is_pointer_v<T>, "Not a pointer type.");
    return p;
}
#endif
