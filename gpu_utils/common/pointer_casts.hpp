#pragma once

#include "cuda_host_dev.H"
#include <type_traits>

#ifdef __USING_GPU__
#include <thrust/device_ptr.h>
#endif

#ifdef __USING_GPU__

template <class T> static constexpr inline auto make_raw_pointer(T p) {
    return thrust::raw_pointer_cast(p);
}

template <class T> static constexpr inline auto make_device_pointer(T p) {
    return thrust::device_pointer_cast(p);
}

#else

template <class T> static constexpr inline auto make_raw_pointer(T p) {
    static_assert(std::is_pointer<T>::value, "Not a pointer type.");
    return p;
}

template <class T> static constexpr inline auto make_device_pointer(T p) {
    static_assert(std::is_pointer<T>::value, "Not a pointer type.");
    return p;
}
#endif
