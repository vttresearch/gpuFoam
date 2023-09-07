#pragma once


namespace topaz {

#ifdef __NVCC__
#define DISABLE_HOST_DEV_WARNING #pragma nv_exec_check_disable
#else
#define DISABLE_HOST_DEV_WARNING
#endif

DISABLE_HOST_DEV_WARNING
template <typename Container>
inline CUDA_HOSTDEV auto adl_begin(Container& c) {
    return c.begin();
}

DISABLE_HOST_DEV_WARNING
template <typename Container>
inline CUDA_HOSTDEV auto adl_begin(const Container& c) {
    return c.begin();
}

DISABLE_HOST_DEV_WARNING
template <typename Container>
inline CUDA_HOSTDEV auto adl_end(Container& c) {
    return c.end();
}

DISABLE_HOST_DEV_WARNING
template <typename Container>
inline CUDA_HOSTDEV auto adl_end(const Container& c) {
    return c.end();
}

DISABLE_HOST_DEV_WARNING
template <typename Container>
inline CUDA_HOSTDEV auto adl_size(const Container& c) {
    return adl_end(c) - adl_begin(c);
}

} // namespace topaz