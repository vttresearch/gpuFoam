#pragma once

#include "range.hpp"
#include "runtime_assert.hpp"

namespace topaz {

template <class HostRange, class DeviceRange>
inline void host_to_device(const HostRange& src, DeviceRange& dst) {

    using value_type = typename HostRange::value_type;
    const auto size  = src.size();
    runtime_assert(size == dst.size(), "Size mismatch in host to device copy");
    const auto bytesize = size * sizeof(value_type);

    cudaMemcpy(thrust::raw_pointer_cast(dst.data()),
               thrust::raw_pointer_cast(src.data()),
               bytesize,
               cudaMemcpyHostToDevice);
}

template <class HostRange, class DeviceRange>
inline void async_host_to_device(const HostRange& src,
                                 DeviceRange&     dst,
                                 cudaStream_t     stream) {

    using value_type = typename HostRange::value_type;
    const auto size  = src.size();
    runtime_assert(size == dst.size(), "Size mismatch in host to device copy");
    const auto bytesize = size * sizeof(value_type);

    cudaMemcpyAsync(thrust::raw_pointer_cast(dst.data()),
                    thrust::raw_pointer_cast(src.data()),
                    bytesize,
                    cudaMemcpyHostToDevice,
                    stream);
}

template <class HostRange, class DeviceRange>
inline void device_to_host(const DeviceRange& src, HostRange& dst) {

    using value_type = typename DeviceRange::value_type;
    const auto size  = src.size();
    runtime_assert(size == dst.size(), "Size mismatch in device to host copy");
    const auto bytesize = size * sizeof(value_type);

    cudaMemcpy(thrust::raw_pointer_cast(dst.data()),
               thrust::raw_pointer_cast(src.data()),
               bytesize,
               cudaMemcpyDeviceToHost);
}

template <class HostRange, class DeviceRange>
inline void async_device_to_host(const DeviceRange& src,
                                 HostRange&         dst,
                                 cudaStream_t       stream) {

    using value_type = typename DeviceRange::value_type;
    const auto size  = src.size();
    runtime_assert(size == dst.size(), "Size mismatch in device to host copy");
    const auto bytesize = size * sizeof(value_type);

    cudaMemcpyAsync(thrust::raw_pointer_cast(dst.data()),
                    thrust::raw_pointer_cast(src.data()),
                    bytesize,
                    cudaMemcpyDeviceToHost,
                    stream);
}


} // namespace topaz