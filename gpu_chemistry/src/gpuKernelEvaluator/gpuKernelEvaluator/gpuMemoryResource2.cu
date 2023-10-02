#include "gpuMemoryResource2.H"
#include <thrust/device_malloc_allocator.h>

using labelAllocator  = thrust::device_malloc_allocator<gLabel>;
using scalarAllocator = thrust::device_malloc_allocator<gScalar>;

namespace FoamGpu {

gpuMemoryResource2::gpuMemoryResource2(gLabel nCells, gLabel nSpecie)
    : memoryResource(nCells, nSpecie) {
    this->allocate();
}

gpuMemoryResource2::~gpuMemoryResource2() { this->deallocate(); }

void gpuMemoryResource2::allocate() {

    labelAllocator  lAllocator;
    scalarAllocator sAllocator;

    for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
        labelData_[i] =
            make_raw_pointer(lAllocator.allocate(nEqns() * nCells()));
    }
    for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
        scalarData_[i] =
            make_raw_pointer(sAllocator.allocate(nEqns() * nCells()));
    }
    for (gLabel i = 0; i < N_TWOD_SCALAR_ARRAYS; ++i) {
        twodScalarData_[i] =
            make_raw_pointer(sAllocator.allocate(nEqns() * nEqns() * nCells()));
    }
}

void gpuMemoryResource2::deallocate() {


    labelAllocator  lAllocator;
    scalarAllocator sAllocator;

    // TODO: these casts dont make any sense because of the templated
    //       allocators
    for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
        auto ptr = make_device_pointer(labelData_[i]);
        lAllocator.deallocate(ptr, nEqns() * nCells());
    }
    for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
        auto ptr = make_device_pointer(scalarData_[i]);
        sAllocator.deallocate(ptr, nEqns() * nCells());
    }
    for (gLabel i = 0; i < N_TWOD_SCALAR_ARRAYS; ++i) {
        auto ptr = make_device_pointer(twodScalarData_[i]);
        sAllocator.deallocate(ptr, nEqns() * nEqns() * nCells());
    }

}

} // namespace FoamGpu