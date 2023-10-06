#include "gpuMemoryResource.H"
#include <thrust/device_malloc_allocator.h>

using labelAllocator  = thrust::device_malloc_allocator<gLabel>;
using scalarAllocator = thrust::device_malloc_allocator<gScalar>;

namespace FoamGpu {

gpuMemoryResource::gpuMemoryResource(gLabel nCells, gLabel nSpecie)
    : memoryResource(nCells, nSpecie) {
    this->allocate();
}

gpuMemoryResource::~gpuMemoryResource() { this->deallocate(); }

void gpuMemoryResource::allocate() {

    labelAllocator  lAllocator;
    scalarAllocator sAllocator;

    for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
        labelData_[i] =
            make_raw_pointer(lAllocator.allocate(labelArrayLength()));
    }
    for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
        scalarData_[i] =
            make_raw_pointer(sAllocator.allocate(scalarArrayLength()));
    }
    for (gLabel i = 0; i < N_TWOD_SCALAR_ARRAYS; ++i) {
        twodScalarData_[i] =
            make_raw_pointer(sAllocator.allocate(twodScalarArrayLength()));
    }
}

void gpuMemoryResource::deallocate() {

    labelAllocator  lAllocator;
    scalarAllocator sAllocator;

    for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
        auto ptr = make_device_pointer(labelData_[i]);
        lAllocator.deallocate(ptr, labelArrayLength());
    }
    for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
        auto ptr = make_device_pointer(scalarData_[i]);
        sAllocator.deallocate(ptr, scalarArrayLength());
    }
    for (gLabel i = 0; i < N_TWOD_SCALAR_ARRAYS; ++i) {
        auto ptr = make_device_pointer(twodScalarData_[i]);
        sAllocator.deallocate(ptr, twodScalarArrayLength());
    }
}

void gpuMemoryResource::resize(gLabel nCells, gLabel nSpecie) {

    if (shouldReallocate(nCells, nSpecie)) {
        this->deallocate();
        nCells_  = nCells;
        nSpecie_ = nSpecie;
        nEqns_   = nSpecie + 2;
        this->allocate();
    }
}

} // namespace FoamGpu