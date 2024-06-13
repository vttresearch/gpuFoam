#include "device_allocate.H"
#include "device_free.H"
#include "gpuMemoryResource.H"

namespace FoamGpu {

gpuMemoryResource::gpuMemoryResource(gLabel nCells, gLabel nSpecie)
    : memoryResource(nCells, nSpecie) {
    this->allocate();
}

gpuMemoryResource::~gpuMemoryResource() { this->deallocate(); }

void gpuMemoryResource::allocate() {

    for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
        labelData_[i] = device_allocate<gLabel>(labelArrayLength());
    }
    for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
        scalarData_[i] =
            device_allocate<gScalar>(scalarArrayLength());
    }
    for (gLabel i = 0; i < N_TWOD_SCALAR_ARRAYS; ++i) {
        twodScalarData_[i] =
            device_allocate<gScalar>(twodScalarArrayLength());
    }
}

void gpuMemoryResource::deallocate() {

    for (gLabel i = 0; i < N_LABEL_ARRAYS; ++i) {
        device_free(labelData_[i]);
    }
    for (gLabel i = 0; i < N_SCALAR_ARRAYS; ++i) {
        device_free(scalarData_[i]);
    }
    for (gLabel i = 0; i < N_TWOD_SCALAR_ARRAYS; ++i) {
        device_free(twodScalarData_[i]);
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