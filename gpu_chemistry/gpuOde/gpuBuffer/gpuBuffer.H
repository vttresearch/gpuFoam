#pragma once

#include "cuda_host_dev.H"
#include "gpuConstants.H"
#include "mdspan.H"

namespace FoamGpu {

/// @brief A buffer of memory to solve a single chemistry ode
struct gpuBuffer {

    gpuBuffer() = default;

    CUDA_HOSTDEV
    gpuBuffer(gLabel nSpecie)
        : nSpecie_(nSpecie)
        , nEqns_(nSpecie + 2) {}

    // Gpu rosenbrock solve
    std::array<gScalar, SPECIE_MAX + 2> dydx0_;
    std::array<gScalar, SPECIE_MAX + 2> yTemp_;

    // Gpu rosenbrock solve2
    std::array<gScalar, SPECIE_MAX + 2> dydx_;
    std::array<gScalar, SPECIE_MAX + 2> dfdx_;

    std::array<gLabel, SPECIE_MAX + 2>  pivotIndices_;
    std::array<gScalar, SPECIE_MAX + 2> k1_;
    std::array<gScalar, SPECIE_MAX + 2> k2_;
    std::array<gScalar, SPECIE_MAX + 2> k3_;
    std::array<gScalar, SPECIE_MAX + 2> k4_;
    std::array<gScalar, SPECIE_MAX + 2> err_;
    std::array<gScalar, SPECIE_MAX + 2> lubuffer_;

    std::array<gScalar, SPECIE_MAX>     c_;
    std::array<gScalar, SPECIE_MAX + 2> tempField1_;
    std::array<gScalar, SPECIE_MAX + 2> tempField2_;

    gLabel nSpecie_;
    gLabel nEqns_;

    CUDA_HOSTDEV auto dydx0() {
        return make_mdspan(dydx0_, extents<1>{nEqns_});
    }
    CUDA_HOSTDEV auto yTemp() {
        return make_mdspan(yTemp_, extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto dydx() { return make_mdspan(dydx_, extents<1>{nEqns_}); }
    CUDA_HOSTDEV auto dfdx() { return make_mdspan(dfdx_, extents<1>{nEqns_}); }

    CUDA_HOSTDEV auto pivotIndices() {
        return make_mdspan(pivotIndices_, extents<1>{nEqns_});
    }
    CUDA_HOSTDEV auto k1() { return make_mdspan(k1_, extents<1>{nEqns_}); }
    CUDA_HOSTDEV auto k2() { return make_mdspan(k2_, extents<1>{nEqns_}); }
    CUDA_HOSTDEV auto k3() { return make_mdspan(k3_, extents<1>{nEqns_}); }
    CUDA_HOSTDEV auto k4() { return make_mdspan(k4_, extents<1>{nEqns_}); }
    CUDA_HOSTDEV auto err() { return make_mdspan(err_, extents<1>{nEqns_}); }
    CUDA_HOSTDEV auto lubuffer() {
        return make_mdspan(lubuffer_, extents<1>{nEqns_});
    }

    CUDA_HOSTDEV auto c() { return make_mdspan(c_, extents<1>{nSpecie_}); } //!

    CUDA_HOSTDEV auto tempField1() {
        return make_mdspan(tempField1_, extents<1>{nEqns_});
    }
    CUDA_HOSTDEV auto tempField2() {
        return make_mdspan(tempField2_, extents<1>{nEqns_});
    }
};

} // namespace FoamGpu