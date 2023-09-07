#pragma once

#include "gpuConstants.H"
#include <array>

namespace FoamGpu {

class gpuThirdBodyEfficiencies {
public:
    using effArray = std::array<gScalar, SPECIE_MAX>;

    gLabel   nSpecie_;
    effArray efficiencies_{};

    gpuThirdBodyEfficiencies() = default;

    inline gpuThirdBodyEfficiencies(gLabel          nSpecie,
                                    const effArray& efficiencies)
        : nSpecie_(nSpecie)
        , efficiencies_(efficiencies) {}

    //- Calculate and return M, the concentration of the third-bodies
    template <class Span> inline CUDA_HOSTDEV gScalar M(const Span c) const {
        gScalar M = 0;
        for (gLabel i = 0; i < nSpecie_; ++i) { M += efficiencies_[i] * c[i]; }
        return M;
    }

    //- Calculate and return the derivative of M, w.r.t. the species
    //  concentrations
    template <class Span>
    inline CUDA_HOSTDEV const effArray& dMdc(const Span c) const {
        (void)c;
        return efficiencies_;
    }
};

} // namespace FoamGpu