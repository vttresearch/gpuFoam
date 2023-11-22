#pragma once

#include "gpu_constants.H"
#include <array>

namespace FoamGpu {

class gpuThirdBodyEfficiencies {
public:
    using effArray = std::array<gScalar, 300>;

    gLabel   nSpecie_;
    effArray efficiencies_{};

    gpuThirdBodyEfficiencies() = default;

    inline gpuThirdBodyEfficiencies(gLabel          nSpecie,
                                    const effArray& efficiencies)
        : nSpecie_(nSpecie)
        , efficiencies_(efficiencies) {}

    ///
    ///@brief Computes the concentration of the third-bodies.
    ///
    ///@param c Concentration.
    ///@return gScalar The concentration of the third-bodies.
    ///
    inline CUDA_HOSTDEV gScalar M(const mdspan<const gScalar, 1> c) const {
        gScalar M = 0;
        for (gLabel i = 0; i < nSpecie_; ++i) { M += efficiencies_[i] * c[i]; }
        return M;
    }

    ///
    ///@brief Calculates the derivative of M w.r.t concentrations.
    ///
    ///@param c Concentration.
    ///@return std::array<gScalar, 300> The derivative of M w.r.t concentrations. 
    ///
    inline CUDA_HOSTDEV const effArray&
    dMdc(const mdspan<const gScalar, 1> c) const {
        (void)c;
        return efficiencies_;
    }
};

} // namespace FoamGpu