#pragma once

#include "gpu_constants.H"

namespace FoamGpu {

class gpuLindemannFallOffFunction {

public:
    gpuLindemannFallOffFunction() = default;

    inline CUDA_HOSTDEV gScalar operator()(const gScalar T,
                                           const gScalar Pr) const {
        return 1.0;
    }

    inline CUDA_HOSTDEV gScalar ddT(const gScalar T,
                                    const gScalar Pr,
                                    const gScalar F) const {
        return 0.0;
    }

    inline CUDA_HOSTDEV gScalar ddPr(const gScalar T,
                                     const gScalar Pr,
                                     const gScalar F) const {
        return 0.0;
    }
};

} // namespace FoamGpu
