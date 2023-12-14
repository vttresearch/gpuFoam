#pragma once

#include "gpu_constants.H"


namespace FoamGpu {

class stepState {
public:
    const bool forward;
    gScalar    dxTry;
    gScalar    dxDid;
    bool       first;
    bool       last;
    bool       reject;
    bool       prevReject;

    CUDA_HOSTDEV stepState(const gScalar dx)
        : forward(dx > 0 ? true : false)
        , dxTry(dx)
        , dxDid(0)
        , first(true)
        , last(false)
        , reject(false)
        , prevReject(false) {}
};

} // namespace FoamGpu