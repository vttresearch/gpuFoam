#pragma once

#include "gpu_fields.H"
#include "async_evaluate.H"



void async_set_zero
(
    cudaStream_t stream,
    gpuScalarField& f
)
{
    auto kernel = topaz::make_constant_range(double(0), f.size());
    async_evaluate
    (
        stream,
        kernel.begin(),
        kernel.end(),
        f.begin()
    );
}
