#pragma once

#include <variant>

#include "gpuODESystem.H"
#include "gpuRosenbrock12.H"
#include "gpuRosenbrock23.H"
#include "gpuRosenbrock34.H"

namespace FoamGpu {

namespace gpu {

// using system = gpuODESystem;

using solver = variant::variant<gpuRosenbrock12<gpuODESystem>,
                                gpuRosenbrock23<gpuODESystem>,
                                gpuRosenbrock34<gpuODESystem>>;

} // namespace gpu

struct ode_visitor {
    CUDA_HOSTDEV
    ode_visitor(const gScalar      xStart,
                const gScalar      xEnd,
                mdspan<gScalar, 1> y,
                gScalar&           dxTry,
                gpuBuffer&         buffer)
        : xStart_(xStart)
        , xEnd_(xEnd)
        , y_(y)
        , dxTry_(dxTry)
        , buffer_(buffer) {}

    template <class T> CUDA_HOSTDEV void operator()(const T& i) {
        odeSolve(i, xStart_, xEnd_, y_, dxTry_, buffer_);
    }

    gScalar            xStart_;
    gScalar            xEnd_;
    mdspan<gScalar, 1> y_;
    gScalar&           dxTry_;
    gpuBuffer&         buffer_;
};

struct gpuODESolver {

    gpu::solver solver_;

    gpuODESolver() = default;

    gpuODESolver(const gpu::solver& solver)
        : solver_(solver) {}

    CUDA_HOSTDEV void solve(const gScalar      xStart,
                            const gScalar      xEnd,
                            mdspan<gScalar, 1> y,
                            gScalar&           dxTry,
                            gpuBuffer&         buffer) const {
        ode_visitor v(xStart, xEnd, y, dxTry, buffer);
        variant::apply_visitor(v, solver_);
    }
};

} // namespace FoamGpu
