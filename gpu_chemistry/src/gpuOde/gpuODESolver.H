#pragma once

#include <variant>

#include "gpuODESystem.H"
#include "gpuRosenbrock23.H"
#include "gpuRosenbrock34.H"

namespace FoamGpu {

namespace gpu {

// using system = gpuODESystem;

using solver = variant::variant<gpuRosenbrock23<gpuODESystem>,
                                gpuRosenbrock34<gpuODESystem>>;

} // namespace gpu

struct ode_visitor {
    CUDA_HOSTDEV
    ode_visitor(const gScalar       xStart,
                const gScalar       xEnd,
                mdspan<gScalar, 1>& y,
                gScalar&            dxTry,
                mdspan<gScalar, 2>& a,
                gpuBuffer&          buffer)
        : xStart_(xStart)
        , xEnd_(xEnd)
        , y_(y)
        , dxTry_(dxTry)
        , a_(a)
        , buffer_(buffer) {}

    template <class T> CUDA_HOSTDEV void operator()(const T& i) {
        i.solve(xStart_, xEnd_, y_, 0, dxTry_, a_, buffer_);
    }

    gScalar             xStart_;
    gScalar             xEnd_;
    mdspan<gScalar, 1>& y_;
    gScalar&            dxTry_;
    mdspan<gScalar, 2>& a_;
    gpuBuffer&          buffer_;
};

struct gpuODESolver {

    gpu::solver solver_;

    gpuODESolver() = default;

    gpuODESolver(const gpu::solver& solver)
        : solver_(solver) {}

    CUDA_HOSTDEV void solve(const gScalar       xStart,
                            const gScalar       xEnd,
                            mdspan<gScalar, 1> y,
                            const gLabel        li,
                            gScalar&            dxTry,
                            mdspan<gScalar, 2> a,
                            gpuBuffer&          buffer) const {
        (void)li;
        ode_visitor v(xStart, xEnd, y, dxTry, a, buffer);
        variant::apply_visitor(v, solver_);
    }
};

} // namespace FoamGpu
