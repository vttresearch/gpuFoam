#pragma once

#include <variant>

#include "gpuODESystem.H"
#include "gpuRosenbrock23.H"
#include "gpuRosenbrock34.H"


namespace FoamGpu{

namespace gpu{

    //using system = gpuODESystem;

    using solver = variant::variant
    <
        gpuRosenbrock23<gpuODESystem>,
        gpuRosenbrock34<gpuODESystem>
    >;

}

template<class Span, class TwoDSpan>
struct ode_visitor
{
    CUDA_HOSTDEV
    ode_visitor
    (
        const gScalar xStart,
        const gScalar xEnd,
        Span&         y,
        gScalar&      dxTry,
        TwoDSpan&     a,
        gpuBuffer&     buffer
    ) :
    xStart_(xStart),
    xEnd_(xEnd),
    y_(y),
    dxTry_(dxTry),
    a_(a),
    buffer_(buffer)
    {}

    template<class T>
    CUDA_HOSTDEV void operator()(const T& i)
    {
        i.solve(xStart_, xEnd_, y_, 0, dxTry_, a_, buffer_);
    }

    gScalar xStart_;
    gScalar xEnd_;
    Span&         y_;
    gScalar&      dxTry_;
    TwoDSpan&     a_;
    gpuBuffer&     buffer_;


};

struct gpuODESolver{

    gpu::solver solver_;

    gpuODESolver() = default;

    gpuODESolver(const gpu::solver& solver) : solver_(solver)
    {}

    template<class Span, class TwoDSpan>
    CUDA_HOSTDEV void solve(const gScalar xStart,
                            const gScalar xEnd,
                            Span&         y,
                            const gLabel  li,
                            gScalar&      dxTry,
                            TwoDSpan&     a,
                            gpuBuffer&     buffer) const
    {
        (void) li;
        ode_visitor<Span, TwoDSpan> v(xStart, xEnd, y, dxTry, a, buffer);
        variant::apply_visitor(v, solver_);
    }


};



}



