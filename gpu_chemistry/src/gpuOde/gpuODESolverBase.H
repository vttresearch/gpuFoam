#pragma once

#include <string>
#include "cuda_host_dev.H"
#include "double_intrinsics.H"
#include "gpuODESystem.H"

namespace FoamGpu {

template<class Derived>
class gpuODESolverBase {

protected:

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




    // Protected data

    //- Absolute convergence tolerance per step
    gScalar absTol_;

    //- Relative convergence tolerance per step
    gScalar relTol_;

    //- The maximum number of sub-steps allowed for the integration step
    gLabel maxSteps_;

    CUDA_HOSTDEV
    gScalar normaliseError(gScalar y0, gScalar y, gScalar err) const {

        double tol = absTol_ + relTol_ * std::max(std::abs(y0), std::abs(y));
        double ret = std::abs(err) / tol;
        return ret;
    }


    template <class Span1, class Span2, class Span3>
    CUDA_HOSTDEV gScalar normaliseError(const Span1& y0,
                                        const Span2& y,
                                        const Span3& err) const {
        // Calculate the maximum error
        gScalar maxErr = 0.0;

        for (gLabel i = 0; i < gLabel(y.size()); ++i) {

            gScalar t = normaliseError(y0[i], y[i], err[i]);
            maxErr    = std::max(maxErr, t);
        }

        return maxErr;
    }

private:

    //CRTP interface for Derived::solve
    template <class Span, class TwoDSpan>
    CUDA_HOSTDEV void interface(gScalar&     x,
                            Span&        y,
                            const gLabel li,
                            stepState&   step,
                            TwoDSpan&    a,
                            gpuBuffer&   buffer) const {

        static_cast<const Derived*>(this)->solve(x, y, li, step, a, buffer);
    }


public:


    // Constructors
    gpuODESolverBase() = default;

    //- Construct for given ODESystem specifying tolerances
    gpuODESolverBase(const gpuODESolverInputs& i)
        : absTol_(i.absTol)
        , relTol_(i.relTol)
        , maxSteps_(i.maxSteps) {}

    //- Solve the ODE system from the current state xStart, y
    //  and the optional index into the list of systems to solve li
    //  to xEnd and return an estimate for the next step in dxTry
    template <class Span, class TwoDSpan>
    CUDA_HOSTDEV void solve(const gScalar xStart,
                            const gScalar xEnd,
                            Span&         y,
                            const gLabel  li,
                            gScalar&      dxTry,
                            TwoDSpan&     a,
                            gpuBuffer&     buffer) const {

        stepState step(dxTry);
        gScalar   x = xStart;

        // printf("dxTry value = %lf\n", dxTry);

        for (gLabel nStep = 0; nStep < maxSteps_; nStep++) {

            // Store previous iteration dxTry
            gScalar dxTry0 = step.dxTry;

            step.reject = false;

            // Check if this is a truncated step and set dxTry to integrate to
            // xEnd
            if ((x + step.dxTry - xEnd) * (x + step.dxTry - xStart) > 0) {
                step.last  = true;
                step.dxTry = xEnd - x;
            }

            // Integrate as far as possible up to step.dxTry
            interface(x, y, li, step, a, buffer);

            // Check if reached xEnd
            if ((x - xEnd) * (xEnd - xStart) >= 0) {
                if (nStep > 0 && step.last) { step.dxTry = dxTry0; }

                dxTry = step.dxTry;

                return;
            }

            step.first = false;

            // If the step.dxTry was reject set step.prevReject
            if (step.reject) { step.prevReject = true; }
        }
    }



    //- Return access to the absolute tolerance field
    inline CUDA_HOSTDEV gScalar& absTol() { return absTol_; }

    //- Return access to the relative tolerance field
    inline CUDA_HOSTDEV gScalar& relTol() { return relTol_; }
};

} // namespace FoamGpu
