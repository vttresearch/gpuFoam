#include "gpuKernelEvaluator.H"

#include <iostream>

#include "cuda_host_dev.H"
#include "host_device_vectors.H"
#include "gpuBuffer.H"

#include <thrust/execution_policy.h>
#include <thrust/extrema.h> //min_element
#include <thrust/host_vector.h>

namespace FoamGpu {


template<class S1, class S2, class S3, class S4>
struct singleCell{

    singleCell(gScalar      deltaT,
               gLabel       nSpecie,
               S1           deltaTChem,
               S2           Yvf,
               S3           Jss,
               S4           buffer,
               gpuODESolver ode)
        : deltaT_(deltaT)
        , nSpecie_(nSpecie)
        , deltaTChem_(deltaTChem)
        , Yvf_(Yvf)
        , Jss_(Jss)
        , buffer_(buffer)
        , ode_(ode){}

    CUDA_HOSTDEV void operator()(gLabel celli) const {
        auto Y = mdspan<gScalar, 1>(&Yvf_(celli, 0), extents<1>{nSpecie_ + 2});
        auto J = mdspan<gScalar, 2>(&Jss_(celli, 0, 0),
                                    extents<2>{nSpecie_ + 2, nSpecie_ + 2});

        // Initialise time progress
        gScalar timeLeft = deltaT_;

        constexpr gLabel li = 0;

        // Calculate the chemical source terms
        while (timeLeft > gpuSmall) {
            gScalar dt = timeLeft;

            ode_.solve(0, dt, Y, li, deltaTChem_[celli], J, buffer_[celli]);

            for (int i = 0; i < nSpecie_; i++) { Y[i] = std::max(0.0, Y[i]); }

            timeLeft -= dt;
        }
        }

    gScalar deltaT_;
    gLabel nSpecie_;
    S1 deltaTChem_;
    S2 Yvf_;
    S3 Jss_;
    S4 buffer_;
    gpuODESolver ode_;
};


template <class Op>
__global__ void cuda_kernel(Op op, gLabel nCells) {

    int celli = blockIdx.x * blockDim.x + threadIdx.x;
    if (celli < nCells)
    {
        op(celli);
    }
}


std::pair<std::vector<gScalar>, std::vector<gScalar>>
GpuKernelEvaluator::computeYNew(gScalar                     deltaT,
                                gScalar                     deltaTChemMax,
                                const std::vector<gScalar>& deltaTChem,
                                const std::vector<gScalar>& Y) {

    const gLabel nCells = deltaTChem.size();

    // Convert thermos and reactions from host to device
    const auto dThermos   = toDeviceVector(hThermos_);
    const auto dReactions = toDeviceVector(hReactions_);

    gpuODESystem odeSystem(nEqns_,
                           dReactions.size(),
                           thrust::raw_pointer_cast(dThermos.data()),
                           thrust::raw_pointer_cast(dReactions.data()));

    gpuODESolver ode = make_gpuODESolver(odeSystem, odeInputs_);

    // Convert fields from host to device
    auto ddeltaTChem_arr = toDeviceVector(deltaTChem);
    auto dYvf_arr        = toDeviceVector(Y);
    auto ddeltaTChem     = make_mdspan(ddeltaTChem_arr, extents<1>{nCells});
    auto dYvf            = make_mdspan(dYvf_arr, extents<2>{nCells, nEqns_});

    device_vector<gScalar> Js(nCells * nEqns_ * nEqns_, 0.0);

    auto Jss = make_mdspan(Js, extents<3>{nCells, nEqns_, nEqns_});

    device_vector<gpuBuffer> buffer_arr =
        host_vector<gpuBuffer>(nCells, gpuBuffer(nSpecie_));

    auto buffer = make_mdspan(buffer_arr, extents<1>{nCells});

    singleCell op(deltaT, nSpecie_, ddeltaTChem, dYvf, Jss, buffer, ode);
    thrust::for_each
    (
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nCells),
        op
    );


    /*
    singleCell op(deltaT, nSpecie_, ddeltaTChem, dYvf, Jss, buffer, ode);
    gLabel NTHREADS = 32;
    gLabel NBLOCKS  = (nCells + NTHREADS - 1) / NTHREADS;
    cuda_kernel<<<NBLOCKS, NTHREADS>>>(op, nCells);
    */

    return std::make_pair(toStdVector(dYvf_arr), toStdVector(ddeltaTChem_arr));
}

std::tuple<std::vector<gScalar>, std::vector<gScalar>, gScalar>
GpuKernelEvaluator::computeRR(gScalar                    deltaT,
                              gScalar                    deltaTChemMax,
                              const std::vector<gScalar> rho,
                              const std::vector<gScalar> deltaTChem,
                              const std::vector<gScalar> Y) {

    const gLabel nCells = rho.size();

    auto pair = computeYNew(deltaT, deltaTChemMax, deltaTChem, Y);
    auto YNew_arr = std::get<0>(pair);
    auto deltaTChemNew = std::get<1>(pair);

    auto YNew = make_mdspan(YNew_arr, extents<2>{nCells, nEqns_});
    auto Y0 = make_mdspan(Y, extents<2>{nCells, nEqns_});

    std::vector<gScalar> RR_arr(nCells * nSpecie_);
    auto                 RR = make_mdspan(RR_arr, extents<2>{nCells, nSpecie_});

    for (gLabel j = 0; j < nCells; ++j) {
        for (gLabel i = 0; i < nSpecie_; ++i) {

            RR(j, i) = rho[j] * (YNew(j, i) - Y0(j, i)) / deltaT;
        }
    }

    gScalar deltaTMin =
        *std::min_element(deltaTChemNew.begin(), deltaTChemNew.end());

    for (auto& e : deltaTChemNew) { e = std::min(e, deltaTChemMax); }

    return std::make_tuple(RR_arr, deltaTChemNew, deltaTMin);
}

} // namespace FoamGpu