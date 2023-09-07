#include "gpuKernelEvaluator.H"

#include <iostream>

#include "cuda_host_dev.H"
#include "gpuBuffer.H"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h> //min_element
#include <thrust/execution_policy.h>


namespace FoamGpu
{

template<class T>
using host_vector = thrust::host_vector<T>;

template<class T>
using device_vector = thrust::device_vector<T>;


template<class Container>
static inline auto toDevice(const Container& c)
{
    using value = typename Container::value_type;
    host_vector<value> temp(c.begin(), c.end());
    return device_vector<value>(temp.begin(), temp.end());
}

template<class T>
static inline std::vector<T> toStdVec(const device_vector<T>& c)
{
    host_vector<T> temp(c.begin(), c.end());
    return std::vector<T>(temp.begin(), temp.end());
}


static inline __host__ void query_memory_limits()
{
    size_t stack_size = 0;
    cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);

    printf("Stack size in doubles: %d\n", int(stack_size/sizeof(gScalar)));


    size_t malloc_size = 0;
    cudaDeviceGetLimit(&malloc_size, cudaLimitMallocHeapSize);

    printf("Malloc size in doubles: %d\n", int(malloc_size/sizeof(gScalar)));

}



template<class S1, class S2, class S3, class S4>
__global__
void cuda_kernel
(
    gScalar deltaT,
    gLabel nCells,
    gLabel nSpecie,
    S1 deltaTChem,
    S2 Yvf,
    S3 Jss,
    S4 buffer,
    gpuRosenbrock34<gpuODESystem> ode
)
{


    int celli = blockIdx.x*blockDim.x + threadIdx.x;
    if (celli >= nCells) return;

    auto Y = mdspan<gScalar, 1>(&Yvf(celli, 0), extents<1>{nSpecie + 2});
    auto J = mdspan<gScalar, 2>(&Jss(celli, 0, 0), extents<2>{nSpecie+2, nSpecie+2});



    // Initialise time progress
    gScalar timeLeft = deltaT;

    constexpr gLabel li = 0;


    // Calculate the chemical source terms
    while (timeLeft > gpuSmall)
    {
        gScalar dt = timeLeft;

        ode.solve(0, dt, Y, li, deltaTChem[celli], J, buffer[celli]);


        for (int i=0; i<nSpecie; i++)
        {
            Y[i] = std::max(0.0, Y[i]);
        }

        timeLeft -= dt;
    }


}

std::pair<std::vector<gScalar>, std::vector<gScalar>>
GpuKernelEvaluator::computeYNew
(
    gScalar deltaT,
    gScalar deltaTChemMax,
    const std::vector<gScalar>& deltaTChem,
    const std::vector<gScalar>& Y
)
{
    const gLabel nCells = deltaTChem.size();
    const auto d_thermos = toDevice(h_thermos_);
    const auto d_reactions = toDevice(h_reactions_);


    gpuODESystem odeSystem
    (
        nEqns_,
        d_reactions.size(),
        //d_thermos.data(),
        //d_reactions.data()
        thrust::raw_pointer_cast(d_thermos.data()),
        thrust::raw_pointer_cast(d_reactions.data())
    );


    gpuRosenbrock34<gpuODESystem> ode
    (
        odeSystem,
        odeInputs_
    );

    auto d_deltaTChem = toDevice(deltaTChem);
    auto d_Yvf = toDevice(Y);

    device_vector<gScalar> Js(nCells*nEqns_*nEqns_, 0.0);

    auto Jss = make_mdspan(Js, extents<3>{nCells, nEqns_, nEqns_});

    device_vector<gpuBuffer> buffer_arr
        = host_vector<gpuBuffer>(nCells, gpuBuffer(nSpecie_));

    auto buffer = make_mdspan(buffer_arr, extents<1>{nCells});

    auto deltaTChem_span = make_mdspan(d_deltaTChem, extents<1>{nCells});
    auto Yvf = make_mdspan(d_Yvf, extents<2>{nCells, nEqns_});

    gLabel NTHREADS = 32;
    gLabel NBLOCKS = (nCells + NTHREADS - 1)/ NTHREADS;
    //kernel<<<(nCells()+255)/256, 256>>>(nCells(), op);
    //kernel<<<NBLOCKS, NTHREADS>>>(nCells_, op);

    cuda_kernel<<<NBLOCKS, NTHREADS>>>
    (
        deltaT,
        nCells,
        nSpecie_,
        deltaTChem_span,
        Yvf,
        Jss,
        buffer,
        ode
    );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        assert(0);
    }

    return std::make_pair
    (
        toStdVec(d_Yvf),
        toStdVec(d_deltaTChem)
    );

}

std::tuple<std::vector<gScalar>, std::vector<gScalar>, gScalar>
GpuKernelEvaluator::computeRR
(
    gScalar deltaT,
    gScalar deltaTChemMax,
    const std::vector<gScalar> rho,
    const std::vector<gScalar> p,
    const std::vector<gScalar> T,
    const std::vector<gScalar> deltaTChem,
    const std::vector<gScalar> Y
)
{

    const gLabel nCells = rho.size();


    auto [YNew_arr, deltaTChemNew]
        = computeYNew(deltaT, deltaTChemMax, deltaTChem, Y);

    auto YNew = make_mdspan(YNew_arr, extents<2>{nCells, nEqns_});

    auto Y0 = make_mdspan(Y, extents<2>{nCells, nEqns_});

    std::vector<gScalar> RR_arr(nCells * nSpecie_);
    auto RR = make_mdspan(RR_arr, extents<2>{nCells, nSpecie_});

    for (gLabel j = 0; j < nCells; ++j){
    for (gLabel i = 0; i < nSpecie_; ++i){

        RR(j, i) = rho[j]*(YNew(j, i) - Y0(j, i))/deltaT;

    }}


    gScalar deltaTMin
        = *std::min_element
        (
            deltaTChemNew.begin(),deltaTChemNew.end()
        );


    for (auto& e : deltaTChemNew)
    {
        e = std::min(e, deltaTChemMax);
    }


    return std::make_tuple(RR_arr, deltaTChemNew, deltaTMin);


}




}