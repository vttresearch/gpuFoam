#pragma once

#include <vector>
#include "gpu_constants.H"
#include "create_inputs.H"

struct BenchmarkParams{

    static constexpr gScalar deltaT = 1E-6;
    static constexpr gScalar deltaTChemMax = 1E8;
    static constexpr gScalar chemdeltaTMin = 1E-7;
    static constexpr gScalar chemDeltaTmax = 1E-6;
    static constexpr gLabel nCells = 1000;

    static constexpr gScalar absTol = 1E-1;
    static constexpr gScalar relTol = 1E-12;

};

static inline
std::vector<gScalar> make_random_y0s(gLabel nCells, gLabel nEqns)
{
    using namespace FoamGpu;

    gLabel nSpecie = nEqns - 2;
    std::vector<gScalar> ret(nCells * nEqns);
    auto yvf = make_mdspan(ret, extents<2>{nCells, nEqns});
    for (gLabel i = 0; i < nCells; ++i){
        for (gLabel j = 0; j < nSpecie; ++j){
            yvf(i, j) = random_number(0.01, 0.435);
        }
        yvf(i, nSpecie) = random_number(500.0, 1000.0);
        yvf(i, nSpecie+1) = random_number(1E5, 1.2E5);
    }
    return ret;
}
static inline
std::vector<gScalar> make_tutorial_y0s(gLabel nCells, TestData::Mechanism mech)
{
    using namespace FoamGpu;
    const gLabel nEqns = TestData::equationCount(mech);
    std::vector<gScalar> ret(nCells * nEqns);
    auto y0 = std::vector<gScalar>(nEqns);
    assign_test_condition(y0, mech);
    auto yvf = make_mdspan(ret, extents<2>{nCells, nEqns});
    for (gLabel i = 0; i < nCells; ++i){

        for (gLabel j = 0; j < nEqns; ++j){
            yvf(i, j) = y0[j];
        }

    }
    return ret;
}


static inline
std::vector<gScalar> make_random_rhos(gLabel nCells)
{
    using namespace FoamGpu;
    std::vector<gScalar> ret(nCells, 1.0);
    fill_random(ret, 0.9, 1.3);
    return ret;
}
static inline
std::vector<gScalar> make_random_deltaTChem(gLabel nCells)
{
    using namespace FoamGpu;
    std::vector<gScalar> ret(nCells, 1E-7);
    fill_random(ret, BenchmarkParams::chemdeltaTMin, BenchmarkParams::chemDeltaTmax);
    return ret;
}


