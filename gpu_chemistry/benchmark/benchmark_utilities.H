#pragma once

#include <vector>
#include "gpu_constants.H"

static inline
std::vector<gScalar> make_random_y0s(gLabel nCells, gLabel nEqns)
{
    using namespace FoamGpu;

    gLabel nSpecie = nEqns - 2;
    std::vector<gScalar> ret(nCells * nEqns);
    auto gri_y = std::vector<gScalar>(nEqns);
    assign_test_condition(gri_y);
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
std::vector<gScalar> make_gri_y0s(gLabel nCells, gLabel nEqns)
{
    using namespace FoamGpu;

    std::vector<gScalar> ret(nCells * nEqns);
    auto gri_y = std::vector<gScalar>(nEqns);
    assign_test_condition(gri_y);
    auto yvf = make_mdspan(ret, extents<2>{nCells, nEqns});
    for (gLabel i = 0; i < nCells; ++i){

        for (gLabel j = 0; j < nEqns; ++j){
            yvf(i, j) = gri_y[j];
        }

    }
    return ret;
}


static inline
std::vector<gScalar> make_random_rhos(gLabel nCells)
{
    using namespace FoamGpu;
    std::vector<gScalar> ret(nCells, 1.0);
    //fill_random(ret, 0.9, 1.3);
    return ret;
}
static inline
std::vector<gScalar> make_random_deltaTChem(gLabel nCells)
{
    using namespace FoamGpu;
    std::vector<gScalar> ret(nCells, 1E-7);
    //fill_random(ret, BenchmarkParams::chemdeltaTMin, BenchmarkParams::chemDeltaTmax);
    return ret;
}


