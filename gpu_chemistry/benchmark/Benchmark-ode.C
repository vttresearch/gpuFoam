#define CATCH_CONFIG_ENABLE_BENCHMARKING
//#define CATCH_CONFIG_MAIN
#include "catch.H"

#include "gpuKernelEvaluator.H"
#include "readGpuOdeInputs.H"
#include "test_utilities.H"

struct BenchmarkParams{

    static constexpr gScalar deltaT = 1E-6;
    static constexpr gScalar deltaTChemMax = 1E8;  
    static constexpr gScalar chemdeltaTMin = 1E-7;
    static constexpr gScalar chemDeltaTmax = 1E-6;
    static constexpr gLabel nCells = 2;

};


auto callGpuSolve
(
    gScalar deltaT,
    gScalar deltaTChemMax,
    const std::vector<gScalar>& rho,
    const std::vector<gScalar>& deltaTChem,
    const std::vector<gScalar>& Yvf,
    FoamGpu::GpuKernelEvaluator& eval
)
{
    using namespace FoamGpu;
    return eval.computeRR(deltaT, deltaTChemMax, rho, deltaTChem, Yvf);
}

std::vector<gScalar> make_random_y0s(gLabel nCells, gLabel nEqns)
{
    using namespace FoamGpu;

    gLabel nSpecie = nEqns - 2;
    std::vector<gScalar> ret(nCells * nEqns);
    auto gri_y = std::vector<gScalar>(nEqns);
    assign_test_condition(gri_y);
    auto yvf = make_mdspan(ret, extents<2>{nCells, nEqns});
    for (gLabel i = 0; i < nCells; ++i){
        //for (gLabel j = 0; j < nSpecie; ++j){
        //    yvf(i, j) = random_number(0.01, 0.435);
        //}
        //yvf(i, nSpecie) = random_number(500.0, 1000.0);
        //yvf(i, nSpecie+1) = random_number(1E5, 1.2E5);

        for (gLabel j = 0; j < nEqns; ++j){
            yvf(i, j) = gri_y[j];
        }

    }
    return ret;
}
std::vector<gScalar> make_random_rhos(gLabel nCells)
{
    using namespace FoamGpu;
    std::vector<gScalar> ret(nCells, 1.0);
    //fill_random(ret, 0.9, 1.3);
    return ret;
}

std::vector<gScalar> make_random_deltaTChem(gLabel nCells)
{
    using namespace FoamGpu;
    std::vector<gScalar> ret(nCells, 1E-7);
    //fill_random(ret, BenchmarkParams::chemdeltaTMin, BenchmarkParams::chemDeltaTmax);
    return ret;
}



//std::vector<gScalar> make_random_

FoamGpu::GpuKernelEvaluator make_evaluator(const Foam::dictionary& odeDict)
{
    using namespace FoamGpu;
    auto thermos = makeGpuThermos();
    auto reactions = makeGpuReactions();
    gLabel nSpecie = make_species_table().size();
    gLabel nEqns = nSpecie + 2;

   
    auto inputs = read_gpuODESolverInputs(odeDict);

    return GpuKernelEvaluator
    (
        nEqns,
        nSpecie,
        thermos,
        reactions,
        inputs
    );

}

TEST_CASE("Benchmark GpuKernelEvaluator")
{
    const gScalar deltaT = BenchmarkParams::deltaT;
    const gScalar deltaTChemMax = BenchmarkParams::deltaTChemMax;
    const gLabel nCells = BenchmarkParams::nCells;
    gLabel nSpecie = FoamGpu::make_species_table().size();
    gLabel nEqns = nSpecie + 2;


    const auto rho = make_random_rhos(nCells);
    const auto deltaTChem = make_random_deltaTChem(nCells);
    const auto Yvf = make_random_y0s(nCells, nEqns);

    
    {
        Foam::dictionary dict;
        dict.add("solver", "Rosenbrock23");
        auto eval = make_evaluator(dict);

        BENCHMARK("Rosenbrock23")
        {
            return callGpuSolve
            (
                deltaT,
                deltaTChemMax,
                rho,
                deltaTChem,
                Yvf,
                eval
            );

        };
    }


    
    {
        Foam::dictionary dict;
        dict.add("solver", "Rosenbrock34");
        auto eval = make_evaluator(dict);

        BENCHMARK("Rosenbrock34")
        {
            return callGpuSolve
            (
                deltaT,
                deltaTChemMax,
                rho,
                deltaTChem,
                Yvf,
                eval
            );

        };
    }

}