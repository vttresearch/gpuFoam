#define CATCH_CONFIG_ENABLE_BENCHMARKING
//#define CATCH_CONFIG_MAIN
#include "catch.H"

#include "gpuKernelEvaluator.H"
#include "readGpuOdeInputs.H"
#include "test_utilities.H"
#include "benchmark_utilities.H"

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

        auto r = callGpuSolve
            (
                deltaT,
                deltaTChemMax,
                rho,
                deltaTChem,
                Yvf,
                eval
            );

        std::cout << std::get<2>(r) << std::endl;
    }


    /*
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
    */

}