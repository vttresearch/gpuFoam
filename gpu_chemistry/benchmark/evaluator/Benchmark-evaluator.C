#define CATCH_CONFIG_ENABLE_BENCHMARKING
//#define CATCH_CONFIG_MAIN
#include "catch.H"

#include "gpuKernelEvaluator.H"
#include "readGpuOdeInputs.H"
#include "test_utilities.H"
#include "create_inputs.H"
#include "benchmark_utilities.H"




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


static inline Foam::dictionary make_dict(std::string solverName)
{
    Foam::dictionary dict;
    dict.add("solver", solverName);
    dict.add("absTol", BenchmarkParams::absTol);
    dict.add("relTol", BenchmarkParams::relTol);
    return dict;
}

//std::vector<gScalar> make_random_

FoamGpu::GpuKernelEvaluator make_evaluator
(
    gLabel nCells, const Foam::dictionary& odeDict, TestData::Mechanism m)
{
    using namespace FoamGpu;
    auto thermos = TestData::makeGpuThermos(m);
    auto reactions = TestData::makeGpuReactions(m);
    gLabel nSpecie = TestData::speciesCount(m);
    gLabel nEqns = TestData::equationCount(m);

    auto inputs = read_gpuODESolverInputs(odeDict);


    return GpuKernelEvaluator
    (
        nCells,
        nEqns,
        nSpecie,
        thermos,
        reactions,
        inputs
    );

}

static inline void warmup()
{
    auto mech = TestData::H2;
    const gScalar deltaT = BenchmarkParams::deltaT;
    const gScalar deltaTChemMax = BenchmarkParams::deltaTChemMax;
    const gLabel nCells = BenchmarkParams::nCells;

    const auto rho = make_random_rhos(nCells);
    const auto deltaTChem = make_random_deltaTChem(nCells);
    const auto Yvf = make_tutorial_y0s(nCells, mech);
    //const auto Yvf = make_random_y0s(nCells, TestData::equationCount(mech));

    //No idea why this is necessary, but without it, the first benchmark results are way off
    BENCHMARK_ADVANCED("WARMUP")(Catch::Benchmark::Chronometer meter) {
        auto dict = make_dict("Rosenbrock23");
        auto eval = make_evaluator(nCells, dict, mech);
        meter.measure([&] { return callGpuSolve(deltaT, deltaTChemMax, rho, deltaTChem, Yvf, eval);});
    };
}

static inline
void runBenchmarks(TestData::Mechanism mech)
{
    const gScalar deltaT = BenchmarkParams::deltaT;
    const gScalar deltaTChemMax = BenchmarkParams::deltaTChemMax;
    const gLabel nCells = 1E4;

    const auto rho = make_random_rhos(nCells);
    const auto deltaTChem = make_random_deltaTChem(nCells);
    const auto Yvf = make_tutorial_y0s(nCells, mech);

    BENCHMARK_ADVANCED("Rosenbrock12")(Catch::Benchmark::Chronometer meter) {
        auto dict = make_dict("Rosenbrock12");
        auto eval = make_evaluator(nCells, dict, mech);
        meter.measure([&] { return callGpuSolve(deltaT, deltaTChemMax, rho, deltaTChem, Yvf, eval);});
    };

    BENCHMARK_ADVANCED("Rosenbrock23")(Catch::Benchmark::Chronometer meter) {
        auto dict = make_dict("Rosenbrock23");
        auto eval = make_evaluator(nCells, dict, mech);
        meter.measure([&] { return callGpuSolve(deltaT, deltaTChemMax, rho, deltaTChem, Yvf, eval);});
    };


    BENCHMARK_ADVANCED("Rosenbrock34")(Catch::Benchmark::Chronometer meter) {
        auto dict = make_dict("Rosenbrock34");
        auto eval = make_evaluator(nCells, dict, mech);
        meter.measure([&] { return callGpuSolve(deltaT, deltaTChemMax, rho, deltaTChem, Yvf, eval);});
    };
}

TEST_CASE("Benchmark warmup")
{
    warmup();
}
TEST_CASE("Benchmark GpuKernelEvaluator (H2)")
{
    runBenchmarks(TestData::H2);
}

TEST_CASE("Benchmark GpuKernelEvaluator (GRI)")
{
    runBenchmarks(TestData::GRI);
}
