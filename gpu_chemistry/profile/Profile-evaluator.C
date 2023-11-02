#include "gpuKernelEvaluator.H"
#include "readGpuOdeInputs.H"
#include "test_utilities.H"
#include "create_gpu_inputs.H"
#include "benchmark_utilities.H"

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

static inline Foam::dictionary make_dict(std::string solverName)
{
    Foam::dictionary dict;
    dict.add("solver", solverName);
    dict.add("absTol", BenchmarkParams::absTol);
    dict.add("relTol", BenchmarkParams::relTol);
    return dict;
}

static inline
auto runProfile(TestData::Mechanism mech, gLabel nTimes)
{
    using namespace FoamGpu;

    const gScalar deltaT = BenchmarkParams::deltaT;
    const gScalar deltaTChemMax = BenchmarkParams::deltaTChemMax;
    const gLabel nCells = BenchmarkParams::nCells;

    const auto rho = make_random_rhos(nCells);
    const auto deltaTChem = make_random_deltaTChem(nCells);
    const auto Yvf = make_tutorial_y0s(nCells, mech);

    auto dict = make_dict("Rosenbrock12");
    auto eval = make_evaluator(nCells, dict, mech);
    (void) nTimes;

    auto r = eval.computeRR(deltaT, deltaTChemMax, rho, deltaTChem, Yvf);

    auto v1 = std::get<0>(r);

    auto sum = std::accumulate(v1.begin(), v1.end(), gScalar(0));

    return sum;


}


int main() {

    auto r = runProfile(TestData::GRI, 1);
    std::cout << r << std::endl;
    return 0;

}