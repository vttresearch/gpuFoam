#include "catch.H"


#include "gpuKernelEvaluator.H"
#include "test_utilities.H"
#include "create_inputs.H"

TEST_CASE("Test GpuKernelEvaluator")
{
    using namespace FoamGpu;

    SECTION("Constructors")
    {
        REQUIRE_NOTHROW(GpuKernelEvaluator());

        const auto m = TestData::GRI;
        auto thermos = TestData::makeGpuThermos(m);
        auto reactions = TestData::makeGpuReactions(m);
        gLabel nSpecie = TestData::speciesCount(m);
        gLabel nEqns = TestData::equationCount(m);
        gLabel nCells = 10;
        gpuODESolverInputs inputs;
        inputs.name = "Rosenbrock34";

        REQUIRE_NOTHROW(GpuKernelEvaluator(nCells, nEqns, nSpecie, thermos, reactions, inputs));




    }




}