#include "catch.H"


#include "gpuKernelEvaluator.H"
#include "test_utilities.H"
#include "create_foam_inputs.H"
#include "create_gpu_inputs.H"

TEST_CASE("Test GpuKernelEvaluator")
{
    using namespace FoamGpu;

    /*
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

        GpuKernelEvaluator evaluator(nCells, nEqns, nSpecie, thermos, reactions, inputs);

        gScalar deltaT = 1e-3;
        gScalar deltaTChemMax = deltaT/4;
        std::vector<gScalar> deltaTChem(deltaT/5, nCells);


        std::vector<gScalar> rho(1.0, nCells);


        std::vector<gScalar> Yvf(nCells*nEqns);

        auto s = make_mdspan(Yvf, extents<2>{nCells, nEqns});

        for (gLabel celli = 0; celli < nCells; ++celli) {
            for (gLabel i = 0; i < nSpecie; ++i) {
                s(celli, i) = 0.1; //concentration like
            }
            s(celli, nSpecie)     = 300.0; //T
            s(celli, nSpecie + 1) = 1E5; //p
        }





        auto tuple = evaluator.computeRR
        (
            deltaT,
            deltaTChemMax,
            rho,
            deltaTChem,
            Yvf
        );

        auto RR = std::get<0>(tuple);
        auto newDts = std::get<1>(tuple);
        gScalar minDt = std::get<2>(tuple);

        CHECK(minDt != gScalar(0));


        //REQUIRE_NOTHROW(GpuKernelEvaluator(nCells, nEqns, nSpecie, thermos, reactions, inputs));

    }
    */




}