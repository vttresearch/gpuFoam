
#include "catch.H"
#include "test_utilities.H"
#include "create_foam_inputs.H"
#include "create_gpu_inputs.H"
#include "mock_of_odesystem.H"
#include "gpuODESystem.H"
#include "makeGpuOdeSolver.H"
#include "readGpuOdeInputs.H"
#include "Rosenbrock34.H"
#include "Rosenbrock23.H"
#include "mdspan.H"






struct TestParams{
    const gScalar xStart;
    const gScalar xEnd;
    const gScalar dxTry;
};

auto callGpuSolve
(
    const Foam::scalarField& y0,
    const FoamGpu::gpuODESolver& ode,
    TestParams p
)
{
    using namespace FoamGpu;

    const gLabel nEqns = y0.size();
    const gLabel nSpecie = nEqns - 2;

    const gScalar xStart = p.xStart;
    const gScalar xEnd = p.xEnd;
    const gScalar dxTry = p.dxTry;

    auto y = toDeviceVector(y0);

    memoryResource_t memory(1, nSpecie);
    auto buffers = toDeviceVector(splitToBuffers(memory));


    auto f = [
        ode = ode,
        xStart = xStart,
        xEnd = xEnd,
        y = make_mdspan(y, extents<1>{nEqns}),
        dxTry = dxTry,
        buffers = make_mdspan(buffers, extents<1>{1})
    ]()
    {
        gScalar dxTry_temp = dxTry;
        ode.solve(xStart, xEnd, y, dxTry_temp, buffers[0]);
        return dxTry_temp;
    };

    auto unused = eval(f);
    (void) unused;

    auto ret = toStdVector(y);

    //Round small values to zero to avoid -0 == 0 comparisons
    for (auto& e : ret)
    {
        if (std::abs(e) < gpuSmall)
        {
            e = 0.0;
        }
    }
    return ret;

}

auto callCpuSolve(const Foam::scalarField& y0, const Foam::ODESolver& ode, TestParams p)
{
    using namespace Foam;

    const scalar xStart = p.xStart;
    const scalar xEnd = p.xEnd; //1E-5;
    const scalar dxTry = p.dxTry;
    const label li = 0;

    scalarField y = y0;
    scalar dxTry_temp = dxTry;
    ode.solve(xStart, xEnd, y, li, dxTry_temp);

    auto ret = std::vector<gScalar>(y.begin(), y.end());

    //Round small values to zero to avoid -0 == 0 comparisons
    for (auto& e : ret)
    {
        if (std::abs(e) < gpuSmall)
        {
            e = 0.0;
        }
    }

    return ret;

}




static inline void runMechanismTests(TestData::Mechanism mech)
{

    using namespace FoamGpu;

    Foam::MockOFSystem cpu_system(mech);

    auto gpu_thermos = toDeviceVector(makeGpuThermos(mech));
    auto gpu_reactions = toDeviceVector(makeGpuReactions(mech));


    gpuODESystem gpu_system
    (
        cpu_system.nEqns(),
        gLabel(gpu_reactions.size()),
        make_raw_pointer(gpu_thermos.data()),
        make_raw_pointer(gpu_reactions.data())
    );

    const Foam::scalarField y0 = [=](){
        gLabel nEqns = TestData::equationCount(mech);
        Foam::scalarField y0(nEqns);
        assign_test_condition(y0, mech);
        return y0;
    }();

    TestParams params{0.0, 1E-5, 1E-7};

    {
        Foam::dictionary dict;
        dict.add("solver", "Rosenbrock12");

        auto cpu = Foam::ODESolver::New(cpu_system, dict);
        auto gpu = make_gpuODESolver(gpu_system, read_gpuODESolverInputs(dict));
        auto y_gpu = callGpuSolve(y0, gpu, params);
        auto y_cpu = callCpuSolve(y0, cpu, params);

        REQUIRE_THAT
        (
            y_gpu,
            Catch::Matchers::Approx(toStdVector(y_cpu)).epsilon(errorTol)
        );
    }

    {
        Foam::dictionary dict;
        dict.add("solver", "Rosenbrock23");

        auto cpu = Foam::ODESolver::New(cpu_system, dict);
        auto gpu = make_gpuODESolver(gpu_system, read_gpuODESolverInputs(dict));
        auto y_gpu = callGpuSolve(y0, gpu, params);
        auto y_cpu = callCpuSolve(y0, cpu, params);

        REQUIRE_THAT
        (
            y_gpu,
            Catch::Matchers::Approx(toStdVector(y_cpu)).epsilon(errorTol)
        );
    }


    {

        Foam::dictionary dict;
        dict.add("solver", "Rosenbrock34");
        auto cpu = Foam::ODESolver::New(cpu_system, dict);
        auto gpu = make_gpuODESolver(gpu_system, read_gpuODESolverInputs(dict));
        auto y_gpu = callGpuSolve(y0, gpu, params);
        auto y_cpu = callCpuSolve(y0, cpu, params);

        REQUIRE_THAT
        (
            y_gpu,
            Catch::Matchers::Approx(toStdVector(y_cpu)).epsilon(errorTol)
        );
    }

}


TEST_CASE("Test ODE")
{
    runMechanismTests(TestData::GRI);
    runMechanismTests(TestData::H2);
}


