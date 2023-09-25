
#include "catch.H"
#include "test_utilities.H"
#include "create_inputs.H"
#include "mock_of_odesystem.H"
#include "gpuODESystem.H"
#include "makeGpuOdeSolver.H"
#include "readGpuOdeInputs.H"
#include "Rosenbrock34.H"
#include "Rosenbrock23.H"
#include "mdspan.H"
#include "ludecompose.H"
#include "mock_of_rosenbrock34.H"



TEST_CASE("Test ludecompose")
{
    using namespace FoamGpu;

    for (int i = 3; i < 50; ++i)
    {
        int size = i;


        std::vector<gScalar> vals(size*size);
        fill_random(vals);

        Foam::scalarSquareMatrix m_cpu(size, 0);
        std::copy(vals.begin(), vals.end(), m_cpu.v());
        Foam::List<Foam::label> pivot_cpu(size, 0);
        gLabel sign = 1;
        //Foam::scalarField v_cpu(size, 0);




        device_vector<gScalar> m_gpu(vals.begin(), vals.end());
        device_vector<gLabel> pivot_gpu(size, 0);
        device_vector<gScalar> v_gpu(size, 0);


        auto m_span = make_mdspan(m_gpu, extents<2>{size, size});
        auto p_span = make_mdspan(pivot_gpu, extents<1>{size});
        auto v_span = make_mdspan(v_gpu, extents<1>{size});

        Foam::LUDecompose(m_cpu, pivot_cpu, sign);

        eval
        (
            [=](){FoamGpu::LUDecompose(m_span, p_span, v_span); return 0;}
        );


        auto r_cpu1 = std::vector(m_cpu.v(), m_cpu.v() + size*size);
        auto r_gpu1 = to_std_vec(m_gpu);

        REQUIRE_THAT
        (
            r_gpu1,
            Catch::Matchers::Approx(r_cpu1).epsilon(errorTol)
        );
        REQUIRE_THAT
        (
            to_std_vec(pivot_gpu),
            Catch::Matchers::Approx(to_std_vec(pivot_cpu)).epsilon(errorTol)
        );




        device_vector<gScalar> source_gpu(size, 0);
        Foam::scalarField source_cpu(size, 0);

        auto s_span = make_mdspan(source_gpu, extents<1>{size});

        Foam::LUBacksubstitute(m_cpu, pivot_cpu, source_cpu);


        eval
        (
            [=](){FoamGpu::LUBacksubstitute(m_span, p_span, s_span); return 0;}
        );


        REQUIRE_THAT
        (
            to_std_vec(source_gpu),
            Catch::Matchers::Approx(to_std_vec(source_cpu)).epsilon(errorTol)
        );


        CHECK(to_std_vec(source_gpu) == to_std_vec(source_cpu));

    }
}

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
    const gLabel li = 0;

    auto y = to_device_vec(y0);
    device_vector<gScalar> J (nEqns*nEqns);
    auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));

    auto f = [
        ode = ode,
        xStart = xStart,
        xEnd = xEnd,
        y = make_mdspan(y, extents<1>{nEqns}),
        li = li,
        dxTry = dxTry,
        J = make_mdspan(J, extents<2>{nEqns, nEqns}),
        buffer = make_mdspan(buffer, extents<1>{1})
    ]()
    {
        gScalar dxTry_temp = dxTry;
        ode.solve(xStart, xEnd, y, li, dxTry_temp, J, buffer[0]);
        return dxTry_temp;
    };

    auto unused = eval(f);
    (void) unused;

    auto ret = to_std_vec(y);

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

    auto ret = to_std_vec(y);

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
    auto gpu_thermos_temp = makeGpuThermos(mech);
    auto gpu_reactions_temp = makeGpuReactions(mech);
    auto gpu_thermos = device_vector<gpuThermo>(gpu_thermos_temp.begin(), gpu_thermos_temp.end());
    auto gpu_reactions = device_vector<gpuReaction>(gpu_reactions_temp.begin(), gpu_reactions_temp.end());


    gpuODESystem gpu_system
    (
        cpu_system.nEqns(),
        gLabel(gpu_reactions.size()),
        get_raw_pointer(gpu_thermos),
        get_raw_pointer(gpu_reactions)
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
        dict.add("solver", "Rosenbrock23");

        auto cpu = Foam::ODESolver::New(cpu_system, dict);
        auto gpu = make_gpuODESolver(gpu_system, read_gpuODESolverInputs(dict));
        auto y_gpu = callGpuSolve(y0, gpu, params);
        auto y_cpu = callCpuSolve(y0, cpu, params);

        REQUIRE_THAT
        (
            y_gpu,
            Catch::Matchers::Approx(to_std_vec(y_cpu)).epsilon(errorTol)
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
            Catch::Matchers::Approx(to_std_vec(y_cpu)).epsilon(errorTol)
        );
    }

}


TEST_CASE("Temp")
{
    runMechanismTests(TestData::GRI);
    runMechanismTests(TestData::H2);
}


