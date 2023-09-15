
#include "catch.H"
#include "test_utilities.H"
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
    return to_std_vec(y);

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


    return to_std_vec(y);

}

TEST_CASE("Test ODE.solve")
{

    using namespace FoamGpu;

    Foam::MockOFSystem cpu_system;
    auto gpu_thermos_temp = makeGpuThermos();
    auto gpu_reactions_temp = makeGpuReactions();
    auto gpu_thermos = device_vector<gpuThermo>(gpu_thermos_temp.begin(), gpu_thermos_temp.end());
    auto gpu_reactions = device_vector<gpuReaction>(gpu_reactions_temp.begin(), gpu_reactions_temp.end());


    gpuODESystem gpu_system
    (
        cpu_system.nEqns(),
        gLabel(gpu_reactions.size()),
        get_raw_pointer(gpu_thermos),
        get_raw_pointer(gpu_reactions)
    );


    gLabel nSpecie = make_species_table().size();
    gLabel nEqns = cpu_system.nEqns();

    const Foam::scalarField y0_random = [=](){
        const gScalar T = 500.0;
        const gScalar p = 1.0E5;
        Foam::scalarField y0(nEqns, 0.0);
        fill_random(y0, 0.0, 0.1);
        y0[nSpecie] = T;
        y0[nSpecie+1] = p;
        return y0;
    }();

    const Foam::scalarField y0_gri = [=](){
        Foam::scalarField y0(nEqns, 0.0);
        assign_test_condition(y0);
        return y0;
    }();

    SECTION("Rosenbrock23")
    {

        Foam::dictionary dict;
        dict.add("solver", "Rosenbrock23");

        auto cpu = Foam::ODESolver::New(cpu_system, dict); 
        auto gpu = make_gpuODESolver(gpu_system, read_gpuODESolverInputs(dict));

        SECTION("Random values")
        {
            TestParams params{0.0, 2E-7, 1E-7};


            auto y_gpu = callGpuSolve(y0_random, gpu, params);
            auto y_cpu = callCpuSolve(y0_random, cpu, params);

            for (size_t i = 0; i < y_cpu.size(); ++i)
            {
                CHECK
                (
                    y_gpu[i] == Approx(y_cpu[i]).epsilon(errorTol)
                );
            }

        }

        SECTION("Gri values")
        {
            TestParams params{0.0, 1E-5, 1E-7};
            Foam::scalarField y0(nEqns, 0.0);   
            assign_test_condition(y0);
            auto y_gpu = callGpuSolve(y0_gri, gpu, params);
            auto y_cpu = callCpuSolve(y0_gri, cpu, params);

            for (size_t i = 0; i < y_cpu.size(); ++i)
            {
                //0 == 0 comparisons fail also with gcc
                if (y_gpu[i] > gpuSmall)
                {
                    CHECK
                    (
                        y_gpu[i] == Approx(y_cpu[i]).epsilon(errorTol)
                    );
                }

            }

        }

    }

    SECTION("Rosenbrock34")
    {

        Foam::dictionary dict;
        dict.add("solver", "Rosenbrock34");

        auto cpu = Foam::ODESolver::New(cpu_system, dict); 
        auto gpu = make_gpuODESolver(gpu_system, read_gpuODESolverInputs(dict));

        SECTION("Random values")
        {
            TestParams params{0.0, 2E-7, 1E-7};


            auto y_gpu = callGpuSolve(y0_random, gpu, params);
            auto y_cpu = callCpuSolve(y0_random, cpu, params);

            for (size_t i = 0; i < y_cpu.size(); ++i)
            {
                CHECK
                (
                    y_gpu[i] == Approx(y_cpu[i]).epsilon(errorTol)
                );
            }

        }

        SECTION("Gri values")
        {
            TestParams params{0.0, 1E-5, 1E-7};
            Foam::scalarField y0(nEqns, 0.0);   
            assign_test_condition(y0);
            auto y_gpu = callGpuSolve(y0_gri, gpu, params);
            auto y_cpu = callCpuSolve(y0_gri, cpu, params);

            for (size_t i = 0; i < y_cpu.size(); ++i)
            {
                //0 == 0 comparisons fail also with gcc
                if (y_gpu[i] > gpuSmall)
                {
                    CHECK
                    (
                        y_gpu[i] == Approx(y_cpu[i]).epsilon(errorTol)
                    );
                }

            }

        }

    }

}




