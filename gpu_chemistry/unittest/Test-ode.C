
#include "catch.H"
#include "test_utilities.H"
#include "mock_of_odesystem.H"
#include "gpuODESystem.H"
#include "gpuRosenbrock34.H"
#include "Rosenbrock34.H"
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



TEST_CASE("Test gpuRosenbrock34")
{

    using namespace FoamGpu;

    Foam::MockOFSystem cpu_system;

    auto gpu_thermos_temp = make_gpu_thermos();
    auto gpu_reactions_temp = make_gpu_reactions();


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


    Foam::dictionary nulldict;
    Foam::Rosenbrock34 cpu(cpu_system, nulldict);
    gpuRosenbrock34<gpuODESystem> gpu = make_Rosenbrock34(gpu_system, nulldict);



    SECTION("solve(x0, y0, li, dydx0, dx, y) random values")
    {
        const gScalar T = 500.0;
        const gScalar p = 1.0E5;

        const gScalar x0 = 0.0;
        const gLabel li = 0;
        const gScalar dx = 1E-7;
        Foam::scalarField y0_cpu(nEqns, 0.0);
        Foam::scalarField dydx0_cpu(nEqns, 0.0);
        Foam::scalarField y_cpu(nEqns, 0);

        fill_random(y0_cpu, 0., 0.1);
        y0_cpu[nSpecie] = T;
        y0_cpu[nSpecie+1] = p;


        auto y0_gpu = to_device_vec(y0_cpu);
        auto dydx0_gpu = to_device_vec(dydx0_cpu);
        auto y_gpu = to_device_vec(y_cpu);

        device_vector<gScalar> J(nEqns*nEqns);


        auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));
        auto f = [
            gpu_system = gpu_system,
            gpuOde = gpu,
            x0 = x0,
            y0 = make_mdspan(y0_gpu, extents<1>{nEqns}),
            li = li,
            dydx0 = make_mdspan(dydx0_gpu, extents<1>{nEqns}),
            dx = dx,
            y = make_mdspan(y_gpu, extents<1>{nEqns}),
            J = make_mdspan(J, extents<2>{nEqns, nEqns}),
            buffer = make_mdspan(buffer, extents<1>{1})
        ]()
        {
            gpu_system.derivatives(0.0, y0, li, dydx0, buffer[0]);
            return gpuOde.solve(x0, y0, li, dydx0, dx, y, J, buffer[0]);
        };

        cpu_system.derivatives(0.0, y0_cpu, li, dydx0_cpu);
        gScalar err_cpu = cpu.solve(x0, y0_cpu, li, dydx0_cpu, dx, y_cpu);
        gScalar err_gpu = eval(f);


        CHECK
        (
            err_gpu == Approx(err_cpu).epsilon(errorTol)
        );

        auto y_new_cpu = to_std_vec(y_cpu);
        auto y_new_gpu = to_std_vec(y_gpu);


        for (size_t i = 0; i < y_new_cpu.size(); ++i)
        {
            CHECK
            (
                y_new_gpu[i] == Approx(y_new_cpu[i]).epsilon(errorTol)
            );
        }

    }

    SECTION("solve(x0, y0, li, dydx0, dx, y) gri values")
    {

        const gScalar x0 = 0.0;
        const gLabel li = 0;
        const gScalar dx = 1E-7;
        Foam::scalarField y0_cpu(nEqns, 0.0);
        Foam::scalarField dydx0_cpu(nEqns, 0.0);
        Foam::scalarField y_cpu(nEqns, 0);

        assign_test_condition(y0_cpu);


        auto y0_gpu = to_device_vec(y0_cpu);
        auto dydx0_gpu = to_device_vec(dydx0_cpu);
        auto y_gpu = to_device_vec(y_cpu);

        device_vector<gScalar> J(nEqns*nEqns);


        auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));
        auto f = [
            gpu_system = gpu_system,
            gpuOde = gpu,
            x0 = x0,
            y0 = make_mdspan(y0_gpu, extents<1>{nEqns}),
            li = li,
            dydx0 = make_mdspan(dydx0_gpu, extents<1>{nEqns}),
            dx = dx,
            y = make_mdspan(y_gpu, extents<1>{nEqns}),
            J = make_mdspan(J, extents<2>{nEqns, nEqns}),
            buffer = make_mdspan(buffer, extents<1>{1})
        ]()
        {
            gpu_system.derivatives(0.0, y0, li, dydx0, buffer[0]);
            return gpuOde.solve(x0, y0, li, dydx0, dx, y, J, buffer[0]);
        };

        cpu_system.derivatives(0.0, y0_cpu, li, dydx0_cpu);
        gScalar err_cpu = cpu.solve(x0, y0_cpu, li, dydx0_cpu, dx, y_cpu);
        gScalar err_gpu = eval(f);


        CHECK
        (
            err_gpu == Approx(err_cpu).epsilon(errorTol)
        );

        auto y_new_cpu = to_std_vec(y_cpu);
        auto y_new_gpu = to_std_vec(y_gpu);


        for (size_t i = 0; i < y_new_cpu.size(); ++i)
        {
            if (y_new_cpu[i] > gpuSmall)
            {
                CHECK
                (
                    y_new_gpu[i] == Approx(y_new_cpu[i]).epsilon(errorTol)
                );
            }
        }

    }



    /*

    SECTION("solve(x, y, li, dxTry) random values")
    {

        const gScalar x = 0.;
        const gLabel li = 0;
        const gScalar dxTry = 1E-7;

        const gScalar T = 500.0;
        const gScalar p = 1.0E5;
        Foam::scalarField y_cpu(nEqns, 0.0);
        fill_random(y_cpu, 0.0, 0.1);
        y_cpu[nSpecie] = T;
        y_cpu[nSpecie+1] = p;

        auto y_gpu = to_device_vec(y_cpu);

        device_vector<gScalar> J(nEqns*nEqns);

        auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));
        auto f = [
            gpu = gpu,
            x = x,
            y = make_mdspan(y_gpu, extents<1>{nEqns}),
            li = li,
            dxTry = dxTry,
            J = make_mdspan(J, extents<2>{nEqns, nEqns}),
            buffer = make_mdspan(buffer, extents<1>{1})
        ]()
        {
            gScalar dxTry_temp = dxTry;
            gScalar x_temp = x;
            gpu.solve(x_temp, y, li, dxTry_temp, J, buffer[0]);
            return x_temp + dxTry_temp;
        };


        auto f2 = [&]()
        {

            gScalar dxTry_temp = dxTry;
            gScalar x_temp = x;
            cpu.solve(x_temp, y_cpu, li, dxTry_temp);
            return x_temp + dxTry_temp;
        };



        CHECK
        (
            eval(f) == Approx(f2()).epsilon(errorTol)
        );

        auto y_new_cpu = to_std_vec(y_cpu);
        auto y_new_gpu = to_std_vec(y_gpu);

        for (size_t i = 0; i < y_new_cpu.size(); ++i)
        {
            CHECK
            (
                y_new_gpu[i] == Approx(y_new_cpu[i]).epsilon(errorTol)
            );
        }
    }

    SECTION("solve(xStart, xEnd, y, li, dxTry) gri values")
    {
        const gScalar xStart = 0.;
        const gScalar xEnd = 1E-5; //1E-5;
        const gLabel li = 0;
        const gScalar dxTry = 1E-7;

        Foam::scalarField y_cpu(nEqns, 0.0);
        assign_test_condition(y_cpu);

        auto y_gpu = to_device_vec(y_cpu);

        device_vector<gScalar> J(nEqns*nEqns);

        auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));
        auto f = [
            gpu = gpu,
            xStart = xStart,
            xEnd = xEnd,
            y = make_mdspan(y_gpu, extents<1>{nEqns}),
            li = li,
            dxTry = dxTry,
            J = make_mdspan(J, extents<2>{nEqns, nEqns}),
            buffer = make_mdspan(buffer, extents<1>{1})
        ]()
        {
            gScalar dxTry_temp = dxTry;
            gpu.solve(xStart, xEnd, y, li, dxTry_temp, J, buffer[0]);
            return dxTry_temp;
        };

        auto f2 =
        [
            &cpu=cpu,
            xStart = xStart,
            xEnd = xEnd,
            &y = y_cpu,
            li = li,
            dxTry = dxTry
        ]()
        {

            gScalar dxTry_temp = dxTry;
            cpu.solve(xStart, xEnd, y, li, dxTry_temp);
            return dxTry_temp;
        };

        CHECK
        (
            eval(f) == Approx(f2()).epsilon(errorTol)
        );

        auto y_new_cpu = to_std_vec(y_cpu);
        auto y_new_gpu = to_std_vec(y_gpu);

        for (size_t i = 0; i < y_new_cpu.size(); ++i)
        {
            if (y_new_cpu[i] > gpuSmall)
            {
                CHECK
                (
                    y_new_gpu[i] == Approx(y_new_cpu[i]).epsilon(errorTol)
                );
            }
        }

    }

    */

    SECTION("solve(xStart, xEnd, y, li, dxTry) random values")
    {
        const gScalar xStart = 0.;
        const gScalar xEnd = 2E-7; //1E-5;
        const gLabel li = 0;
        const gScalar dxTry = 1E-7;

        const gScalar T = 500.0;
        const gScalar p = 1.0E5;
        Foam::scalarField y_cpu(nEqns, 0.0);
        fill_random(y_cpu, 0.0, 0.1);
        y_cpu[nSpecie] = T;
        y_cpu[nSpecie+1] = p;

        auto y_gpu = to_device_vec(y_cpu);

        device_vector<gScalar> J(nEqns*nEqns);

        auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));
        auto f = [
            gpu = gpu,
            xStart = xStart,
            xEnd = xEnd,
            y = make_mdspan(y_gpu, extents<1>{nEqns}),
            li = li,
            dxTry = dxTry,
            J = make_mdspan(J, extents<2>{nEqns, nEqns}),
            buffer = make_mdspan(buffer, extents<1>{1})
        ]()
        {
            gScalar dxTry_temp = dxTry;
            gpu.solve(xStart, xEnd, y, li, dxTry_temp, J, buffer[0]);
            return dxTry_temp;
        };

        auto f2 =
        [
            &cpu=cpu,
            xStart = xStart,
            xEnd = xEnd,
            &y = y_cpu,
            li = li,
            dxTry = dxTry
        ]()
        {

            gScalar dxTry_temp = dxTry;
            cpu.solve(xStart, xEnd, y, li, dxTry_temp);
            return dxTry_temp;
        };


        CHECK
        (
            eval(f) == Approx(f2()).epsilon(errorTol)
        );

        auto y_new_cpu = to_std_vec(y_cpu);
        auto y_new_gpu = to_std_vec(y_gpu);

        for (size_t i = 0; i < y_new_cpu.size(); ++i)
        {
            CHECK
            (
                y_new_gpu[i] == Approx(y_new_cpu[i]).epsilon(errorTol)
            );
        }

    }



    SECTION("solve(xStart, xEnd, y, li, dxTry) gri values")
    {
        const gScalar xStart = 0.;
        const gScalar xEnd = 1E-5; //1E-5;
        const gLabel li = 0;
        const gScalar dxTry = 1E-7;

        Foam::scalarField y_cpu(nEqns, 0.0);

        assign_test_condition(y_cpu);

        auto y_gpu = to_device_vec(y_cpu);

        device_vector<gScalar> J(nEqns*nEqns);

        auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));
        auto f = [
            gpu = gpu,
            xStart = xStart,
            xEnd = xEnd,
            y = make_mdspan(y_gpu, extents<1>{nEqns}),
            li = li,
            dxTry = dxTry,
            J = make_mdspan(J, extents<2>{nEqns, nEqns}),
            buffer = make_mdspan(buffer, extents<1>{1})
        ]()
        {
            gScalar dxTry_temp = dxTry;
            gpu.solve(xStart, xEnd, y, li, dxTry_temp, J, buffer[0]);
            return dxTry_temp;
        };

        auto f2 =
        [
            &cpu=cpu,
            xStart = xStart,
            xEnd = xEnd,
            &y = y_cpu,
            li = li,
            dxTry = dxTry
        ]()
        {

            gScalar dxTry_temp = dxTry;
            cpu.solve(xStart, xEnd, y, li, dxTry_temp);
            return dxTry_temp;
        };


        CHECK
        (
            eval(f) == Approx(f2()).epsilon(errorTol)
        );

        auto y_new_cpu = to_std_vec(y_cpu);
        auto y_new_gpu = to_std_vec(y_gpu);

        for (size_t i = 0; i < y_new_cpu.size(); ++i)
        {
            if (y_new_cpu[i] > gpuSmall)
            {
                CHECK
                (
                    y_new_gpu[i] == Approx(y_new_cpu[i]).epsilon(errorTol)
                );
            }
        }

    }





}

