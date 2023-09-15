
#include "catch.H"

#include "mdspan.H"
#include "test_utilities.H"
#include "mock_of_odesystem.H"
#include "gpuODESystem.H"

TEST_CASE("Test gpuBuffer")
{
    using namespace FoamGpu;

    //No idea why this cant be created directly on device...
    //auto buffer_temp = host_vector<gpuBuffer>(1, gpuBuffer(43));
    auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(43)));


    auto f =
    [
        bspan = make_mdspan(buffer, extents<1>{1})
    ]()
    {
        auto c = bspan[0].c();
        return c[32];
    };

    //This is here just to ensure that compiler doesnt optimize away
    //the return value
    CHECK(eval(f) != Approx(gScalar(3)).epsilon(errorTol));

}


TEST_CASE("Test gpuOdeSystem")
{
    using namespace FoamGpu;

    Foam::MockOFSystem cpu;

    auto gpu_thermos_temp = makeGpuThermos();
    auto gpu_reactions_temp = makeGpuReactions();


    auto gpu_thermos = device_vector<gpuThermo>(gpu_thermos_temp.begin(), gpu_thermos_temp.end());
    auto gpu_reactions = device_vector<gpuReaction>(gpu_reactions_temp.begin(), gpu_reactions_temp.end());


    gpuODESystem gpu
    (
        cpu.nEqns(),
        gLabel(gpu_reactions.size()),
        get_raw_pointer(gpu_thermos),
        get_raw_pointer(gpu_reactions)
    );

    gLabel nSpecie = make_species_table().size();
    gLabel nEqns = cpu.nEqns();





    SECTION("derivatives")
    {

        const gLabel li = 0;


        for (gLabel i = 0; i < 10; ++i)
        {

            Foam::scalarField YTp_cpu(nEqns);
            fill_random(YTp_cpu);

            Foam::scalarField dYTpdt_cpu(nEqns, 0.31);

            auto YTp_gpu = to_device_vec(YTp_cpu);
            //auto dYTpdt_gpu = to_device_vec(dYTpdt_cpu);
            auto dYTpdt_gpu = device_vector<gScalar>(dYTpdt_cpu.size(), 0.21);

            cpu.derivatives(0.0, YTp_cpu, li, dYTpdt_cpu);

            auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));
            //device_vector<gpuBuffer> buffer(1, gpuBuffer(nSpecie));

            auto f =
            [
                =,
                buffer = make_mdspan(buffer, extents<1>{1}),
                YTp = make_mdspan(YTp_gpu, extents<1>{nEqns}),
                dYTpdt = make_mdspan(dYTpdt_gpu, extents<1>{nEqns})
            ]
            ()
            {
                gpu.derivatives(0.0, YTp, li, dYTpdt, buffer[0]);
                return 0;
            };

            eval(f);

            auto derivative_gpu = to_std_vec(dYTpdt_gpu);
            auto derivative_cpu = to_std_vec(dYTpdt_cpu);

            for(size_t i = 0; i < derivative_gpu.size(); ++i)
            {
                REQUIRE
                (
                    derivative_gpu[i] == Approx(derivative_cpu[i]).epsilon(errorTol)
                );
            }


        }

    }

    SECTION("jacobian random values")
    {
        const gLabel li = 0;

        const gScalar time = 0.1;

        Foam::scalarSquareMatrix J_cpu(nEqns, 0.1);
        device_vector<gScalar> J_gpu(J_cpu.size(), 0.2);

        for (gLabel i = 0; i < 10; ++i)
        {
            const gScalar p = 0.32E5;
            const gScalar T = 500;
            Foam::scalarField YTp_cpu(nEqns, 0.1);

            fill_random(YTp_cpu);
            YTp_cpu[nSpecie] = T + i*30;
            YTp_cpu[nSpecie+1] = p + i*1E4;
            Foam::scalarField dYTpdt_cpu(nEqns, 0.1);

            //This needs to be sme for both
            device_vector<gScalar> YTp_gpu(YTp_cpu.begin(), YTp_cpu.end());

            //device_vector<gScalar> dYTpdt_gpu(dYTpdt_cpu.begin(), dYTpdt_cpu.end());
            device_vector<gScalar> dYTpdt_gpu(dYTpdt_cpu.size(), 43.0);

            auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));


            cpu.derivatives(time, YTp_cpu, li, dYTpdt_cpu);
            cpu.jacobian(time, YTp_cpu, li, dYTpdt_cpu, J_cpu);


            auto f =
            [
                =,
                buffer = make_mdspan(buffer, extents<1>{1}),
                YTp = make_mdspan(YTp_gpu, extents<1>{nEqns}),
                dYTpdt = make_mdspan(dYTpdt_gpu, extents<1>{nEqns}),
                J = make_mdspan(J_gpu, extents<2>{nEqns, nEqns})
            ]
            ()
            {
                gpu.derivatives(time, YTp, li, dYTpdt, buffer[0]);
                gpu.jacobian(time, YTp, li, dYTpdt, J, buffer[0]);
                return 0;
            };

            eval(f);

            auto Jderivative_cpu = to_std_vec(dYTpdt_cpu);
            auto Jderivative_gpu = to_std_vec(dYTpdt_gpu);

            for (size_t i = 0; i < Jderivative_cpu.size(); ++i)
            {
                REQUIRE
                (
                    Jderivative_gpu[i] == Approx(Jderivative_cpu[i]).epsilon(errorTol)
                );
            }

            auto Jacobian_cpu = std::vector<gScalar>(J_cpu.v(), J_cpu.v()+J_cpu.size());
            auto Jacobian_gpu = to_std_vec(J_gpu);

            for (size_t i = 0; i < Jacobian_cpu.size(); ++i)
            {
                if (Jacobian_cpu[i] > 1E-8)
                {
                    CHECK(Jacobian_gpu[i] == Approx(Jacobian_cpu[i]).epsilon(errorTol));
                }
                //CHECK(Jacobian_gpu[i] == Approx(Jacobian_cpu[i]).epsilon(errorTol));
            }


        }

    }


    SECTION("jacobian gri values")
    {
        const gLabel li = 0;

        const gScalar time = 0.1;

        Foam::scalarSquareMatrix J_cpu(nEqns, 0.1);
        device_vector<gScalar> J_gpu(J_cpu.size(), 0.2);

        Foam::scalarField YTp_cpu(nEqns, 0.1);

        assign_test_condition(YTp_cpu);
        Foam::scalarField dYTpdt_cpu(nEqns, 0.1);

        //This needs to be sme for both
        device_vector<gScalar> YTp_gpu(YTp_cpu.begin(), YTp_cpu.end());

        //device_vector<gScalar> dYTpdt_gpu(dYTpdt_cpu.begin(), dYTpdt_cpu.end());
        device_vector<gScalar> dYTpdt_gpu(dYTpdt_cpu.size(), 43.0);

        auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));


        cpu.derivatives(time, YTp_cpu, li, dYTpdt_cpu);
        cpu.jacobian(time, YTp_cpu, li, dYTpdt_cpu, J_cpu);


        auto f =
        [
            =,
            buffer = make_mdspan(buffer, extents<1>{1}),
            YTp = make_mdspan(YTp_gpu, extents<1>{nEqns}),
            dYTpdt = make_mdspan(dYTpdt_gpu, extents<1>{nEqns}),
            J = make_mdspan(J_gpu, extents<2>{nEqns, nEqns})
        ]
        ()
        {
            gpu.derivatives(time, YTp, li, dYTpdt, buffer[0]);
            gpu.jacobian(time, YTp, li, dYTpdt, J, buffer[0]);
            return 0;
        };

        eval(f);

        auto Jderivative_cpu = to_std_vec(dYTpdt_cpu);
        auto Jderivative_gpu = to_std_vec(dYTpdt_gpu);

        for (size_t i = 0; i < Jderivative_cpu.size(); ++i)
        {
            REQUIRE
            (
                Jderivative_gpu[i] == Approx(Jderivative_cpu[i]).epsilon(errorTol)
            );
        }

        auto Jacobian_cpu = std::vector<gScalar>(J_cpu.v(), J_cpu.v()+J_cpu.size());
        auto Jacobian_gpu = to_std_vec(J_gpu);

        for (size_t i = 0; i < Jacobian_cpu.size(); ++i)
        {
            if (Jacobian_cpu[i] > 1E-8)
            {
                REQUIRE(Jacobian_gpu[i] == Approx(Jacobian_cpu[i]).epsilon(errorTol));
            }
            //CHECK(Jacobian_gpu[i] == Approx(Jacobian_cpu[i]).epsilon(errorTol));
        }




    }




}