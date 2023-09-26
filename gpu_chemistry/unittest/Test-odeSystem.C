
#include "catch.H"

#include "mdspan.H"
#include "test_utilities.H"
#include "mock_of_odesystem.H"
#include "gpuODESystem.H"
#include "create_inputs.H"

TEST_CASE("Test gpuBuffer")
{
    using namespace FoamGpu;

    //No idea why this cant be created directly on device...
    //auto buffer_temp = host_vector<gpuBuffer>(1, gpuBuffer(43));
    auto buffer = toDeviceVector(host_vector<gpuBuffer>(1, gpuBuffer(43)));


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

static inline void runMechanismTests(TestData::Mechanism mech)
{

    using namespace FoamGpu;



    Foam::MockOFSystem cpu(mech);

    auto gpu_thermos_temp = TestData::makeGpuThermos(mech);
    auto gpu_reactions_temp = TestData::makeGpuReactions(mech);


    auto gpu_thermos = toDeviceVector(gpu_thermos_temp);
    auto gpu_reactions = toDeviceVector(gpu_reactions_temp);


    gpuODESystem gpu
    (
        cpu.nEqns(),
        gLabel(gpu_reactions.size()),
        get_raw_pointer(gpu_thermos),
        get_raw_pointer(gpu_reactions)
    );

    const gLabel nSpecie = TestData::speciesCount(mech);
    const gLabel nEqns = TestData::equationCount(mech);

    const gLabel li = 0;

    const Foam::scalarField y0 = [=](){
        gLabel nEqns = TestData::equationCount(mech);
        Foam::scalarField y0(nEqns);
        assign_test_condition(y0, mech);
        return y0;
    }();

    {

        const Foam::scalarField y_cpu = y0;
        const auto y_gpu = toDeviceVector(y0);

        Foam::scalarField dy_cpu(nEqns, 0.31);
        auto dy_gpu = toDeviceVector(dy_cpu);


        cpu.derivatives(0.0, y_cpu, li, dy_cpu);

        auto buffer = toDeviceVector(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));

        auto f =
        [
            =,
            buffer = make_mdspan(buffer, extents<1>{1}),
            y = make_mdspan(y_gpu, extents<1>{nEqns}),
            dy = make_mdspan(dy_gpu, extents<1>{nEqns})
        ]
        ()
        {
            gpu.derivatives(0.0, y, li, dy, buffer[0]);
            return 0;
        };

        eval(f);

        REQUIRE_THAT
        (
            toStdVector(dy_gpu),
            Catch::Matchers::Approx(toStdVector(dy_cpu)).epsilon(errorTol)
        );

    }



    //Jacobian tests
    {
        const gLabel li = 0;

        const gScalar time = 0.1;

        const Foam::scalarField y_cpu = y0;
        const auto y_gpu = toDeviceVector(y0);

        Foam::scalarField dy_cpu(nEqns, 0.31);
        auto dy_gpu = toDeviceVector(dy_cpu);

        Foam::scalarSquareMatrix J_cpu(nEqns, 0.1);
        device_vector<gScalar> J_gpu(J_cpu.size(), 0.2);

        auto buffer = toDeviceVector(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));

        cpu.jacobian(time, y_cpu, li, dy_cpu, J_cpu);


        auto f =
        [
            =,
            buffer = make_mdspan(buffer, extents<1>{1}),
            y = make_mdspan(y_gpu, extents<1>{nEqns}),
            dy = make_mdspan(dy_gpu, extents<1>{nEqns}),
            J = make_mdspan(J_gpu, extents<2>{nEqns, nEqns})
        ]
        ()
        {
            gpu.jacobian(time, y, li, dy, J, buffer[0]);
            return 0;
        };

        eval(f);

        REQUIRE_THAT
        (
            toStdVector(dy_gpu),
            Catch::Matchers::Approx(toStdVector(dy_cpu)).epsilon(errorTol)
        );

        auto Jacobian_cpu = std::vector<gScalar>(J_cpu.v(), J_cpu.v()+J_cpu.size());
        auto Jacobian_gpu = toStdVector(J_gpu);

        REQUIRE_THAT
        (
            toStdVector(Jacobian_gpu),
            Catch::Matchers::Approx(Jacobian_cpu).epsilon(errorTol)
        );

    }
}

TEST_CASE("Test gpuOdeSystem")
{


    SECTION("GRI")
    {
        runMechanismTests(TestData::GRI);
    }

    SECTION("H2")
    {
        runMechanismTests(TestData::H2);
    }



}