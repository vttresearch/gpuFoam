
#include "catch.H"

#include "mdspan.H"
#include "test_utilities.H"
#include "mock_of_odesystem.H"
#include "gpuODESystem.H"
#include "create_foam_inputs.H"
#include "create_gpu_inputs.H"



static inline void runMechanismTests(TestData::Mechanism mech)
{

    using namespace FoamGpu;



    Foam::MockOFSystem cpu(mech);

    auto gpu_thermos = toDeviceVector(makeGpuThermos(mech));
    auto gpu_reactions = toDeviceVector(makeGpuReactions(mech));

    gpuODESystem gpu
    (
        cpu.nEqns(),
        gLabel(gpu_reactions.size()),
        make_raw_pointer(gpu_thermos.data()),
        make_raw_pointer(gpu_reactions.data())
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

        memoryResource_t memory(1, nSpecie);
        auto buffers = toDeviceVector(splitToBuffers(memory));

        auto f =
        [
            =,
            buffers = make_mdspan(buffers, extents<1>{1}),
            y = make_mdspan(y_gpu, extents<1>{nEqns}),
            dy = make_mdspan(dy_gpu, extents<1>{nEqns})
        ]
        ()
        {
            gpu.derivatives(y, dy, buffers[0]);
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

        memoryResource_t memory(1, nSpecie);
        auto buffers = toDeviceVector(splitToBuffers(memory));

        cpu.jacobian(time, y_cpu, li, dy_cpu, J_cpu);


        auto f =
        [
            =,
            buffers = make_mdspan(buffers, extents<1>{1}),
            y = make_mdspan(y_gpu, extents<1>{nEqns}),
            dy = make_mdspan(dy_gpu, extents<1>{nEqns}),
            J = make_mdspan(J_gpu, extents<2>{nEqns, nEqns})
        ]
        ()
        {
            gpu.jacobian(y, dy, J, buffers[0]);
            return 0;
        };

        eval(f);

        /*
        auto Jtemp = make_mdspan(J_gpu, extents<2>{nEqns, nEqns});
        for (gLabel j = 0; j < nEqns; ++j)
        {
            for (gLabel i = 0; i < nEqns; ++i)
            {
                if (std::abs(J_cpu(j, i) - Jtemp(j, i)) > gpuSmall  )
                {
                    std::cout << J_cpu(j, i) << " " << Jtemp(j, i) << " " << j << " " << i << std::endl;
                }
            }
        }
        */

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