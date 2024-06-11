#include "catch.H"

#include "cpu_reference_results.H"
#include "gpu_reference_results.H"
#include "test_utilities.H"
#include "mechanisms.H"
#include "mdspan.H"

TEST_CASE("make_mdspan"){


    std::vector<int> v = {1,2,3,4,5,6};

    auto s = make_mdspan(v, extents<2>(2,3));

    CHECK(s(0,0) == 1);

    //REQUIRE_THROWS(make_mdspan(v, extents<2>(2,1)));

    CHECK(s.size() == 2*3);

}



TEST_CASE("Test gpuConstants")
{
    auto cpu_result = TestData::constant_results_cpu();
    auto gpu_result = TestData::constant_results_gpu();

    SECTION("Physical")
    {
        CHECK(gpu_result.RR == cpu_result.RR);
        CHECK(gpu_result.Pstd == cpu_result.Pstd);
        CHECK(gpu_result.Tstd == cpu_result.Tstd);
        CHECK(gpu_result.NA == cpu_result.NA);
        CHECK(gpu_result.k == cpu_result.k);

    }

    SECTION("Numeric")
    {
        CHECK(gpu_result.vGreat == cpu_result.vGreat);
        CHECK(gpu_result.vSmall == cpu_result.vSmall);
        CHECK(gpu_result.small == cpu_result.small);
        CHECK(gpu_result.great == cpu_result.great);

    }
}


TEST_CASE("Test perfectGas"){

    using namespace FoamGpu;



    SECTION("thermo properties")
    {
        const gScalar p = 1E5;
        const gScalar T = 3542.324;
        const gScalar molWeight = 0.32;
        const gScalar Y = 0.1;

        auto cpu_result = TestData::perfect_gas_results_cpu(p, T, Y, molWeight);
        auto gpu_result = TestData::perfect_gas_results_gpu(p, T, Y, molWeight);

        CHECK(gpu_result.R == cpu_result.R);
        CHECK(gpu_result.rho == cpu_result.rho);
        CHECK(gpu_result.h == cpu_result.h);
        CHECK(gpu_result.Cp == cpu_result.Cp);
        CHECK(gpu_result.e == cpu_result.e);
        CHECK(gpu_result.Cv == cpu_result.Cv);
        CHECK(gpu_result.sp == cpu_result.sp);
        CHECK(gpu_result.psi == cpu_result.psi);
        CHECK(gpu_result.Z == cpu_result.Z);
        CHECK(gpu_result.CpMCv == cpu_result.CpMCv);
        CHECK(gpu_result.alphav == cpu_result.alphav);

    }


}


static inline void thermoTests(TestData::Mechanism mech)
{
    auto results_cpu = TestData::thermo_results_cpu(mech);
    auto results_gpu = TestData::thermo_results_gpu(mech);

    constexpr double errorTol = 1E-8;




    CHECK_THAT
    (
        results_cpu.W,
        Catch::Matchers::Approx(results_gpu.W).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.Y,
        Catch::Matchers::Approx(results_gpu.Y).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.R,
        Catch::Matchers::Approx(results_gpu.R).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.Cp,
        Catch::Matchers::Approx(results_gpu.Cp).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.ha,
        Catch::Matchers::Approx(results_gpu.ha).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.hs,
        Catch::Matchers::Approx(results_gpu.hs).epsilon(errorTol)
    );


    remove_negative(results_cpu.hf);
    remove_negative(results_gpu.hf);

    CHECK_THAT
    (
        results_cpu.hf,
        Catch::Matchers::Approx(results_gpu.hf).epsilon(errorTol)
    );

    CHECK_THAT
    (
        results_cpu.s,
        Catch::Matchers::Approx(results_gpu.s).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.gStd,
        Catch::Matchers::Approx(results_gpu.gStd).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.dCpdT,
        Catch::Matchers::Approx(results_gpu.dCpdT).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.Cv,
        Catch::Matchers::Approx(results_gpu.Cv).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.es,
        Catch::Matchers::Approx(results_gpu.es).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.ea,
        Catch::Matchers::Approx(results_gpu.ea).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.K,
        Catch::Matchers::Approx(results_gpu.K).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.Kp,
        Catch::Matchers::Approx(results_gpu.Kp).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.Kc,
        Catch::Matchers::Approx(results_gpu.Kc).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.dKcdTbyKc,
        Catch::Matchers::Approx(results_gpu.dKcdTbyKc).epsilon(errorTol)
    );


}

TEST_CASE("Test gpuThermo")
{


    SECTION("GRI")
    {
        thermoTests(TestData::GRI);
    }

    SECTION("H2")
    {
        thermoTests(TestData::H2);
    }


}


static inline void reactionTests(TestData::Mechanism mech)
{

    using namespace FoamGpu;

    constexpr double errorTol = 1E-7;


    auto results_gpu = TestData::reaction_results_gpu(mech);
    auto results_cpu = TestData::reaction_results_cpu(mech);


    CHECK_THAT
    (
        results_cpu.Thigh,
        Catch::Matchers::Approx(results_gpu.Thigh).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.Tlow,
        Catch::Matchers::Approx(results_gpu.Tlow).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.Kc,
        Catch::Matchers::Approx(results_gpu.Kc).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.kf,
        Catch::Matchers::Approx(results_gpu.kf).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.kr,
        Catch::Matchers::Approx(results_gpu.kr).epsilon(errorTol)
    );
    CHECK_THAT
    (
        results_cpu.omega,
        Catch::Matchers::Approx(results_gpu.omega).epsilon(errorTol)
    );



    for (size_t i = 0; i < results_cpu.dNdtByV.size(); ++i)
    {
        REQUIRE_THAT
        (
            results_cpu.dNdtByV[i],
            Catch::Matchers::Approx(results_gpu.dNdtByV[i]).epsilon(errorTol)
        );
    }


    for (size_t i = 0; i < results_cpu.ddNdtByVdcTp.size(); ++i)
    {
        REQUIRE_THAT
        (
            results_cpu.ddNdtByVdcTp[i],
            Catch::Matchers::Approx(results_gpu.ddNdtByVdcTp[i]).epsilon(errorTol)
        );
    }


}

TEST_CASE("Test gpuReaction functions")
{


    SECTION("GRI")
    {
        reactionTests(TestData::GRI);
    }

    SECTION("H2")
    {
        reactionTests(TestData::H2);
    }


}


TEST_CASE("memoryResource"){
    using namespace FoamGpu;

    using MR_t = memoryResource_t;

    SECTION("Constructors")
    {
        REQUIRE_NOTHROW(MR_t());

        REQUIRE_NOTHROW(MR_t(10, 1));
    }

    SECTION("resize"){

        MR_t m(10, 3);
        REQUIRE_NOTHROW(m.resize(12, 1));
        REQUIRE_NOTHROW(m.resize(0, 0));
        REQUIRE_NOTHROW(m.resize(3, 5));
        CHECK(m.nCells() == 3);
        CHECK(m.nEqns() == 7);
        CHECK(m.nSpecie() == 5);
    }

    SECTION("splitToBuffers")
    {

        gLabel nCells = 3;
        gLabel nSpecie = 6;
        gLabel nEqns = nSpecie +2;
        MR_t mr(nCells, nSpecie);

        auto buffers_arr = toDeviceVector(splitToBuffers(mr));

        CHECK(gLabel(buffers_arr.size()) == nCells);



        auto f = [
            nCells = nCells,
            nEqns = nEqns,
            buffers = make_mdspan(buffers_arr, extents<1>{nCells})
        ]()
        {

            for (gLabel i = 0; i < nCells; ++i)
            {
                for (gLabel j = 0; j < nEqns; ++j)
                {
                    buffers[i].pivotIndices()[j] = i;
                    buffers[i].dydx0()[j] = gScalar(i);
                    buffers[i].yTemp()[j] = gScalar(i);
                    buffers[i].dydx()[j] = gScalar(i);
                    buffers[i].dfdx()[j] = gScalar(i);
                    buffers[i].k1()[j] = gScalar(i);
                    buffers[i].k2()[j] = gScalar(i);
                    buffers[i].k3()[j] = gScalar(i);
                    buffers[i].k4()[j] = gScalar(i);
                    buffers[i].err()[j] = gScalar(i);
                    buffers[i].lubuffer()[j] = gScalar(i);
                    buffers[i].c()[j] = gScalar(i);
                    buffers[i].tempField1()[j] = gScalar(i);
                    buffers[i].tempField2()[j] = gScalar(i);
                }
            }
            return buffers[0].pivotIndices()[2];
        };

        CHECK(eval(f) == 0);

    }

}



TEST_CASE("Test ludecompose")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-7;


    for (int i = 3; i < 50; ++i)
    {
        int size = i;


        std::vector<gScalar> vals(size*size);
        fill_random(vals);
        std::vector<gScalar> source(size, 1);


        auto [m_gpu, p_gpu, s_gpu] = TestData::lu_results_gpu(vals, source);
        auto [m_cpu, p_cpu, s_cpu] = TestData::lu_results_cpu(vals, source);

        REQUIRE_THAT
        (
            m_gpu,
            Catch::Matchers::Approx(m_cpu).epsilon(errorTol)
        );
        REQUIRE_THAT
        (
            p_gpu,
            Catch::Matchers::Approx(p_cpu).epsilon(errorTol)
        );
        REQUIRE_THAT
        (
            s_gpu,
            Catch::Matchers::Approx(s_cpu).epsilon(errorTol)
        );

    }
}

TEST_CASE("Test gpuOdeSystem")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-7;


    SECTION("H2"){

        auto cpu_results = TestData::odesystem_results_cpu(TestData::Mechanism::H2);
        auto gpu_results = TestData::odesystem_results_gpu(TestData::Mechanism::H2);


        CHECK_THAT
        (
            cpu_results.derivative,
            Catch::Matchers::Approx(gpu_results.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            cpu_results.jacobian,
            Catch::Matchers::Approx(gpu_results.jacobian).epsilon(errorTol)
        );


    }

    SECTION("GRI"){

        auto cpu_results = TestData::odesystem_results_cpu(TestData::Mechanism::GRI);
        auto gpu_results = TestData::odesystem_results_gpu(TestData::Mechanism::GRI);

        CHECK_THAT
        (
            cpu_results.derivative,
            Catch::Matchers::Approx(gpu_results.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            cpu_results.jacobian,
            Catch::Matchers::Approx(gpu_results.jacobian).epsilon(errorTol)
        );


    }
}

TEST_CASE("Test gpuOdeSolver")
{
    using namespace FoamGpu;

    const gScalar xStart = 0.0;
    const gScalar xEnd = 2E-7;
    const gScalar dxTry = 1E-7;

    constexpr double errorTol = 1E-7;


    SECTION("H2"){

        TestData::Mechanism mech = TestData::Mechanism::H2;

        SECTION("Rosenbrock12")
        {
            auto cpu_results = TestData::ode_results_cpu(mech, "Rosenbrock12", xStart, xEnd, dxTry);
            auto gpu_results = TestData::ode_results_gpu(mech, "Rosenbrock12", xStart, xEnd, dxTry);

            CHECK_THAT
            (
                cpu_results,
                Catch::Matchers::Approx(gpu_results).epsilon(errorTol)
            );

        }

        SECTION("Rosenbrock23")
        {
            auto cpu_results = TestData::ode_results_cpu(mech, "Rosenbrock23", xStart, xEnd, dxTry);
            auto gpu_results = TestData::ode_results_gpu(mech, "Rosenbrock23", xStart, xEnd, dxTry);
            CHECK_THAT
            (
                cpu_results,
                Catch::Matchers::Approx(gpu_results).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            auto cpu_results = TestData::ode_results_cpu(mech, "Rosenbrock34", xStart, xEnd, dxTry);
            auto gpu_results = TestData::ode_results_gpu(mech, "Rosenbrock34", xStart, xEnd, dxTry);
            CHECK_THAT
            (
                cpu_results,
                Catch::Matchers::Approx(gpu_results).epsilon(errorTol)
            );

        }


    }


    SECTION("GRI"){

        TestData::Mechanism mech = TestData::Mechanism::GRI;

        SECTION("Rosenbrock12")
        {
            auto cpu_results = TestData::ode_results_cpu(mech, "Rosenbrock12", xStart, xEnd, dxTry);
            auto gpu_results = TestData::ode_results_gpu(mech, "Rosenbrock12", xStart, xEnd, dxTry);


            CHECK_THAT
            (
                cpu_results,
                Catch::Matchers::Approx(gpu_results).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock23")
        {
            auto cpu_results = TestData::ode_results_cpu(mech, "Rosenbrock23", xStart, xEnd, dxTry);
            auto gpu_results = TestData::ode_results_gpu(mech, "Rosenbrock23", xStart, xEnd, dxTry);


            CHECK_THAT
            (
                cpu_results,
                Catch::Matchers::Approx(gpu_results).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            auto cpu_results = TestData::ode_results_cpu(mech, "Rosenbrock34", xStart, xEnd, dxTry);
            auto gpu_results = TestData::ode_results_gpu(mech, "Rosenbrock34", xStart, xEnd, dxTry);


            CHECK_THAT
            (
                cpu_results,
                Catch::Matchers::Approx(gpu_results).epsilon(errorTol)
            );

        }

    }



}
TEST_CASE("for_each_index"){

    CHECK(TestData::test_for_each_index() == true);

}

TEST_CASE("Test gpuKernelEvaluator"){


    CHECK(TestData::test_evaluator(1) == true);
    CHECK(TestData::test_evaluator(2) == true);
    /*
    CHECK(TestData::test_evaluator(100) == true);

    CHECK(TestData::test_evaluator(250) == true);
    CHECK(TestData::test_evaluator(400) == true);
    CHECK(TestData::test_evaluator(600) == true);
    */

}