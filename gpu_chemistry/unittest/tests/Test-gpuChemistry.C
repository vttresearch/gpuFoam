#include "catch.H"

#include "openfoam_reference_kernels.H"
#include "gpu_test_kernels.H"
#include "cpu_test_kernels.H"
#include "test_utilities.H"
#include "mechanisms.H"
#include "mdspan.H"


TEST_CASE("make_mdspan", "[CPU]"){

    std::vector<int> v = {1,2,3,4,5,6};
    auto s = make_mdspan(v, extents<2>(2,3));
    CHECK(s(0,0) == 1);

    CHECK(s.size() == 2*3);

}



TEST_CASE("Test gpuConstants (on CPU)", "[CPU]")
{
    auto reference = OFReferenceKernels::constants();
    auto test_result = CpuTestKernels::constants();

    SECTION("Physical")
    {
        CHECK(test_result.RR == reference.RR);
        CHECK(test_result.Pstd == reference.Pstd);
        CHECK(test_result.Tstd == reference.Tstd);
        CHECK(test_result.NA == reference.NA);
        CHECK(test_result.k == reference.k);

    }

    SECTION("Numeric")
    {
        CHECK(test_result.vGreat == reference.vGreat);
        CHECK(test_result.vSmall == reference.vSmall);
        CHECK(test_result.small == reference.small);
        CHECK(test_result.great == reference.great);

    }
}

TEST_CASE("Test gpuConstants (on GPU)", "[GPU]")
{
    auto reference = OFReferenceKernels::constants();
    auto test_result = GpuTestKernels::constants();

    SECTION("Physical")
    {
        CHECK(test_result.RR == reference.RR);
        CHECK(test_result.Pstd == reference.Pstd);
        CHECK(test_result.Tstd == reference.Tstd);
        CHECK(test_result.NA == reference.NA);
        CHECK(test_result.k == reference.k);

    }

    SECTION("Numeric")
    {
        CHECK(test_result.vGreat == reference.vGreat);
        CHECK(test_result.vSmall == reference.vSmall);
        CHECK(test_result.small == reference.small);
        CHECK(test_result.great == reference.great);

    }
}



TEST_CASE("Test perfectGas (on CPU)", "[CPU]"){

    using namespace FoamGpu;

    SECTION("thermo properties")
    {
        const gScalar p = 1E5;
        const gScalar T = 3542.324;
        const gScalar molWeight = 0.32;
        const gScalar Y = 0.1;

        auto reference = OFReferenceKernels::perfect_gas(p, T, Y, molWeight);
        auto test_result = CpuTestKernels::perfect_gas(p, T, Y, molWeight);

        CHECK(test_result.R == reference.R);
        CHECK(test_result.rho == reference.rho);
        CHECK(test_result.h == reference.h);
        CHECK(test_result.Cp == reference.Cp);
        CHECK(test_result.e == reference.e);
        CHECK(test_result.Cv == reference.Cv);
        CHECK(test_result.sp == reference.sp);
        CHECK(test_result.psi == reference.psi);
        CHECK(test_result.Z == reference.Z);
        CHECK(test_result.CpMCv == reference.CpMCv);
        CHECK(test_result.alphav == reference.alphav);

    }

}

TEST_CASE("Test perfectGas (on GPU)", "[GPU]"){

    using namespace FoamGpu;

    SECTION("thermo properties")
    {
        const gScalar p = 1E5;
        const gScalar T = 3542.324;
        const gScalar molWeight = 0.32;
        const gScalar Y = 0.1;

        auto reference = OFReferenceKernels::perfect_gas(p, T, Y, molWeight);
        auto test_result = GpuTestKernels::perfect_gas(p, T, Y, molWeight);

        CHECK(test_result.R == reference.R);
        CHECK(test_result.rho == reference.rho);
        CHECK(test_result.h == reference.h);
        CHECK(test_result.Cp == reference.Cp);
        CHECK(test_result.e == reference.e);
        CHECK(test_result.Cv == reference.Cv);
        CHECK(test_result.sp == reference.sp);
        CHECK(test_result.psi == reference.psi);
        CHECK(test_result.Z == reference.Z);
        CHECK(test_result.CpMCv == reference.CpMCv);
        CHECK(test_result.alphav == reference.alphav);

    }

}




static inline void thermoTests(TestData::thermoResults& test_result, TestData::thermoResults& reference, double errorTol)
{

    CHECK_THAT
    (
        reference.W,
        Catch::Matchers::Approx(test_result.W).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Y,
        Catch::Matchers::Approx(test_result.Y).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.R,
        Catch::Matchers::Approx(test_result.R).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Cp,
        Catch::Matchers::Approx(test_result.Cp).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.ha,
        Catch::Matchers::Approx(test_result.ha).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.hs,
        Catch::Matchers::Approx(test_result.hs).epsilon(errorTol)
    );


    remove_negative(reference.hf, errorTol);
    remove_negative(test_result.hf, errorTol);

    CHECK_THAT
    (
        reference.hf,
        Catch::Matchers::Approx(test_result.hf).epsilon(errorTol)
    );

    CHECK_THAT
    (
        reference.s,
        Catch::Matchers::Approx(test_result.s).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.gStd,
        Catch::Matchers::Approx(test_result.gStd).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.dCpdT,
        Catch::Matchers::Approx(test_result.dCpdT).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Cv,
        Catch::Matchers::Approx(test_result.Cv).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.es,
        Catch::Matchers::Approx(test_result.es).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.ea,
        Catch::Matchers::Approx(test_result.ea).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.K,
        Catch::Matchers::Approx(test_result.K).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Kp,
        Catch::Matchers::Approx(test_result.Kp).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Kc,
        Catch::Matchers::Approx(test_result.Kc).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.dKcdTbyKc,
        Catch::Matchers::Approx(test_result.dKcdTbyKc).epsilon(errorTol)
    );


}

TEST_CASE("Test gpuThermo (on CPU)", "[CPU]")
{

    constexpr double errorTol = 1E-9;
    SECTION("GRI")
    {
        auto test_result = CpuTestKernels::thermo(TestData::GRI);
        auto reference = OFReferenceKernels::thermo(TestData::GRI);
        thermoTests(test_result, reference, errorTol);
    }

    SECTION("H2")
    {
        auto test_result = CpuTestKernels::thermo(TestData::H2);
        auto reference = OFReferenceKernels::thermo(TestData::H2);
        thermoTests(test_result, reference, errorTol);
    }

}

TEST_CASE("Test gpuThermo (on GPU)", "[GPU]")
{

    constexpr double errorTol = 1E-7;
    SECTION("GRI")
    {
        auto test_result = GpuTestKernels::thermo(TestData::GRI);
        auto reference = OFReferenceKernels::thermo(TestData::GRI);
        thermoTests(test_result, reference, errorTol);
    }

    SECTION("H2")
    {
        auto test_result = GpuTestKernels::thermo(TestData::H2);
        auto reference = OFReferenceKernels::thermo(TestData::H2);
        thermoTests(test_result, reference, errorTol);
    }

}









static inline void reactionTests(const TestData::reactionResults& test_result, const TestData::reactionResults& reference, double errorTol)
{

    using namespace FoamGpu;

    CHECK_THAT
    (
        reference.Thigh,
        Catch::Matchers::Approx(test_result.Thigh).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Tlow,
        Catch::Matchers::Approx(test_result.Tlow).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.Kc,
        Catch::Matchers::Approx(test_result.Kc).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.kf,
        Catch::Matchers::Approx(test_result.kf).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.kr,
        Catch::Matchers::Approx(test_result.kr).epsilon(errorTol)
    );
    CHECK_THAT
    (
        reference.omega,
        Catch::Matchers::Approx(test_result.omega).epsilon(errorTol)
    );



    for (size_t i = 0; i < reference.dNdtByV.size(); ++i)
    {
        REQUIRE_THAT
        (
            reference.dNdtByV[i],
            Catch::Matchers::Approx(test_result.dNdtByV[i]).epsilon(errorTol)
        );
    }


    for (size_t i = 0; i < reference.ddNdtByVdcTp.size(); ++i)
    {
        REQUIRE_THAT
        (
            reference.ddNdtByVdcTp[i],
            Catch::Matchers::Approx(test_result.ddNdtByVdcTp[i]).epsilon(errorTol)
        );
    }


}

TEST_CASE("Test gpuReaction (on CPU)", "[CPU]")
{
    constexpr double errorTol = 1E-9;


    SECTION("GRI")
    {

        auto test_result = CpuTestKernels::reaction(TestData::GRI);
        auto reference = OFReferenceKernels::reaction(TestData::GRI);
        reactionTests(test_result, reference, errorTol);
    }

    SECTION("H2")
    {

        auto test_result = CpuTestKernels::reaction(TestData::H2);
        auto reference = OFReferenceKernels::reaction(TestData::H2);
        reactionTests(test_result, reference, errorTol);
    }


}

TEST_CASE("Test gpuReaction (on GPU)", "[GPU]")
{
    constexpr double errorTol = 1E-9;

    SECTION("GRI")
    {

        auto test_result = GpuTestKernels::reaction(TestData::GRI);
        auto reference = OFReferenceKernels::reaction(TestData::GRI);
        reactionTests(test_result, reference, errorTol);
    }

    SECTION("H2")
    {

        auto test_result = GpuTestKernels::reaction(TestData::H2);
        auto reference = OFReferenceKernels::reaction(TestData::H2);
        reactionTests(test_result, reference, errorTol);
    }


}



TEST_CASE("Test ludecompose (on CPU)", "[CPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;


    for (int i = 3; i < 50; ++i)
    {
        int size = i;


        std::vector<gScalar> vals(size*size);
        fill_random(vals);
        std::vector<gScalar> source(size, 1);


        auto [m_test, p_test, s_test] = CpuTestKernels::lu(vals, source);
        auto [m_ref, p_ref, s_ref] = OFReferenceKernels::lu(vals, source);

        REQUIRE_THAT
        (
            m_test,
            Catch::Matchers::Approx(m_ref).epsilon(errorTol)
        );
        REQUIRE_THAT
        (
            p_test,
            Catch::Matchers::Approx(p_ref).epsilon(errorTol)
        );
        REQUIRE_THAT
        (
            s_test,
            Catch::Matchers::Approx(s_ref).epsilon(errorTol)
        );

    }
}

TEST_CASE("Test ludecompose (on GPU)", "[GPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;


    for (int i = 3; i < 50; ++i)
    {
        int size = i;

        std::vector<gScalar> vals(size*size);
        fill_random(vals);
        std::vector<gScalar> source(size, 1);


        auto [m_test, p_test, s_test] = GpuTestKernels::lu(vals, source);
        auto [m_ref, p_ref, s_ref] = OFReferenceKernels::lu(vals, source);

        REQUIRE_THAT
        (
            m_test,
            Catch::Matchers::Approx(m_ref).epsilon(errorTol)
        );
        REQUIRE_THAT
        (
            p_test,
            Catch::Matchers::Approx(p_ref).epsilon(errorTol)
        );
        REQUIRE_THAT
        (
            s_test,
            Catch::Matchers::Approx(s_ref).epsilon(errorTol)
        );

    }
}



TEST_CASE("Test gpuOdeSystem (on CPU)", "[CPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;


    SECTION("H2"){

        auto test_result = CpuTestKernels::odesystem(TestData::Mechanism::H2);
        auto reference = OFReferenceKernels::odesystem(TestData::Mechanism::H2);


        CHECK_THAT
        (
            test_result.derivative,
            Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            test_result.jacobian,
            Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
        );


    }

    SECTION("GRI"){

        auto test_result = CpuTestKernels::odesystem(TestData::Mechanism::GRI);
        auto reference = OFReferenceKernels::odesystem(TestData::Mechanism::GRI);

        CHECK_THAT
        (
            test_result.derivative,
            Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            test_result.jacobian,
            Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
        );


    }
}

TEST_CASE("Test gpuOdeSystem (on GPU)", "[GPU]")
{
    using namespace FoamGpu;

    constexpr double errorTol = 1E-9;


    SECTION("H2"){

        auto test_result = GpuTestKernels::odesystem(TestData::Mechanism::H2);
        auto reference = OFReferenceKernels::odesystem(TestData::Mechanism::H2);


        CHECK_THAT
        (
            test_result.derivative,
            Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            test_result.jacobian,
            Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
        );


    }

    SECTION("GRI"){

        auto test_result = GpuTestKernels::odesystem(TestData::Mechanism::GRI);
        auto reference = OFReferenceKernels::odesystem(TestData::Mechanism::GRI);

        CHECK_THAT
        (
            test_result.derivative,
            Catch::Matchers::Approx(reference.derivative).epsilon(errorTol)
        );

        CHECK_THAT
        (
            test_result.jacobian,
            Catch::Matchers::Approx(reference.jacobian).epsilon(errorTol)
        );


    }
}


TEST_CASE("Test gpuOdeSolver (on CPU)", "[CPU]")
{
    using namespace FoamGpu;

    const gScalar xStart = 0.0;
    const gScalar xEnd = 1E-6;
    const gScalar dxTry = 1E-7;

    constexpr double errorTol = 1E-8;


    SECTION("GRI"){

        TestData::Mechanism mech = TestData::Mechanism::GRI;

        SECTION("Rosenbrock12")
        {
            std::string name = "Rosenbrock12";
            auto test_result = CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock23")
        {
            std::string name = "Rosenbrock23";
            auto test_result = CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            std::string name = "Rosenbrock34";
            auto test_result = CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }

    }

    SECTION("H2"){

        TestData::Mechanism mech = TestData::Mechanism::H2;

        SECTION("Rosenbrock12")
        {
            std::string name = "Rosenbrock12";
            auto test_result = CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock23")
        {
            std::string name = "Rosenbrock23";
            auto test_result = CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            std::string name = "Rosenbrock34";
            auto test_result = CpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }

    }

}

TEST_CASE("Test gpuOdeSolver (on GPU)", "[GPU]")
{
    using namespace FoamGpu;

    const gScalar xStart = 0.0;
    const gScalar xEnd = 1E-6;
    const gScalar dxTry = 1E-7;

    constexpr double errorTol = 1E-8;


    SECTION("GRI"){

        TestData::Mechanism mech = TestData::Mechanism::GRI;

        SECTION("Rosenbrock12")
        {
            std::string name = "Rosenbrock12";
            auto test_result = GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock23")
        {
            std::string name = "Rosenbrock23";
            auto test_result = GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            std::string name = "Rosenbrock34";
            auto test_result = GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }

    }

    SECTION("H2"){

        TestData::Mechanism mech = TestData::Mechanism::H2;

        SECTION("Rosenbrock12")
        {
            std::string name = "Rosenbrock12";
            auto test_result = GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock23")
        {
            std::string name = "Rosenbrock23";
            auto test_result = GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }
        SECTION("Rosenbrock34")
        {
            std::string name = "Rosenbrock34";
            auto test_result = GpuTestKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            auto reference = OFReferenceKernels::ode_solve(mech, name, xStart, xEnd, dxTry);
            remove_negative(test_result, errorTol);
            remove_negative(reference, errorTol);
            CHECK_THAT
            (
                test_result,
                Catch::Matchers::Approx(reference).epsilon(errorTol)
            );

        }

    }

}



TEST_CASE("for_each_index (on GPU)", "[GPU]"){

    CHECK(GpuTestKernels::test_for_each_index() == true);

}
/*
TEST_CASE("Test gpuKernelEvaluator"){

    CHECK(TestData::test_evaluator(1) == true);
    CHECK(TestData::test_evaluator(2) == true);

}
*/