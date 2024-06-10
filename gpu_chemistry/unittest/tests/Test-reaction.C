#include "catch.H"

#include "speciesTable.H"
#include "ArrheniusReactionRate.H"

#include "gpuReaction.H"

#include "test_utilities.H"
#include "create_gpu_inputs.H"
#include "create_foam_inputs.H"
#include "mechanisms.H"
#include "cpu_reference_results.H"
#include "gpu_reference_results.H"

/*
TEST_CASE("variant")
{
    using namespace FoamGpu;

    using Arrhenius = gpuArrheniusReactionRate;
    using ThirdBodyArrhenius = gpuThirdBodyArrheniusReactionRate;
    using ArrheniusLindemannFallOff = gpuFallOffReactionRate<gpuArrheniusReactionRate, gpuLindemannFallOffFunction>;
    using ArrheniusTroeFallOff = gpuFallOffReactionRate<gpuArrheniusReactionRate, gpuTroeFallOffFunction>;


    using ReactionRate =
    variant::variant
    <
        Arrhenius,
        ThirdBodyArrhenius,
        ArrheniusLindemannFallOff,
        ArrheniusTroeFallOff
    >;


    SECTION("Constructors")
    {
        REQUIRE_NOTHROW
        (
            ReactionRate()
        );
    }

}

TEST_CASE("Test gpuReactionRate")
{
    using namespace FoamGpu;


    SECTION("Constructors")
    {
        SECTION("Default")
        {
            auto f = [](){

                gpuReactionRate r;
                return gScalar(r.hasDdc());
            };

            CHECK(eval(f) == gScalar(false));
        }

        SECTION("Assignment")
        {
            gpuReactionRate rhs(gpuArrheniusReactionRate(0.1, 0.2, 0.3), false);
            auto f = [rhs](){

                gpuReactionRate lhs = rhs;
                return gScalar(lhs.hasDdc());
            };

            CHECK(eval(f) == gScalar(false));
        }

    }


    SECTION("operator()")
    {
        auto nSpecie = makeSpeciesTable(TestData::GRI).size();
        const Foam::scalarField c_cpu(nSpecie, 0.123);
        const device_vector<gScalar> c_gpu = host_vector<gScalar>(c_cpu.begin(), c_cpu.end());
        const auto c = make_mdspan(c_gpu, extents<1>{nSpecie});

        const gScalar p = 1E5;
        const gScalar T = 431.4321;
        const gLabel li = 0;
        SECTION("Arrhenius")
        {
            SECTION("Test 1")
            {
                gScalar A = 4.6e16;
                gScalar beta = -1.41;
                gScalar Ta = 14567.38636;
                gpuArrheniusReactionRate gpu(A, beta, Ta);
                Foam::ArrheniusReactionRate cpu(A, beta, Ta);
                REQUIRE(eval([=](){return gpu(p, T, c);}) == Approx(cpu(p, T, c_cpu, li)).epsilon(errorTol));

            }
            SECTION("Test 1")
            {
                gScalar A = 1E+11;
                gScalar beta = 0;
                gScalar Ta = 20127.64955;
                gpuArrheniusReactionRate gpu(A, beta, Ta);
                Foam::ArrheniusReactionRate cpu(A, beta, Ta);
                REQUIRE(eval([=](){return gpu(p, T, c);}) == Approx(cpu(p, T, c_cpu, li)).epsilon(errorTol));

            }
        }

    }



}


TEST_CASE("gpuSpeciesCoeffs pow")
{

    using namespace FoamGpu;

    SECTION("Test1")
    {
        const gScalar base = 43.421;
        const gScalar exp = 1.0;

        Foam::specieExponent cpu(exp);
        gpuSpecieExponent gpu(exp);

        REQUIRE
        (
            eval([=](){return FoamGpu::speciePow(base, exp); })
            == Approx(Foam::pow(base, cpu)).epsilon(errorTol)
        );
    }



    SECTION("Test2")
    {
        const gScalar base = 43.421;
        const gLabel exp = 1;

        const Foam::specieExponent cpu(exp);
        const gpuSpecieExponent gpu(exp);
        REQUIRE
        (
            eval([=](){return FoamGpu::speciePow(base, exp); })
            == Approx(Foam::pow(base, cpu)).epsilon(errorTol)
        );
    }

    SECTION("Test3")
    {
        const gScalar base = 43.421;
        const gLabel exp = 5;

        const Foam::specieExponent cpu(exp);
        const gpuSpecieExponent gpu(exp);

        REQUIRE
        (
            eval([=](){return FoamGpu::speciePow(base, exp); })
            == Approx(Foam::pow(base, cpu)).epsilon(errorTol)
        );
    }


    SECTION("Test4")
    {

        const gScalar base = 1.32;
        const gLabel exp = 2;
        const gLabel n = 60;
        const Foam::specieExponent cpu(exp);
        const FoamGpu::gpuSpecieExponent gpu(exp);

        auto f = [=]()
        {

            const Foam::specieExponent er = cpu;
            gScalar dCrcj = 1.0;
            for (gLabel i = 0; i < n; ++i)
            {
                dCrcj *= er*Foam::pow(base, er - Foam::specieExponent(gLabel(1)));
            }
            return dCrcj;
        };

        auto f2 = [=]()
        {

            const FoamGpu::gpuSpecieExponent er =gpu;
            gScalar dCrcj = 1.0;
            for (gLabel i = 0; i < n; ++i)
            {
                dCrcj *= er*FoamGpu::speciePow(base, er - FoamGpu::gpuSpecieExponent(gLabel(1)));

            }
            return dCrcj;
        };

        REQUIRE
        (
            eval(f) == Approx(f2()).epsilon(errorTol)
        );

    }


}
*/


static inline void reactionTests(TestData::Mechanism mech)
{

    using namespace FoamGpu;

    auto results_gpu = TestData::reaction_results_gpu(mech);
    auto results_cpu = TestData::reaction_results(mech);


    REQUIRE_THAT
    (
        results_cpu.Thigh,
        Catch::Matchers::Approx(results_gpu.Thigh).epsilon(errorTol)
    );
    REQUIRE_THAT
    (
        results_cpu.Tlow,
        Catch::Matchers::Approx(results_gpu.Tlow).epsilon(errorTol)
    );
    REQUIRE_THAT
    (
        results_cpu.Kc,
        Catch::Matchers::Approx(results_gpu.Kc).epsilon(errorTol)
    );
    REQUIRE_THAT
    (
        results_cpu.kf,
        Catch::Matchers::Approx(results_gpu.kf).epsilon(errorTol)
    );
    REQUIRE_THAT
    (
        results_cpu.kr,
        Catch::Matchers::Approx(results_gpu.kr).epsilon(errorTol)
    );
    REQUIRE_THAT
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
