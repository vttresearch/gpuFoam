#include "catch.H"

#include "cpu_reference_results.H"
#include "gpu_reference_results.H"


#include "test_utilities.H"
#include "create_foam_inputs.H"
#include "create_gpu_inputs.H"

#include "thermodynamicConstants.H"
#include "fundamentalConstants.H"
#include "physicoChemicalConstants.H"
#include "specieExponent.H"





TEST_CASE("Test gpuConstans")
{
    auto cpu_result = TestData::constant_results();
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

    SECTION("Constructors")
    {
        REQUIRE_NOTHROW(gpuPerfectGas());
    }




    SECTION("thermo properties")
    {
        const gScalar p = 1E5;
        const gScalar T = 3542.324;
        const gScalar molWeight = 0.32;
        const gScalar Y = 0.1;

        auto cpu_result = TestData::thermo_result(p, T, Y, molWeight);
        auto gpu_result = TestData::thermo_result_gpu(p, T, Y, molWeight);

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

    /*




    const Foam::perfectGas<Foam::specie> cpu
    (
        Foam::specie("temp", Y, molWeight)
        //specie(dict) //Can not construct from dict because of implicit conversions...
    );

    const gpuPerfectGas gpu
    (
        Y, molWeight
    );



    //Arithmetic operations are not run on kernels, only upon construction.
    //Therefore only tested here on the host

    SECTION("operator+=")
    {

        SECTION("self assignment")
        {
            auto gpu1(gpu);
            auto cpu1(cpu);

            gpu1 += gpu1;
            cpu1 += cpu1;

            CHECK(gpu1.W() == cpu1.W());
            CHECK(gpu1.Y() == cpu1.Y());
        }

        SECTION("non-self assignment")
        {
            auto gpu1(gpu);
            auto cpu1(cpu);
            auto gpu2(gpu);
            auto cpu2(cpu);

            gpu1 += gpu2;
            cpu1 += cpu2;

            CHECK(gpu1.W() == cpu1.W());
            CHECK(gpu1.Y() == cpu1.Y());
        }

    }

    SECTION("operator+")
    {
        auto gpu1(gpu);
        auto cpu1(cpu);
        auto gpu2(gpu);
        auto cpu2(cpu);

        CHECK((gpu1+gpu2).W() == (cpu1+cpu2).W());
        CHECK((gpu1+gpu2).Y() == (cpu1+cpu2).Y());

    }
    SECTION("operator*")
    {
        auto gpu1(gpu);
        auto cpu1(cpu);
        CHECK((3*gpu1).W() == (3*cpu1).W());
        CHECK((3*gpu1).Y() == (3*cpu1).Y());

    }

    SECTION("operator==")
    {
        auto gpu1(gpu);
        auto cpu1(cpu);
        auto gpu2(gpu);
        auto cpu2(cpu);

        CHECK((gpu1==gpu2).W() == (cpu1==cpu2).W());
        CHECK((gpu1==gpu2).Y() == (cpu1==cpu2).Y());

    }
    */
}