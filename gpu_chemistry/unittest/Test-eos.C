#include "catch.H"


#include "test_utilities.H"


#include "thermodynamicConstants.H"
#include "fundamentalConstants.H"
#include "physicoChemicalConstants.H"
#include "specieExponent.H"

TEST_CASE("Test gpuConstans")
{
    SECTION("Physical")
    {


        CHECK(eval([](){return gpuRR;}) == Foam::constant::thermodynamic::RR);
        CHECK(eval([](){return gpuPstd;}) == Foam::constant::thermodynamic::Pstd);
        CHECK(eval([](){return gpuTstd;}) == Foam::constant::thermodynamic::Tstd);
        CHECK(eval([](){return gpuNA;}) == Foam::constant::physicoChemical::NA.value());
        CHECK(eval([](){return gpuk;}) == Foam::constant::physicoChemical::k.value());
        CHECK(eval([](){return gpuR;}) == Foam::constant::physicoChemical::R.value());

    }

    SECTION("Numeric")
    {
        CHECK(eval([](){return gpuVGreat;}) == Foam::vGreat);
        CHECK(eval([](){return gpuVSmall;}) == Foam::vSmall);
        CHECK(eval([](){return gpuSmall;}) == Foam::small);
        CHECK(eval([](){return gpuGreat;}) == Foam::great);
    }
}


TEST_CASE("Test perfectGas"){

    using namespace FoamGpu;
    
    SECTION("Constructors")
    {
        REQUIRE_NOTHROW(gpuPerfectGas());
    }


    gScalar molWeight = 0.32;
    gScalar Y = 0.1;

    const Foam::perfectGas<Foam::specie> cpu
    (
        Foam::specie("temp", Y, molWeight)
        //specie(dict) //Can not construct from dict because of implicit conversions...
    );

    const gpuPerfectGas gpu
    (
        Y, molWeight
    );


    SECTION("thermo properties")
    {
        const gScalar p = 1E5;
        const gScalar T = 3542.324;

        CHECK(eval([=](){return gpu.R();}) == Approx(cpu.R()).epsilon(errorTol));
        CHECK(eval([=](){return gpu.rho(p, T);}) == Approx(cpu.rho(p, T)).epsilon(errorTol));
        CHECK(eval([=](){return gpu.H(p, T);}) == Approx(cpu.H(p, T)).epsilon(errorTol));
        CHECK(eval([=](){return gpu.Cp(p, T);}) == Approx(cpu.Cp(p, T)).epsilon(errorTol));
        CHECK(eval([=](){return gpu.E(p, T);}) == Approx(cpu.E(p, T)).epsilon(errorTol));
        CHECK(eval([=](){return gpu.Cv(p, T);}) == Approx(cpu.Cv(p, T)).epsilon(errorTol));
        CHECK(eval([=](){return gpu.Sp(p, T);}) == Approx(cpu.Sp(p, T)).epsilon(errorTol));
        //CHECK(eval([=](){return gpu.Sv(p, T);}) == cpu.Sv(p, T)); //throws
        CHECK(eval([=](){return gpu.psi(p, T);}) == Approx(cpu.psi(p, T)).epsilon(errorTol));
        CHECK(eval([=](){return gpu.Z(p, T);}) == Approx(cpu.Z(p, T)).epsilon(errorTol));
        CHECK(eval([=](){return gpu.CpMCv(p, T);}) == Approx(cpu.CpMCv(p, T)).epsilon(errorTol));
        CHECK(eval([=](){return gpu.alphav(p, T);}) == Approx(cpu.alphav(p, T)).epsilon(errorTol));

    }


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

}