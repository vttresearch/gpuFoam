#include "catch.H"


#include "test_utilities.H"
#include "create_gpu_inputs.H"
#include "create_foam_inputs.H"
#include "makeGpuThermo.H"

template<class T, unsigned N>
static inline auto toArray(Foam::FixedList<T, N> a)
{

    std::array<T, N> ret{};
    for (size_t i = 0; i < N; ++i)
    {
        ret[i] = a[i];
    }
    return ret;
}



static inline auto toArray(typename FoamGpu::gpuThermo::coeffArray a)
{
    std::array<double, 7> ret{};
    for (size_t i = 0; i < 7; ++i)
    {
        ret[i] = a[i];
    }
    return ret;
}

Foam::dictionary get_dictionary()
{
    using namespace Foam;

    OStringStream os;
    os << "OH" << endl;
    os << "{" << endl;
    os << "specie {molWeight 17; massFraction 1;}" << endl;
    os << "thermodynamics { Tlow 200; Thigh 3500; Tcommon 1000; " << endl;
    os << "highCpCoeffs (1 1 1 1 1 1 1);" << endl;
    os << "lowCpCoeffs  (1 1 1 1 1 1 1);" << endl;

    os << "}" << endl;
    os << "}" << endl;


    IStringStream is(os.str());

    dictionary dict(is);

    auto sdict = dict.subDict("OH");


    return dict.subDict("OH");

}

template<class T>
static T arithmetic_kernel(T& lhs, T& rhs){
    lhs += rhs;
    auto r1 = lhs + rhs;
    auto r2 = 3*r1;
    auto r3 = (r1 == r2);
    return r3;
}

TEST_CASE("Test gpuThermo"){

    using namespace FoamGpu;


    SECTION("Constructors")
    {
        REQUIRE_NOTHROW(gpuThermo());
    }


    auto dict = get_dictionary();



    const Foam::species::thermo<Foam::janafThermo<Foam::perfectGas<Foam::specie> >,Foam::sensibleEnthalpy>
    cpu
    (
        "Something",
        dict
    );

    const gpuThermo gpu = makeGpuThermo(cpu, dict);


    CHECK(toArray(cpu.highCpCoeffs())  == toArray(gpu.highCpCoeffs()));
    CHECK(toArray(cpu.lowCpCoeffs())  == toArray(gpu.lowCpCoeffs()));



    SECTION("gpuThermo operator +=")
    {

        SECTION("non self assignment"){
            auto cpu1(cpu);
            auto cpu2(cpu1);

            auto gpu1(gpu);
            auto gpu2(gpu1);

            gpu1 += gpu2;
            cpu1 += cpu2;

            CHECK(cpu1.W() == gpu1.W());
            CHECK(cpu1.Y() == gpu1.Y());
            CHECK(toArray(cpu1.highCpCoeffs())  == toArray(gpu1.highCpCoeffs()));
            CHECK(toArray(cpu1.lowCpCoeffs())  == toArray(gpu1.lowCpCoeffs()));

        }


        SECTION("self assignment"){

            auto cpu1(cpu);
            auto gpu1(gpu);

            cpu1 += cpu1;
            gpu1 += gpu1;

            CHECK(cpu1.W() == gpu1.W());
            CHECK(cpu1.Y() == gpu1.Y());
            CHECK(toArray(cpu1.highCpCoeffs())  == toArray(gpu1.highCpCoeffs()));
            CHECK(toArray(cpu1.lowCpCoeffs())  == toArray(gpu1.lowCpCoeffs()));


        }

    }

    SECTION("gpuThermo arithmetic_kernel")
    {
        auto cpu1(cpu);
        auto cpu2(cpu1);

        auto gpu1(gpu);
        auto gpu2(gpu1);

        auto rgpu = arithmetic_kernel(gpu1, gpu2);
        auto rcpu = arithmetic_kernel(cpu1, cpu2);



        CHECK(rcpu.W() == rgpu.W());
        CHECK(rcpu.Y() == rgpu.Y());
        CHECK(toArray(rcpu.highCpCoeffs())  == toArray(rgpu.highCpCoeffs()));
        CHECK(toArray(rcpu.lowCpCoeffs())  == toArray(rgpu.lowCpCoeffs()));

    }

}




TEST_CASE("Test gpuThermo properties")
{
    using namespace FoamGpu;


    SECTION("Gri")
    {
        auto cpuThermos = TestData::makeCpuThermos(TestData::GRI);
        auto gpuThermos_temp = TestData::makeGpuThermos(TestData::GRI);
        device_vector<gpuThermo> gpuThermos(gpuThermos_temp.begin(), gpuThermos_temp.end());
        CHECK(cpuThermos.size() == gLabel(gpuThermos.size()));

        for (gLabel i = 0; i < cpuThermos.size(); ++i)
        {
            const auto& cpu = cpuThermos[i];
            const auto  gpu = &(gpuThermos[i]);

            gScalar p = 1E5;
            gScalar T = 431.4321;

            REQUIRE(eval([=](){return gpu->W();}) == Approx(cpu.W()).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Y();}) == Approx(cpu.Y()));
            REQUIRE(eval([=](){return gpu->R();}) == Approx(cpu.R()));
            REQUIRE(eval([=](){return gpu->Cp(p, T);}) == Approx(cpu.Cp(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Ha(p, T);}) == Approx(cpu.Ha(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Hs(p, T);}) == Approx(cpu.Hs(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Hf(    );}) == Approx(cpu.Hf(    )).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->S(p, T);}) == Approx(cpu.S(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Gstd(T);}) == Approx(cpu.Gstd(T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->dCpdT(p, T);}) == Approx(cpu.dCpdT(p, T)).epsilon(errorTol));

            REQUIRE(eval([=](){return gpu->Cv(p, T);}) == Approx(cpu.Cv(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Es(p, T);}) == Approx(cpu.Es(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Ea(p, T);}) == Approx(cpu.Ea(p, T)).epsilon(errorTol));

            REQUIRE(eval([=](){return gpu->K(p, T);}) == Approx(cpu.K(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Kp(p, T);}) == Approx(cpu.Kp(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Kc(p, T);}) == Approx(cpu.Kc(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->dKcdTbyKc(p, T);}) == Approx(cpu.dKcdTbyKc(p, T)).epsilon(errorTol));

        }
    }

    SECTION("H2")
    {
        auto cpuThermos = TestData::makeCpuThermos(TestData::H2);
        auto gpuThermos_temp = TestData::makeGpuThermos(TestData::H2);
        device_vector<gpuThermo> gpuThermos(gpuThermos_temp.begin(), gpuThermos_temp.end());
        CHECK(cpuThermos.size() == gLabel(gpuThermos.size()));

        for (gLabel i = 0; i < cpuThermos.size(); ++i)
        {
            const auto& cpu = cpuThermos[i];
            const auto  gpu = &(gpuThermos[i]);

            gScalar p = 1E5;
            gScalar T = 431.4321;

            REQUIRE(eval([=](){return gpu->W();}) == Approx(cpu.W()).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Y();}) == Approx(cpu.Y()));
            REQUIRE(eval([=](){return gpu->R();}) == Approx(cpu.R()));
            REQUIRE(eval([=](){return gpu->Cp(p, T);}) == Approx(cpu.Cp(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Ha(p, T);}) == Approx(cpu.Ha(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Hs(p, T);}) == Approx(cpu.Hs(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Hf(    );}) == Approx(cpu.Hf(    )).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->S(p, T);}) == Approx(cpu.S(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Gstd(T);}) == Approx(cpu.Gstd(T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->dCpdT(p, T);}) == Approx(cpu.dCpdT(p, T)).epsilon(errorTol));

            REQUIRE(eval([=](){return gpu->Cv(p, T);}) == Approx(cpu.Cv(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Es(p, T);}) == Approx(cpu.Es(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Ea(p, T);}) == Approx(cpu.Ea(p, T)).epsilon(errorTol));

            REQUIRE(eval([=](){return gpu->K(p, T);}) == Approx(cpu.K(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Kp(p, T);}) == Approx(cpu.Kp(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Kc(p, T);}) == Approx(cpu.Kc(p, T)).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->dKcdTbyKc(p, T);}) == Approx(cpu.dKcdTbyKc(p, T)).epsilon(errorTol));

        }
    }



}





