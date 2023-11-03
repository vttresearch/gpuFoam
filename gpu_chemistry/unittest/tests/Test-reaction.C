#include "catch.H"

#include "speciesTable.H"
#include "ArrheniusReactionRate.H"

#include "gpuReaction.H"

#include "test_utilities.H"
#include "create_gpu_inputs.H"
#include "create_foam_inputs.H"

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


static inline void reactionTests(TestData::Mechanism mech)
{

    using namespace FoamGpu;

    auto cpu_reactions = makeCpuReactions(mech);
    auto gpu_reactions_temp = makeGpuReactions(mech);

    const gLabel nSpecie = TestData::speciesCount(mech);
    const gLabel nEqns = TestData::equationCount(mech);
    const gScalar p = 1E5;
    const gScalar T = 431.4321;
    const gLabel li = 0;

    const Foam::scalarField c_cpu(nSpecie, 0.123);
    auto c_gpu = toDeviceVector(c_cpu);
    //const device_vector<gScalar> c_gpu = host_vector<gScalar>(c_cpu.begin(), c_cpu.end());

    //device_vector<gpuReaction>  gpu_reactions(gpu_reactions_temp.begin(), gpu_reactions_temp.end());

    auto gpu_reactions = toDeviceVector(gpu_reactions_temp);
    CHECK(cpu_reactions.size() == gLabel(gpu_reactions.size()));

    SECTION("Thigh/Tlow")
    {
        for (gLabel i = 0; i < cpu_reactions.size(); ++i)
        {
            const auto& cpu = cpu_reactions[i];
            const auto  gpu = &(gpu_reactions[i]);

            REQUIRE(eval([=](){return gpu->Tlow();}) == Approx(cpu.Tlow()).epsilon(errorTol));
            REQUIRE(eval([=](){return gpu->Thigh();}) == Approx(cpu.Thigh()).epsilon(errorTol));
        }

    }


    SECTION("omega")
    {

        const auto c = make_mdspan(c_gpu, extents<1>{nSpecie});

        for (gLabel i = 0; i < cpu_reactions.size(); ++i)
        {
            const auto& cpu = cpu_reactions[i];
            const auto  gpu =&(gpu_reactions[i]);

            auto f_gpu = [=](){
                return gpu->omega(p, T, c);
            };

            auto f_cpu = [&](){
                Foam::scalar omegaf = 0.3;
                Foam::scalar omegar = 0.4;
                return cpu.omega(p, T, c_cpu, li, omegaf, omegar);
            };

            REQUIRE(eval(f_gpu) == Approx(f_cpu()).epsilon(errorTol));
        }

    }



    SECTION("Kc/kf/kr/dkfdT/dkrdT")
    {


        const auto c = make_mdspan(c_gpu, extents<1>{nSpecie});

        for (gLabel i = 0; i < cpu_reactions.size(); ++i)
        {
            const auto& cpu = cpu_reactions[i];
            const auto  gpu =&(gpu_reactions[i]);



            REQUIRE
            (
                eval([=](){return gpu->Kc(p, T);})
                == Approx(cpu.Kc(p, T)).epsilon(errorTol)
            );

            REQUIRE
            (
                eval([=](){return gpu->kf(p, T, c);})
                == Approx(cpu.kf(p, T, c_cpu, li)).epsilon(errorTol)
            );
            REQUIRE
            (
                eval([=](){return gpu->kr(p, T, c);})
                == Approx(cpu.kr(p, T, c_cpu, li)).epsilon(errorTol)
            );

            REQUIRE
            (
                eval([=](){return gpu->kr(32.0, p, T, c);})
                == Approx(cpu.kr(32.0, p, T, c_cpu, li)).epsilon(errorTol)
            );

            REQUIRE
            (
                eval([=](){return gpu->dkfdT(p, T, c);})
                == Approx(cpu.dkfdT(p, T, c_cpu, li)).epsilon(errorTol)
            );

            REQUIRE
            (
                eval([=](){return gpu->dkrdT(p, T, c, 0.1, 0.3);})
                == Approx(cpu.dkrdT(p, T, c_cpu, li, 0.1, 0.3)).epsilon(errorTol)
            );



            Foam::scalarField dkfdc_cpu(nSpecie, 0.1);
            device_vector<gScalar> dkfdc_gpu_temp(nSpecie, 0.4);
            auto dkfdc_gpu = make_mdspan(dkfdc_gpu_temp, extents<1>{nSpecie});

            eval([=](){gpu->dkfdc(p, T, c, dkfdc_gpu); return 0;});
            cpu.dkfdc(p, T, c_cpu, li, dkfdc_cpu);

            REQUIRE_THAT
            (
                toStdVector(dkfdc_gpu_temp),
                Catch::Matchers::Approx(toStdVector(dkfdc_cpu)).epsilon(errorTol)
            );


            Foam::scalarField dkrdc_cpu(nSpecie, 0.3);
            device_vector<gScalar> dkrdc_gpu_temp(nSpecie, 0.45);
            auto dkrdc_gpu = make_mdspan(dkrdc_gpu_temp, extents<1>{nSpecie});


            eval([=](){gpu->dkrdc(p, T, c, dkfdc_gpu, 0.1, dkrdc_gpu); return 0;});
            cpu.dkrdc(p, T, c_cpu, li, dkfdc_cpu, 0.1, dkrdc_cpu);

            REQUIRE_THAT
            (
                toStdVector(dkrdc_gpu_temp),
                Catch::Matchers::Approx(toStdVector(dkrdc_cpu)).epsilon(errorTol)
            );


        }
    }




    SECTION("dNdtByV")
    {


        //Here it is important to have same initial condition as the function only modifies
        //certain values
        Foam::scalarField res_cpu(nSpecie, 0.435);
        device_vector<gScalar> res_gpu(nSpecie, 0.435);

        for (gLabel i = 0; i < cpu_reactions.size(); ++i)
        {
            const auto& cpu = cpu_reactions[i];
            const auto  gpu = &(gpu_reactions[i]);


            auto c = make_mdspan(c_gpu, extents<1>{nSpecie});
            auto res = make_mdspan(res_gpu, extents<1>{nSpecie});

            auto f = [=](){
                gpu->dNdtByV(p, T, c, res);
                return 0;
            };
            eval(f);

            cpu.dNdtByV(p, T, c_cpu, li, res_cpu, false, Foam::List<gLabel>{}, 0);

            auto dNdtByV_cpu = toStdVector(res_gpu);
            auto dNdtByV_gpu = toStdVector(res_cpu);

            for (gLabel i = 0; i < gLabel(dNdtByV_cpu.size()); ++i)
            {
                REQUIRE(dNdtByV_gpu[i] == Approx(dNdtByV_cpu[i]).epsilon(errorTol));
            }
        }

    }

    SECTION("ddNdtByVdcTp")
    {

        Foam::List<gLabel> c2s;
        gLabel csi0 = 0;
        gLabel Tsi = nSpecie;



        Foam::scalarField dNdtByV_cpu(nSpecie, 1.0);
        Foam::scalarSquareMatrix ddNdtByVdcTp_cpu(nEqns, 11.3);
        Foam::scalarField cTpWork0_cpu(nSpecie, 11.1);
        Foam::scalarField cTpWork1_cpu(nSpecie, 13.5);

        auto dNdtByV_gpu = toDeviceVector(dNdtByV_cpu);
        auto ddNdtByVdcTp_gpu = device_vector<gScalar>
            (
                ddNdtByVdcTp_cpu.v(),
                ddNdtByVdcTp_cpu.v() + ddNdtByVdcTp_cpu.size()
            );

        auto cTpWork0_gpu = toDeviceVector(cTpWork0_cpu);
        auto cTpWork1_gpu = toDeviceVector(cTpWork1_cpu);

        for (gLabel i = 0; i < cpu_reactions.size(); ++i)
        {
            const auto& cpu = cpu_reactions[i];
            const auto  gpu = &(gpu_reactions[i]);

            cpu.ddNdtByVdcTp
            (
                p,
                T,
                c_cpu,
                li,
                dNdtByV_cpu,
                ddNdtByVdcTp_cpu,
                false,
                c2s,
                csi0,
                Tsi,
                cTpWork0_cpu,
                cTpWork1_cpu
            );

            auto f =
            [
                =,
                c = make_mdspan(c_gpu, extents<1>{nSpecie}),
                ddNdtByVdcTp = make_mdspan(ddNdtByVdcTp_gpu, extents<2>{nEqns, nEqns}),
                cTpWork0 = make_mdspan(cTpWork0_gpu, extents<1>{nSpecie}),
                cTpWork1 = make_mdspan(cTpWork1_gpu, extents<1>{nSpecie})
            ]
            ()
            {
                auto params = computeReactionParameters(*gpu, c, p, T);

                gpu->ddNdtByVdcTp
                (
                    p,
                    T,
                    c,
                    ddNdtByVdcTp,
                    cTpWork0,
                    cTpWork1,
                    params

                );

                return 0;
            };

            eval(f);

            std::vector<double> r_cpu(ddNdtByVdcTp_cpu.v(), ddNdtByVdcTp_cpu.v() + ddNdtByVdcTp_cpu.size());
            std::vector<double> r_gpu = toStdVector(ddNdtByVdcTp_gpu);
            REQUIRE_THAT(r_gpu, Catch::Matchers::Approx(r_cpu).epsilon(errorTol));

        }

    }


    SECTION("ddNdtByVdcTp with small concentrations")
    {
        Foam::List<gLabel> c2s;
        gLabel csi0 = 0;
        gLabel Tsi = nSpecie;

        Foam::scalarField c_cpu2(nSpecie);
        fill_random(c_cpu2);
        c_cpu2[3] = 1E-4;
        c_cpu2[5] = 1E-5;
        c_cpu2[6] = 1E-7;
        c_cpu2[7] = gpuSmall;
        c_cpu2[8] = 0.9*gpuSmall;
        c_cpu2[9] = gpuVSmall;

        auto c_gpu2 = toDeviceVector(c_cpu2);

        Foam::scalarField dNdtByV_cpu(nSpecie, 1.0);
        Foam::scalarSquareMatrix ddNdtByVdcTp_cpu(nEqns, 11.3);
        Foam::scalarField cTpWork0_cpu(nSpecie, 11.1);
        Foam::scalarField cTpWork1_cpu(nSpecie, 13.5);


        auto dNdtByV_gpu = toDeviceVector(dNdtByV_cpu);
        auto ddNdtByVdcTp_gpu = device_vector<gScalar>
            (
                ddNdtByVdcTp_cpu.v(),
                ddNdtByVdcTp_cpu.v() + ddNdtByVdcTp_cpu.size()
            );

        auto cTpWork0_gpu = toDeviceVector(cTpWork0_cpu);
        auto cTpWork1_gpu = toDeviceVector(cTpWork1_cpu);


        for (gLabel i = 0; i < cpu_reactions.size(); ++i)
        {
            const auto& cpu = cpu_reactions[i];
            const auto  gpu = &(gpu_reactions[i]);

            cpu.ddNdtByVdcTp
            (
                p,
                T,
                c_cpu2,
                li,
                dNdtByV_cpu,
                ddNdtByVdcTp_cpu,
                false,
                c2s,
                csi0,
                Tsi,
                cTpWork0_cpu,
                cTpWork1_cpu
            );

            auto f =
            [
                =,
                c = make_mdspan(c_gpu2, extents<1>{nSpecie}),
                ddNdtByVdcTp = make_mdspan(ddNdtByVdcTp_gpu, extents<2>{nEqns, nEqns}),
                cTpWork0 = make_mdspan(cTpWork0_gpu, extents<1>{nSpecie}),
                cTpWork1 = make_mdspan(cTpWork1_gpu, extents<1>{nSpecie})
            ]
            ()
            {
                auto params = computeReactionParameters(*gpu, c, p, T);

                gpu->ddNdtByVdcTp
                (
                    p,
                    T,
                    c,
                    ddNdtByVdcTp,
                    cTpWork0,
                    cTpWork1,
                    params
                );

                return 0;
            };


            eval(f);




            std::vector<double> r_cpu(ddNdtByVdcTp_cpu.v(), ddNdtByVdcTp_cpu.v() + ddNdtByVdcTp_cpu.size());
            std::vector<double> r_gpu = toStdVector(ddNdtByVdcTp_gpu);

            REQUIRE_THAT(r_gpu, Catch::Matchers::Approx(r_cpu).epsilon(errorTol));


        }


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
