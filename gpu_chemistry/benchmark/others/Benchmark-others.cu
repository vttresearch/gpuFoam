#define CATCH_CONFIG_ENABLE_BENCHMARKING
//#define CATCH_CONFIG_MAIN
#include "catch.H"

#include "host_device_vectors.H"
#include "test_utilities.H"
#include "create_gpu_inputs.H"
#include "ludecompose.H"
#include "gpuODESystem.H"
#include "makeGpuOdeSolver.H"



static constexpr TestData::Mechanism mech = TestData::GRI;

TEST_CASE("LU")
{



    const gLabel nEqns = TestData::equationCount(mech);



    std::vector<gScalar> vals(nEqns * nEqns);


    fill_random(vals);

    device_vector<gScalar> matrix(vals.begin(), vals.end());

    device_vector<gLabel> pivot(nEqns, 0);
    device_vector<gScalar> v(nEqns, 0);
    device_vector<gScalar> source(nEqns, 0);

    auto m_span = make_mdspan(matrix, extents<2>{nEqns, nEqns});
    auto p_span = make_mdspan(pivot, extents<1>{nEqns});
    auto v_span = make_mdspan(v, extents<1>{nEqns});
    auto s_span = make_mdspan(source, extents<1>{nEqns});

    auto op1 = [=] __device__() {
        FoamGpu::LUDecompose(m_span, p_span, v_span);
        return p_span(4);
    };

    auto op2 = [=] __device__ (){
        FoamGpu::LUBacksubstitute(m_span, p_span, s_span);
        return s_span(5);
    };


    BENCHMARK("WARMUP"){
        return eval(op1);
    };

    BENCHMARK("LUdecompose"){

        return eval(op1);
    };

    BENCHMARK("LUbacksubstitute"){

        return eval(op2);
    };




}


TEST_CASE("gpuODESystem"){

    using namespace FoamGpu;

    auto gpu_thermos = toDeviceVector(makeGpuThermos(mech));
    auto gpu_reactions = toDeviceVector(makeGpuReactions(mech));

    const gLabel nCells = 1;
    const gLabel nSpecie = TestData::speciesCount(mech);
    const gLabel nEqns = TestData::equationCount(mech);


    gpuODESystem system
    (
        nEqns,
        gLabel(gpu_reactions.size()),
        make_raw_pointer(gpu_thermos.data()),
        make_raw_pointer(gpu_reactions.data())
    );

    std::vector<gScalar> vals(nEqns);
    fill_random(vals);
    device_vector<gScalar> y(vals.begin(), vals.end());

    device_vector<gScalar> dy(nEqns);
    device_vector<gScalar> J(nEqns * nEqns);

    memoryResource_t memory(nCells, nSpecie);
    auto buffers = toDeviceVector(splitToBuffers(memory));

    auto op1 = [        =,
                buffers = make_mdspan(buffers, extents<1>{1}),
                y       = make_mdspan(y, extents<1>{nEqns}),
                dy      = make_mdspan(dy, extents<1>{nEqns})
                ]__device__() {
        system.derivatives(y, dy, buffers[0]);
        return dy(5);
    };

    auto op2 =
        [        =,
         buffers = make_mdspan(buffers, extents<1>{nCells}),
         y       = make_mdspan(y, extents<1>{nEqns}),
         dy      = make_mdspan(dy, extents<1>{nEqns}),
         J = make_mdspan(J, extents<2>{nEqns, nEqns})
         ] __device__() {
            system.jacobian(y, dy, J, buffers[0]);
            return J(3, 3);
        };

    BENCHMARK("derivatives"){
        return eval(op1);
    };

    BENCHMARK("Jacobian"){
        return eval(op2);
    };

}

TEST_CASE("gpuReactionRate"){

}

TEST_CASE("gpuReaction"){

    using namespace FoamGpu;
    auto reactions = toDeviceVector(makeGpuReactions(mech));

    const gLabel nReactions = reactions.size();
    const gLabel nSpecie = TestData::speciesCount(mech);
    const gLabel nEqns = TestData::equationCount(mech);

    const gScalar p = 1E5;
    const gScalar T = 900.0;

    device_vector<gScalar> c = [=](){
        std::vector<gScalar> vals(nSpecie);
        assign_test_concentration(vals, mech);
        //fill_random(vals);
        return device_vector<gScalar>(vals.begin(), vals.end());
    }();

    device_vector<gScalar> dndt = [=](){
        std::vector<gScalar> vals(nEqns);
        return device_vector<gScalar>(vals.begin(), vals.end());
    }();

    BENCHMARK("dNdtByV"){

        auto op =
        [
            =,
            c = make_mdspan(c, extents<1>{nSpecie}),
            dndt = make_mdspan(dndt, extents<1>{nEqns}),
            reactions = make_mdspan(reactions, extents<1>{nReactions})
        ]__device__(){

            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i){

                reactions[i].dNdtByV(p, T, c, dndt);
                ret += dndt[4];
            }
            return ret;

        };

        return eval(op);
    };


    device_vector<gScalar> work1(nEqns);
    device_vector<gScalar> work2(nEqns);
    device_vector<gScalar> J(nEqns * nEqns);

    BENCHMARK("ddNdtByVdcTp"){

        auto op =
        [
            =,
            c = make_mdspan(c, extents<1>{nSpecie}),
            work1 = make_mdspan(work1, extents<1>{nEqns}),
            work2 = make_mdspan(work2, extents<1>{nEqns}),
            dJ = make_mdspan(J, extents<2>{nEqns, nEqns}),
            reactions = make_mdspan(reactions, extents<1>{nReactions})
        ]__device__(){

            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i){
                const auto& reaction = reactions[i];
                auto params = computeReactionParameters(reaction, c, p, T);
                reaction.ddNdtByVdcTp( p, T, c, dJ, work1, work2, params);

                ret += dJ(4,4);
            }
            return ret;

        };

        return eval(op);
    };

    BENCHMARK("jac_dCfdcj_contribution + jac_dCrdcj_contribution"){

        auto op =
        [
            =,
            c = make_mdspan(c, extents<1>{nSpecie}),
            dJ = make_mdspan(J, extents<2>{nEqns, nEqns}),
            reactions = make_mdspan(reactions, extents<1>{nReactions})
        ]__device__(){

            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i){
                const auto& reaction = reactions[i];
                reactionParams params{};
                reaction.jac_dCfdcj_contribution(params, dJ);
                reaction.jac_dCrdcj_contribution(params, dJ);
                ret += dJ(4,4);
            }
            return ret;

        };

        return eval(op);
    };

    BENCHMARK("jac_dCdT_contribution"){

        auto op =
        [
            =,
            c = make_mdspan(c, extents<1>{nSpecie}),
            dJ = make_mdspan(J, extents<2>{nEqns, nEqns}),
            reactions = make_mdspan(reactions, extents<1>{nReactions})
        ]__device__(){

            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i){
                const auto& reaction = reactions[i];
                reactionParams params{};
                reaction.jac_dCdT_contribution(params, nSpecie, dJ);
                ret += dJ(4,4);
            }
            return ret;

        };

        return eval(op);
    };

    BENCHMARK("jac_dCdC_contribution"){

        auto op =
        [
            =,
            c = make_mdspan(c, extents<1>{nSpecie}),
            work1 = make_mdspan(work1, extents<1>{nEqns}),
            work2 = make_mdspan(work2, extents<1>{nEqns}),
            dJ = make_mdspan(J, extents<2>{nEqns, nEqns}),
            reactions = make_mdspan(reactions, extents<1>{nReactions})
        ]__device__(){

            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i){
                const auto& reaction = reactions[i];
                reactionParams params{};
                reaction.jac_dCdC_contribution(p, T, params, c, work1, work2, dJ);
                ret += dJ(4,4);
            }
            return ret;

        };

        return eval(op);
    };

    BENCHMARK("computeReactionParams"){

        auto op = [          =,
                   c         = make_mdspan(c, extents<1>{nSpecie}),
                   reactions = make_mdspan(reactions, extents<1>{nReactions})
                  ] __device__() {
            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i) {
                const auto& reaction = reactions[i];
                auto        params = computeReactionParameters(reaction, c, p, T);

                ret += params.Cf + params.Cr
                            + params.dCfdjs[5]
                            + params.dCrdjs[5]
                            + params.dkfdT
                            + params.dkrdT
                            + params.kf
                            + params.kr;
            }
            return ret;
        };

        return eval(op);
    };

    BENCHMARK("kf"){

        auto op = [          =,
                   c         = make_mdspan(c, extents<1>{nSpecie}),
                   reactions = make_mdspan(reactions, extents<1>{nReactions})
                  ] __device__() {
            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i) {
                const auto& reaction = reactions[i];
                gScalar kf = reaction.kf(p, T, c);
                ret += kf;
            }
            return ret;
        };

        return eval(op);
    };

    BENCHMARK("kr"){

        auto op = [          =,
                   c         = make_mdspan(c, extents<1>{nSpecie}),
                   reactions = make_mdspan(reactions, extents<1>{nReactions})
                  ] __device__() {
            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i) {
                const auto& reaction = reactions[i];
                gScalar kf = 0.535654;
                gScalar Kc = 0.6546546;
                gScalar kr = reaction.kr(kf, p, T, Kc, c);
                ret += kr;
            }
            return ret;
        };

        return eval(op);
    };




    BENCHMARK("dkfdT"){

        auto op = [          =,
                   c         = make_mdspan(c, extents<1>{nSpecie}),
                   reactions = make_mdspan(reactions, extents<1>{nReactions})
                  ] __device__() {
            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i) {
                const auto& reaction = reactions[i];
                gScalar dkfdT = reaction.dkfdT(p, T, c);
                ret += dkfdT;
            }
            return ret;
        };

        return eval(op);
    };

    BENCHMARK("dkrdT"){

        auto op = [          =,
                   c         = make_mdspan(c, extents<1>{nSpecie}),
                   reactions = make_mdspan(reactions, extents<1>{nReactions})
                  ] __device__() {
            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i) {
                const auto& reaction = reactions[i];
                gScalar dkfdT = 0.543534;
                gScalar kr = 534534.0;
                gScalar Kc = 0.765756;
                gScalar dkrdT = reaction.dkrdT(p, T, Kc, c, dkfdT, kr);
                ret += dkrdT;
            }
            return ret;
        };

        return eval(op);
    };

    BENCHMARK("Kc"){

        auto op = [          =,
                   reactions = make_mdspan(reactions, extents<1>{nReactions})
                  ] __device__() {
            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i) {
                const auto& reaction = reactions[i];

                ret += reaction.Kc(p, T);
            }
            return ret;
        };

        return eval(op);
    };


    BENCHMARK("Cf + Cr"){

        auto op = [          =,
                   c         = make_mdspan(c, extents<1>{nSpecie}),
                   reactions = make_mdspan(reactions, extents<1>{nReactions})
                  ] __device__() {
            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i) {
                const auto& reaction = reactions[i];

                gScalar Cf = reaction.calcCf(reaction.lhsPowers(c));
                gScalar Cr = reaction.calcCr(reaction.rhsPowers(c));
                ret += Cf + Cr;
            }
            return ret;
        };

        return eval(op);
    };

    BENCHMARK("dCf + dCr"){

        auto op = [          =,
                   c         = make_mdspan(c, extents<1>{nSpecie}),
                   reactions = make_mdspan(reactions, extents<1>{nReactions})
                  ] __device__() {
            gScalar ret = 0.0;
            for (int i = 0; i < nReactions; ++i) {
                const auto& reaction = reactions[i];

                const auto lhsPow = reaction.lhsPowers(c);
                gScalar Cf = 0.342423;
                auto dCf = reaction.calcdCfdcj(lhsPow, Cf, c);

                const auto rhsPow = reaction.rhsPowers(c);
                gScalar Cr = 0.534653;
                auto dCr = reaction.calcdCrdcj(rhsPow, Cr, c);

                ret += dCf[3] + dCr[3];

            }
            return ret;
        };

        return eval(op);
    };



}


TEST_CASE("gpuODESolver"){

    using namespace FoamGpu;

    auto thermos = toDeviceVector(makeGpuThermos(mech));
    auto reactions = toDeviceVector(makeGpuReactions(mech));

    const gLabel nCells = 1;
    const gLabel nSpecie = TestData::speciesCount(mech);
    const gLabel nEqns = TestData::equationCount(mech);


    gpuODESystem system
    (
        nEqns,
        gLabel(reactions.size()),
        make_raw_pointer(thermos.data()),
        make_raw_pointer(reactions.data())
    );


    const device_vector<gScalar> y0 = [=](){
        std::vector<gScalar> vals(nEqns);
        assign_test_condition(vals, mech);
        return device_vector<gScalar>(vals.begin(), vals.end());
    }();

    const device_vector<gScalar> dy = [=](){
        std::vector<gScalar> vals(nEqns);
        fill_random(vals);
        return device_vector<gScalar>(vals.begin(), vals.end());
    }();


    device_vector<gScalar> y(y0.size());


    memoryResource_t memory(nCells, nSpecie);
    auto buffers = toDeviceVector(splitToBuffers(memory));

    const gScalar dx = 1E-7;


    {

        gpuRosenbrock12<gpuODESystem> solver(
            system, TestData::makeGpuODEInputs("Rosenbrock12", mech));

        auto op =
            [        =,
             buffers = make_mdspan(buffers, extents<1>{nCells}),
             y0      = make_mdspan(y0, extents<1>{nEqns}),
             y       = make_mdspan(y, extents<1>{nEqns}),
             dy = make_mdspan(dy, extents<1>{nEqns})
             ] __device__() {
                solver.solve(y0, dy, dx, y, buffers[0]);
                return y[4];
            };

        BENCHMARK("Rosenbrock12"){
            return eval(op);
        };

    }



    {

        gpuRosenbrock23<gpuODESystem> solver(
            system, TestData::makeGpuODEInputs("Rosenbrock23", mech));

        auto op =
            [        =,
             buffers = make_mdspan(buffers, extents<1>{nCells}),
             y0      = make_mdspan(y0, extents<1>{nEqns}),
             y       = make_mdspan(y, extents<1>{nEqns}),
             dy = make_mdspan(dy, extents<1>{nEqns})
             ] __device__() {
                solver.solve(y0, dy, dx, y, buffers[0]);
                return y[4];
            };

        BENCHMARK("Rosenbrock23"){
            return eval(op);
        };

    }

    {

        gpuRosenbrock34<gpuODESystem> solver(
            system, TestData::makeGpuODEInputs("Rosenbrock34", mech));

        auto op =
            [        =,
             buffers = make_mdspan(buffers, extents<1>{nCells}),
             y0      = make_mdspan(y0, extents<1>{nEqns}),
             y       = make_mdspan(y, extents<1>{nEqns}),
             dy = make_mdspan(dy, extents<1>{nEqns})
             ] __device__() {
                solver.solve(y0, dy, dx, y, buffers[0]);
                return y[4];
            };

        BENCHMARK("Rosenbrock34"){
            return eval(op);
        };

    }




}