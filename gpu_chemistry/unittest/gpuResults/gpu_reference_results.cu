#include "gpu_reference_results.H"
#include "test_utilities.H"
#include "gpuReaction.H"
#include "gpuODESystem.H"
#include "gpuKernelEvaluator.H"
#include "for_each_index.H"

#include "ludecompose.H"
#include "create_gpu_inputs.H"
#include "makeGpuOdeSolver.H"



namespace TestData{



TestData::constantResults constant_results_gpu(){

    using namespace FoamGpu;

    TestData::constantResults ret;
    ret.RR = eval([] DEVICE (){return gpuRR;});
    ret.Pstd = eval([] DEVICE (){return gpuPstd;});
    ret.Tstd = eval([] DEVICE (){return gpuTstd;});
    ret.NA = eval([] DEVICE (){return gpuNA;});
    ret.k = eval([] DEVICE (){return gpuk;});
    ret.vGreat = eval([] DEVICE (){return gpuVGreat;});
    ret.vSmall = eval([] DEVICE (){return gpuVSmall;});
    ret.small = eval([] DEVICE (){return gpuSmall;});
    ret.great = eval([] DEVICE (){return gpuGreat;});
    return ret;

}

TestData::perfectGasResult perfect_gas_results_gpu(gScalar p, gScalar T, gScalar Y, gScalar molWeight)
{
    using namespace FoamGpu;

    const gpuPerfectGas eos
    (
        Y, molWeight
    );

    TestData::perfectGasResult ret;


    ret.R = eval([=] DEVICE (){return eos.R();});
    ret.rho = eval([=] DEVICE (){return eos.rho(p, T);});
    ret.h = eval([=] DEVICE (){return eos.H(p, T);});
    ret.Cp = eval([=] DEVICE (){return eos.Cp(p, T);});
    ret.e = eval([=] DEVICE (){return eos.E(p, T);});
    ret.Cv = eval([=] DEVICE (){return eos.Cv(p, T);});
    ret.sp = eval([=] DEVICE (){return eos.Sp(p, T);});
    //ret.sv = eval([=] DEVICE (){return eos.Sv(p, T);});
    ret.psi = eval([=] DEVICE (){return eos.psi(p, T);});
    ret.Z = eval([=] DEVICE (){return eos.Z(p, T);});
    ret.CpMCv = eval([=] DEVICE (){return eos.CpMCv(p, T);});
    ret.alphav = eval([=] DEVICE (){return eos.alphav(p, T);});


    return ret;
}

thermoResults thermo_results_gpu(Mechanism mech)
{
    using namespace FoamGpu;

    const auto p = TestData::pInf(mech);
    const auto T = TestData::TInf(mech);

    auto thermos_temp = TestData::makeGpuThermos(mech);
    auto thermos = toDeviceVector(thermos_temp);


    const gLabel nThermo = thermos.size();

    thermoResults ret(nThermo);

    for (size_t i = 0; i < thermos.size(); ++i)
    {
        gpuThermo* thermo = thermos.data() + i;

        ret.W[i] = eval([=] DEVICE (){return thermo->W();});
        ret.Y[i] = eval([=] DEVICE (){return thermo->Y();});
        ret.R[i] = eval([=] DEVICE (){return thermo->R();});
        ret.Cp[i] = eval([=] DEVICE (){return thermo->Cp(p, T);});
        ret.ha[i] = eval([=] DEVICE (){return thermo->Ha(p, T);});
        ret.hs[i] = eval([=] DEVICE (){return thermo->Hs(p, T);});
        ret.hf[i] = eval([=] DEVICE (){return thermo->Hf(    );});
        ret.s[i] = eval([=] DEVICE (){return thermo->S(p, T);});
        ret.gStd[i] = eval([=] DEVICE (){return thermo->Gstd(T);});
        ret.dCpdT[i] = eval([=] DEVICE (){return thermo->dCpdT(p, T);});
        ret.Cv[i] = eval([=] DEVICE (){return thermo->Cv(p, T);});
        ret.es[i] = eval([=] DEVICE (){return thermo->Es(p, T);});
        ret.ea[i] = eval([=] DEVICE (){return thermo->Ea(p, T);});
        ret.K[i] = eval([=] DEVICE (){return thermo->K(p, T);});
        ret.Kp[i] = eval([=] DEVICE (){return thermo->Kp(p, T);});
        ret.Kc[i] = eval([=] DEVICE (){return thermo->Kc(p, T);});
        ret.dKcdTbyKc[i] = eval([=] DEVICE (){return thermo->dKcdTbyKc(p, T);});

    }


    return ret;

}



reactionResults reaction_results_gpu(Mechanism mech)
{

    using namespace FoamGpu;

    auto reactions_temp = makeGpuReactions(mech);
    auto reactions = toDeviceVector(reactions_temp);

    const gLabel nSpecie = TestData::speciesCount(mech);
    const gLabel nEqns = TestData::equationCount(mech);
    const size_t nReactions = reactions.size();
    const gScalar p = TestData::pInf(mech);
    const gScalar T = TestData::TInf(mech);

    const auto cc = toDeviceVector(get_concentration_vector(mech));
    std::vector<gScalar> Thigh(nReactions);
    std::vector<gScalar> Tlow(nReactions);
    std::vector<gScalar> Kc(nReactions);
    std::vector<gScalar> kf(nReactions);
    std::vector<gScalar> kr(nReactions);
    std::vector<gScalar> omega(nReactions);
    std::vector<std::vector<gScalar>> dNdtByV(nReactions);
    std::vector<std::vector<gScalar>> ddNdtByVdcTp(nReactions);

    for (size_t i = 0; i < nReactions; ++i){

        gpuReaction* reaction = reactions.data() + i;

        Thigh[i] = eval([=] DEVICE (){return reaction->Thigh();});
        Tlow[i] = eval([=] DEVICE (){return reaction->Tlow();});

        auto c = make_mdspan(cc, extents<1>{nSpecie});

        Kc[i] = eval([=] DEVICE (){return reaction->Kc(p, T);});
        kf[i] = eval([=] DEVICE (){return reaction->kf(p, T, c);});
        kr[i] = eval([=] DEVICE (){return reaction->kr(p, T, c);});
        omega[i] = eval([=] DEVICE (){return reaction->omega(p, T, c);});



        ///==================dNdtByV==================
        device_vector<gScalar> dNdtByV_i = toDeviceVector(std::vector<gScalar>(nSpecie, 0));
        auto f =
        [
            =,
            res = make_mdspan(dNdtByV_i, extents<1>{nSpecie})
        ] DEVICE
        (){
            reaction->dNdtByV(p, T, c, res);
            return 0;
        };
        eval(f);
        dNdtByV[i] = toStdVector(dNdtByV_i);



        ///==================ddNdtByVdcTp==================
        device_vector<gScalar> ddNdtByVdcTp_i = toDeviceVector(std::vector<gScalar>(nEqns*nEqns, 0));
        device_vector<gScalar> cTpWork0_i(nSpecie);
        auto f2 =
            [
                =,
                res = make_mdspan(ddNdtByVdcTp_i, extents<2>{nEqns, nEqns}),
                cTpWork0 = make_mdspan(cTpWork0_i, extents<1>{nSpecie})
            ] DEVICE
            ()
            {
                auto params = computeReactionParameters(*reaction, c, p, T, cTpWork0);

                reaction->ddNdtByVdcTp
                (
                    p,
                    T,
                    c,
                    res,
                    params

                );

                return 0;
            };

        eval(f2);
        ddNdtByVdcTp[i] = toStdVector(ddNdtByVdcTp_i);


    }

    reactionResults ret;
    ret.Thigh = Thigh;
    ret.Tlow = Tlow;
    ret.Kc = Kc;
    ret.kf = kf;
    ret.kr = kr;
    ret.omega = omega;
    ret.dNdtByV = dNdtByV;
    ret.ddNdtByVdcTp = ddNdtByVdcTp;
    return ret;

}


std::tuple<std::vector<gScalar>, std::vector<gLabel>, std::vector<gScalar>>
lu_results_gpu(const std::vector<gScalar>& m_vals, const std::vector<gScalar>& s_vals)
{

    gLabel size = std::sqrt(m_vals.size());

    device_vector<gScalar> matrix(m_vals.begin(), m_vals.end());
    device_vector<gLabel> pivot = toDeviceVector(std::vector<gLabel>(size, 0));
    //device_vector<gLabel> pivot(size, 0);
    //device_vector<gScalar> v(size, 0);
    device_vector<gScalar> v = toDeviceVector(std::vector<gScalar>(size, 0));
    device_vector<gScalar> source(s_vals.begin(), s_vals.end());

    auto m_span = make_mdspan(matrix, extents<2>{size, size});
    auto p_span = make_mdspan(pivot, extents<1>{size});
    auto v_span = make_mdspan(v, extents<1>{size});

    eval
    (
        [=] DEVICE (){FoamGpu::LUDecompose(m_span, p_span, v_span); return 0;}
    );

    auto s_span = make_mdspan(source, extents<1>{size});

    eval
    (
        [=] DEVICE (){FoamGpu::LUBacksubstitute(m_span, p_span, s_span); return 0;}
    );


    auto m_ret = toStdVector(matrix);
    auto p_ret = toStdVector(pivot);
    auto s_ret = toStdVector(source);

    return std::make_tuple(m_ret, p_ret, s_ret);

}

odeSystemResults odesystem_results_gpu(Mechanism mech)
{
    using namespace FoamGpu;

    auto gpu_thermos = toDeviceVector(makeGpuThermos(mech));
    auto gpu_reactions = toDeviceVector(makeGpuReactions(mech));
    const auto nEqns = TestData::equationCount(mech);
    const auto nSpecie = TestData::speciesCount(mech);

    gpuODESystem gpu
    (
        nEqns,
        gLabel(gpu_reactions.size()),
        make_raw_pointer(gpu_thermos.data()),
        make_raw_pointer(gpu_reactions.data())
    );


    const device_vector<gScalar> y0_v = toDeviceVector(TestData::get_solution_vector(mech));



    odeSystemResults ret;

    {
        memoryResource_t memory(1, nSpecie);
        auto buffers = toDeviceVector(splitToBuffers(memory));

        device_vector<gScalar> dy_v(nEqns);
        auto f =
        [
            =,
            buffers = make_mdspan(buffers, extents<1>{1}),
            y = make_mdspan(y0_v, extents<1>{nEqns}),
            dy = make_mdspan(dy_v, extents<1>{nEqns})
        ] DEVICE
        ()
        {
            gpu.derivatives(y, dy, buffers[0]);
            return 0;
        };
        eval(f);
        ret.derivative = toStdVector(dy_v);

    }

    {

        memoryResource_t memory(1, nSpecie);
        auto buffers = toDeviceVector(splitToBuffers(memory));
        device_vector<gScalar> dy_v(nEqns);
        device_vector<gScalar> J_v(nEqns*nEqns);

        auto f =
        [
            =,
            buffers = make_mdspan(buffers, extents<1>{1}),
            y = make_mdspan(y0_v, extents<1>{nEqns}),
            dy = make_mdspan(dy_v, extents<1>{nEqns}),
            J = make_mdspan(J_v, extents<2>{nEqns, nEqns})
        ] DEVICE
        ()
        {
            gpu.jacobian(y, dy, J, buffers[0]);
            return 0;
        };
        eval(f);

        ret.jacobian = toStdVector(J_v);

    }



    return ret;

}


std::vector<gScalar> ode_results_gpu(Mechanism mech, std::string solver_name, gScalar xStart, gScalar xEnd, gScalar dxTry)
{
    using namespace FoamGpu;

    auto thermos = toDeviceVector(makeGpuThermos(mech));
    auto reactions = toDeviceVector(makeGpuReactions(mech));

    const gLabel nEqns = TestData::equationCount(mech);
    const gLabel nSpecie = TestData::speciesCount(mech);

    gpuODESystem system
    (
        nEqns,
        gLabel(reactions.size()),
        make_raw_pointer(thermos.data()),
        make_raw_pointer(reactions.data())
    );

    gpuODESolverInputs i = TestData::makeGpuODEInputs(solver_name, mech);


    auto solver = make_gpuODESolver(system, i);

    device_vector<gScalar> y0_v = toDeviceVector(TestData::get_solution_vector(mech));

    memoryResource_t memory(1, nSpecie);
    auto buffers = toDeviceVector(splitToBuffers(memory));

    auto f = [
        ode = solver,
        xStart = xStart,
        xEnd = xEnd,
        y = make_mdspan(y0_v, extents<1>{nEqns}),
        dxTry = dxTry,
        buffers = make_mdspan(buffers, extents<1>{1})
    ] DEVICE ()
    {
        gScalar dxTry_temp = dxTry;
        ode.solve(xStart, xEnd, y, dxTry_temp, buffers[0]);
        return dxTry_temp;
    };


    //Round small values to zero to avoid -0 == 0 comparisons
    auto ret = toStdVector(y0_v);


    remove_negative_zero(ret);

    return ret;



}

bool test_for_each_index(){

    using namespace FoamGpu;

    device_vector<gScalar> v1 = toDeviceVector(std::vector<gScalar>(100, 1.0));
    device_vector<gScalar> v2 = toDeviceVector(std::vector<gScalar>(100, 2.0));
    device_vector<gScalar> v3 = toDeviceVector(std::vector<gScalar>(100, 3.0));



    auto op = [v1 = v1.data(), v2 = v2.data(), v3 = v3.data()] DEVICE (gLabel idx){

        v1[idx] = v2[idx] + v3[idx] + 4.0;

    };

    for_each_index(op, 100);

    std::vector<gScalar> correct(100, 2.0 + 3.0 + 4.0);

    return toStdVector(v1) == correct;


}

bool test_evaluator(){

    using namespace FoamGpu;

    const auto m = TestData::GRI;
    auto thermos = TestData::makeGpuThermos(m);
    auto reactions = TestData::makeGpuReactions(m);
    gLabel nSpecie = TestData::speciesCount(m);
    gLabel nEqns = TestData::equationCount(m);
    gLabel nCells = 100;
    gpuODESolverInputs inputs;
    inputs.name = "Rosenbrock34";
    GpuKernelEvaluator evaluator(nCells, nEqns, nSpecie, thermos, reactions, inputs);

    gScalar deltaT = 1e-6;
    gScalar deltaTChemMax = deltaT/4;
    std::vector<gScalar> deltaTChem(nCells, deltaT/5);


    std::vector<gScalar> rho(nCells, 1.0);


    std::vector<gScalar> Yvf(nCells*nEqns);

    auto s = make_mdspan(Yvf, extents<2>{nCells, nEqns});

    auto Yi = TestData::get_solution_vector(m);

    for (gLabel celli = 0; celli < nCells; ++celli) {
        for (gLabel i = 0; i < nEqns; ++i){
            s(celli, i) = Yi[i];
        }

    }





    auto tuple = evaluator.computeRR
    (
        deltaT,
        deltaTChemMax,
        rho,
        deltaTChem,
        Yvf
    );

    auto RR = std::get<0>(tuple);
    auto newDts = std::get<1>(tuple);
    gScalar minDt = std::get<2>(tuple);

    return minDt != gScalar(0);


}



}



