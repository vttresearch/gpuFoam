#include "gpu_test_kernels.H"
#include "test_utilities.H"
#include "gpuReaction.H"
#include "gpuODESystem.H"
#include "gpuKernelEvaluator.H"
#include "for_each_index.H"
#include "gpuMemoryResource.H"
#include "singleCellSolver.H"

#include "ludecompose.H"
#include "create_gpu_inputs.H"
#include "makeGpuOdeSolver.H"
#include "host_device_vectors.H"


namespace GpuTestKernels{






using memoryResource_t = FoamGpu::gpuMemoryResource;





template<class T, class R>
__global__ void on_device(T t, R* r)
{
    *r = t();
}



#ifdef __NVIDIA_BACKEND__

    template<class T>
    static inline gScalar eval(T t)
    {

        gScalar *d_result;
        gpuErrorCheck(cudaMalloc(&d_result, sizeof(gScalar)));
        on_device<<<1,1>>>(t, d_result);
        gpuErrorCheck(cudaGetLastError());
        gpuErrorCheck(cudaDeviceSynchronize());
        gScalar h_result;
        gpuErrorCheck(cudaMemcpy(&h_result, d_result, sizeof(gScalar), cudaMemcpyDeviceToHost));
        gpuErrorCheck(cudaDeviceSynchronize());
        gpuErrorCheck(cudaFree(d_result));
        gpuErrorCheck(cudaDeviceSynchronize());
        return h_result;

    }

    //AMD-backend
    #else

    template<class T>
    static inline gScalar eval(T t)
    {

        gScalar *d_result;
        gpuErrorCheck(hipMalloc(&d_result, sizeof(gScalar)));
        hipLaunchKernelGGL
        (
            on_device, dim3(1), dim3(1), 0, 0, t, d_result
        );
        gpuErrorCheck(hipGetLastError());
        gpuErrorCheck(hipDeviceSynchronize());
        gScalar h_result;
        gpuErrorCheck(hipMemcpy(&h_result, d_result, sizeof(gScalar), hipMemcpyDeviceToHost));
        gpuErrorCheck(hipDeviceSynchronize());
        gpuErrorCheck(hipFree(d_result));
        gpuErrorCheck(hipDeviceSynchronize());
        return h_result;

    }

#endif



TestData::constantResults constants(){

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


TestData::perfectGasResult perfect_gas(gScalar p, gScalar T, gScalar Y, gScalar molWeight)
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


TestData::thermoResults thermo_h(TestData::Mechanism mech)
{
    using namespace FoamGpu;

    const auto p = TestData::pInf(mech);
    const auto T = TestData::TInf(mech);

    auto thermos_temp = TestData::makeGpuThermos_h(mech);
    auto thermos = toDeviceVector(thermos_temp);


    const gLabel nThermo = thermos.size();

    TestData::thermoResults ret(nThermo);

    for (size_t i = 0; i < thermos.size(); ++i)
    {
        gpuThermo* thermo = make_raw_pointer(thermos.data()) + i;

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

TestData::thermoResults thermo_e(TestData::Mechanism mech)
{
    using namespace FoamGpu;

    const auto p = TestData::pInf(mech);
    const auto T = TestData::TInf(mech);

    auto thermos_temp = TestData::makeGpuThermos_e(mech);
    auto thermos = toDeviceVector(thermos_temp);


    const gLabel nThermo = thermos.size();

    TestData::thermoResults ret(nThermo);

    for (size_t i = 0; i < thermos.size(); ++i)
    {
        gpuThermo* thermo = make_raw_pointer(thermos.data()) + i;

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


TestData::reactionResults reaction(TestData::Mechanism mech)
{

    using namespace FoamGpu;

    auto reactions_temp = makeGpuReactions(mech);
    auto reactions = toDeviceVector(reactions_temp);

    const gLabel nSpecie = TestData::speciesCount(mech);
    const gLabel nEqns = TestData::equationCount(mech);
    const size_t nReactions = reactions.size();
    const gScalar p = TestData::pInf(mech);
    const gScalar T = TestData::TInf(mech);

    device_vector<gScalar> cc =
    [&](){
        std::vector<gScalar> t(nSpecie);
        fill_linear(t);
        return toDeviceVector(t);
    }();

    std::vector<gScalar> Thigh(nReactions);
    std::vector<gScalar> Tlow(nReactions);
    std::vector<gScalar> Kc(nReactions);
    std::vector<gScalar> kf(nReactions);
    std::vector<gScalar> kr(nReactions);
    std::vector<gScalar> omega(nReactions);
    std::vector<std::vector<gScalar>> dNdtByV(nReactions);
    std::vector<std::vector<gScalar>> ddNdtByVdcTp(nReactions);

    for (size_t i = 0; i < nReactions; ++i){

        gpuReaction* reaction = make_raw_pointer(reactions.data()) + i;

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

    TestData::reactionResults ret;
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
lu(const std::vector<gScalar>& m_vals, const std::vector<gScalar>& s_vals)
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


TestData::odeSystemResults odesystem(TestData::Mechanism mech)
{
    using namespace FoamGpu;

    auto gpu_thermos = toDeviceVector(makeGpuThermos_h(mech));
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


    const device_vector<gScalar> y0_v
    = [&](){

        std::vector<gScalar> t(nEqns);
        fill_linear(t);
        t[nSpecie] = TestData::TInf(mech);
        t[nSpecie + 1] = TestData::pInf(mech);
        return toDeviceVector(t);
    }();



    TestData::odeSystemResults ret;

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


std::vector<gScalar> ode_solve(TestData::Mechanism mech, std::string solver_name, gScalar xStart, gScalar xEnd, gScalar dxTry)
{
    using namespace FoamGpu;

    auto thermos = toDeviceVector(makeGpuThermos_h(mech));
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

    /*
    device_vector<gScalar> y0_v
    = [&](){

        std::vector<gScalar> t(nEqns);
        fill_linear(t);
        t[nSpecie] = TestData::TInf(mech);
        t[nSpecie + 1] = TestData::pInf(mech);
        return toDeviceVector(t);
    }();
    */

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

    eval(f);

    return toStdVector(y0_v);

}

bool test_memory_resource(TestData::Mechanism mech, gLabel nCells){

    using namespace FoamGpu;

    gLabel nSpecie = TestData::speciesCount(mech);


    memoryResource_t memory(nCells, nSpecie);

    auto buffers = toDeviceVector(splitToBuffers(memory));

    auto buffers_span = make_mdspan(buffers, extents<1>(nCells));


    auto op = [nSpecie = nSpecie, buffs = buffers_span] DEVICE (gLabel celli) {

        gpuBuffer& buffer = buffs[celli];
        //gLabel nEqns = nSpecie + 2;

        auto assign_label = [=] DEVICE (mdspan<gLabel, 1> s){
            for (gLabel i = 0; i < s.size(); ++i){
                s[i] = celli;
            }
        };

        auto assign_oned_scalar = [=] DEVICE (mdspan<gScalar, 1> s){
            for (gLabel i = 0; i < s.size(); ++i){
                s[i] = gScalar(celli);
            }
        };

        auto assign_twod_scalar = [=] DEVICE (mdspan<gScalar, 2> s){
            auto ext = s.extents();
            gLabel nj = ext.extent(1);
            gLabel ni = ext.extent(2);
            for (gLabel j = 0; j < nj; ++j){
                for (gLabel i = 0; i < ni; ++i){
                    s(j, i) = gScalar(celli);
                }
            }
            
        };
        


        assign_label(buffer.pivotIndices());

        assign_oned_scalar(buffer.dydx0());
        assign_oned_scalar(buffer.yTemp());
        assign_oned_scalar(buffer.dydx());
        assign_oned_scalar(buffer.dfdx());
        assign_oned_scalar(buffer.k1());
        assign_oned_scalar(buffer.k2());
        assign_oned_scalar(buffer.k3());
        assign_oned_scalar(buffer.k4());
        assign_oned_scalar(buffer.err());
        assign_oned_scalar(buffer.lubuffer());
        assign_oned_scalar(buffer.c());
        auto c = buffer.c();
        c[nSpecie] = gScalar(celli);
        c[nSpecie + 1] = gScalar(celli);

        assign_oned_scalar(buffer.tempField1());
        assign_oned_scalar(buffer.tempField2());


        assign_twod_scalar(buffer.J());


    };


    for_each_index(op, nCells);


    auto h_buffers = toStdVector(buffers);

    gLabel nEqns = nSpecie + 2;


    for (size_t celli = 0; celli < h_buffers.size(); ++celli){

        auto buffer = h_buffers[celli];

        for (auto lPtr : buffer.lData_){
            std::vector<gLabel> hvec(nEqns);

            device_to_host(lPtr, lPtr + nEqns, hvec.begin());

            if (hvec != std::vector<gLabel>(nEqns, celli)){
                return false;
            }
        }

        for (auto sPtr : buffer.sData_){
            std::vector<gScalar> hvec(nEqns);

            device_to_host(sPtr, sPtr + nEqns, hvec.begin());

            if (hvec != std::vector<gScalar>(nEqns, gScalar(celli))){
                return false;
            }
        }

        for (auto sPtr : buffer.twodSData_){
            std::vector<gScalar> hvec(nEqns*nEqns);

            device_to_host(sPtr, sPtr + nEqns*nEqns, hvec.begin());

            if (hvec != std::vector<gScalar>(nEqns*nEqns, gScalar(celli))){
                return false;
            }
        }

    }


    return true;

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

bool test_single_cell_solver(TestData::Mechanism mech, gLabel nCells){

    using namespace FoamGpu;

    auto thermos = toDeviceVector(makeGpuThermos_h(mech));
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

    

    gpuODESolverInputs i = TestData::makeGpuODEInputs("Rosenbrock23", mech);


    auto ode = make_gpuODESolver(system, i);

    

    std::vector<gScalar> Y_h(nCells*nEqns);

    {
        auto s = make_mdspan(Y_h, extents<2>{nCells, nEqns});

        const auto Yi = TestData::get_solution_vector(mech);

        for (gLabel celli = 0; celli < nCells; ++celli) {
            for (gLabel i = 0; i < nEqns; ++i){
                s(celli, i) = Yi[i];
            }

        }
    }

    auto Y_d = toDeviceVector(Y_h);
    auto Y_span = make_mdspan(Y_d, extents<2>{nCells, nEqns});

    memoryResource_t memory(nCells, nSpecie);
    auto buffers = toDeviceVector(splitToBuffers(memory));
    auto buffers_span = make_mdspan(buffers, extents<1>{nCells});

    gScalar deltaT = 1e-5;
    gScalar deltaTChemMax = deltaT/4;
    std::vector<gScalar> deltaTChem_h(nCells, deltaT/10);
    auto deltaTChem_d = toDeviceVector(deltaTChem_h);
    auto deltaTChem_span = make_mdspan(deltaTChem_d, extents<1>{nCells});

    singleCellSolver op(
        deltaT,
        nSpecie,
        deltaTChem_span,
        Y_span,
        buffers_span,
        ode
    );


    for_each_index(op, nCells);



    auto Y_result = toStdVector(Y_d);
    auto s = make_mdspan(Y_result, extents<2>{nCells, nEqns});

    if (s(0, 0) == gScalar(0)){
        return false;
    }

    for (gLabel celli = 1; celli < nCells; ++celli){
        for (gLabel j = 0; j < nEqns; ++j){

            if (s(celli -1, j) != s(celli, j)){
                return false;
            }
        }
    }


    return true;

}



bool test_evaluator(gLabel nCells){

    using namespace FoamGpu;

    const auto m = TestData::GRI;
    auto thermos = TestData::makeGpuThermos_h(m);
    auto reactions = TestData::makeGpuReactions(m);
    gLabel nSpecie = TestData::speciesCount(m);
    gLabel nEqns = TestData::equationCount(m);
    gpuODESolverInputs inputs;
    inputs.name = "Rosenbrock34";
    inputs.absTol = 1E-6;
    inputs.relTol = 1e-1;
    GpuKernelEvaluator evaluator(nCells, nEqns, nSpecie, thermos, reactions, inputs);

    gScalar deltaT = 1e-5;
    gScalar deltaTChemMax = deltaT/4;
    std::vector<gScalar> deltaTChem(nCells, deltaT/10);

    std::vector<gScalar> Yvf(nCells*nEqns);

    auto s = make_mdspan(Yvf, extents<2>{nCells, nEqns});

    auto Yi = TestData::get_solution_vector(m);

    for (gLabel celli = 0; celli < nCells; ++celli) {
        for (gLabel i = 0; i < nEqns; ++i){
            s(celli, i) = Yi[i];
        }

    }

    auto tuple = evaluator.computeYNew
    (
        deltaT,
        deltaTChemMax,
        deltaTChem,
        Yvf
    );

    auto newY = std::get<0>(tuple);
    auto newDeltaTs = std::get<1>(tuple);


    for (gLabel i = 0; i < nEqns; ++i){
        std::cout << newY[i] << std::endl;
    }

    /*
    auto s2 = make_mdspan(newY, extents<2>{nCells, nEqns});

    for (gLabel i = 0; i < nEqns; ++i){
            std::cout << s2(0, i) << std::endl;
    }
    */

    return newY[0] != 0;


}



}



