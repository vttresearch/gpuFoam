#include "cpu_reference_results.H"
#include "volFields.H"
#include "thermodynamicConstants.H"
#include "fundamentalConstants.H"
#include "physicoChemicalConstants.H"
#include "scalarMatrices.H"
#include "ODESolver.H"

#include "mock_of_odesystem.H"


namespace TestData{

constantResults constant_results_cpu()
{
    constantResults ret;

    ret.RR = Foam::constant::thermodynamic::RR;
    ret.Pstd = Foam::constant::thermodynamic::Pstd;
    ret.Tstd = Foam::constant::thermodynamic::Tstd;
    ret.NA = Foam::constant::physicoChemical::NA.value();
    ret.k = Foam::constant::physicoChemical::k.value();
    ret.vGreat = Foam::vGreat;
    ret.vSmall = Foam::vSmall;
    ret.small = Foam::small;
    ret.great = Foam::great;
    return ret;
}


perfectGasResult perfect_gas_results_cpu(gScalar p, gScalar T, gScalar Y, gScalar molWeight)
{

    const Foam::perfectGas<Foam::specie> eos
    (
        Foam::specie("temp", Y, molWeight)
    );

    perfectGasResult ret;

    ret.R = eos.R();
    ret.rho = eos.rho(p, T);
    ret.h = eos.h(p, T);
    ret.Cp = eos.Cp(p, T);
    ret.e = eos.e(p, T);
    ret.Cv = eos.Cv(p, T);
    ret.sp = eos.sp(p, T);
    //ret.sv = eos.sv(p, T);
    ret.psi = eos.psi(p, T);
    ret.Z = eos.Z(p, T);
    ret.CpMCv = eos.CpMCv(p, T);
    ret.alphav = eos.alphav(p, T);
    return ret;


}



thermoResults thermo_results_cpu(Mechanism mech)
{
    const Foam::scalar p = TestData::pInf(mech);
    const Foam::scalar T = TestData::TInf(mech);

    auto thermos = TestData::makeCpuThermos(mech);

    const gLabel nThermo = thermos.size();

    thermoResults ret(nThermo);

    for (gLabel i = 0; i < thermos.size(); ++i)
    {
        ret.W[i] = thermos[i].W();
        ret.Y[i] = thermos[i].Y();
        ret.R[i] = thermos[i].R();
        ret.Cp[i] = thermos[i].Cp(p, T);
        ret.ha[i] = thermos[i].ha(p, T);
        ret.hs[i] = thermos[i].hs(p, T);
        ret.hf[i] = thermos[i].hf(    );
        ret.s[i] = thermos[i].s(p, T);
        ret.gStd[i] = thermos[i].gStd(T);
        ret.dCpdT[i] = thermos[i].dCpdT(p, T);
        ret.Cv[i] = thermos[i].Cv(p, T);
        ret.es[i] = thermos[i].es(p, T);
        ret.ea[i] = thermos[i].ea(p, T);
        ret.K[i] = thermos[i].K(p, T);
        ret.Kp[i] = thermos[i].Kp(p, T);
        ret.Kc[i] = thermos[i].Kc(p, T);
        ret.dKcdTbyKc[i] = thermos[i].dKcdTbyKc(p, T);
    }

    return ret;

}



reactionResults reaction_results_cpu(Mechanism mech)
{
    const Foam::ReactionList<FoamThermoType> reactions(
        TestData::makeSpeciesTable(mech),
        TestData::makeCpuThermos(mech),
        TestData::makeReactionDict(mech)
    );


    const gLabel nSpecie = TestData::speciesCount(mech);
    const gLabel nEqns = TestData::equationCount(mech);
    const gLabel nReactions = reactions.size();

    const Foam::scalarField c = [&](){
        Foam::scalarField ret(nSpecie);
        assign_test_concentration(ret, mech);
        return ret;

    }();
    Foam::scalar p = TestData::pInf(mech);
    Foam::scalar T = TestData::TInf(mech);
    Foam::label li = 0;


    Foam::List<gLabel> c2s;
    gLabel csi0 = 0;
    gLabel Tsi = nSpecie;
    Foam::scalarField cTpWork0(nSpecie, 0);
    Foam::scalarField cTpWork1(nSpecie, 0);



    std::vector<gScalar> Thigh(nReactions);
    std::vector<gScalar> Tlow(nReactions);
    std::vector<gScalar> Kc(nReactions);
    std::vector<gScalar> kf(nReactions);
    std::vector<gScalar> kr(nReactions);
    std::vector<gScalar> omega(nReactions);

    std::vector<std::vector<gScalar>> dNdtByV(nReactions);
    std::vector<std::vector<gScalar>> ddNdtByVdcTp(nReactions);

    for (Foam::label i = 0; i < reactions.size(); ++i)
    {
        Thigh[i] = reactions[i].Thigh();
        Tlow[i] = reactions[i].Tlow();
        Kc[i] = reactions[i].Kc(p, T);
        kf[i] = reactions[i].kf(p, T, c, li);
        kr[i] = reactions[i].kr(p, T, c, li);

        //arbitrary
        Foam::scalar omegaf = 0.3;
        Foam::scalar omegar = 0.4;
        omega[i] = reactions[i].omega(p, T, c, li, omegaf, omegar);


        Foam::scalarField dNdtByV_f(c.size(), 0);
        reactions[i].dNdtByV(p, T, c, li, dNdtByV_f, false, Foam::List<Foam::label>{}, 0);
        dNdtByV[i] = std::vector<gScalar>(dNdtByV_f.begin(), dNdtByV_f.end());


        Foam::scalarSquareMatrix ddNdtByVdcTp_f(nEqns, 0);

        dNdtByV_f = 0; //probably not necessary
        reactions[i].ddNdtByVdcTp
        (
            p,
            T,
            c,
            li,
            dNdtByV_f,
            ddNdtByVdcTp_f,
            false,
            c2s,
            csi0,
            Tsi,
            cTpWork0,
            cTpWork1
        );


        ddNdtByVdcTp[i] = std::vector<gScalar>
        (
            ddNdtByVdcTp_f.v(),
            ddNdtByVdcTp_f.v() + ddNdtByVdcTp_f.size()
        );


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
lu_results_cpu(const std::vector<gScalar>& m_vals, const std::vector<gScalar>& s_vals)
{
    gLabel size = std::sqrt(m_vals.size());

    Foam::scalarSquareMatrix matrix(size, 0);
    std::copy(m_vals.begin(), m_vals.end(), matrix.v());
    Foam::List<Foam::label> pivot(size, 0);
    Foam::scalarField source(size);
    std::copy(s_vals.begin(), s_vals.end(), source.begin());

    gLabel sign;
    Foam::LUDecompose(matrix, pivot, sign);
    Foam::LUBacksubstitute(matrix, pivot, source);


    auto m_ret = std::vector<gScalar>(matrix.v(), matrix.v() + size*size);
    auto p_ret = std::vector<gLabel>(pivot.begin(), pivot.end());
    auto s_ret = std::vector<gScalar>(source.begin(), source.end());

    return std::make_tuple(m_ret, p_ret, s_ret);

}


odeSystemResults odesystem_results_cpu(Mechanism mech)
{
    const gLabel nEqns = TestData::equationCount(mech);
    Foam::MockOFSystem system(mech);

    odeSystemResults ret;

    const Foam::scalarField y0 = [=](){
        gLabel nEqns = TestData::equationCount(mech);
        Foam::scalarField y0_t(nEqns);
        assign_test_condition(y0_t, mech);
        return y0_t;
    }();

    const Foam::scalar time = 0.32423;

    const gLabel li = 0;

    {
        Foam::scalarField dy(nEqns, 0.31);
        system.derivatives(0.0, y0, li, dy);
        ret.derivative = std::vector<gScalar>(dy.begin(), dy.end());

    }

    {
        Foam::scalarField dy(nEqns, 0.31);
        Foam::scalarSquareMatrix J(nEqns, 0.1);
        system.jacobian(time, y0, li, dy, J);
        ret.jacobian = std::vector<gScalar>(J.v(), J.v()+J.size());
    }



    return ret;

}

std::vector<gScalar> ode_results_cpu(Mechanism mech, std::string solver_name, gScalar xStart, gScalar xEnd, gScalar dxTry)
{
    Foam::dictionary dict;
    dict.add("solver", solver_name);
    Foam::MockOFSystem system(mech);

    auto ode = Foam::ODESolver::New(system, dict);

    Foam::scalarField y = [&](){

        std::vector<gScalar> v = TestData::get_solution_vector(mech);
        Foam::scalarField ret(v.size());
        std::copy(v.begin(), v.end(), ret.begin());
        return ret;

    }();

    const Foam::label li = 0;


    Foam::scalar dxTry_temp = dxTry;
    ode->solve(xStart, xEnd, y, li, dxTry_temp);


    auto ret = std::vector<gScalar>(y.begin(), y.end());


    remove_negative_zero(ret);



    return ret;

}



}



