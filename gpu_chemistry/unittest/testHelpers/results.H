#pragma once

#include <vector>
#include "gpu_constants.H"


namespace TestData{

struct constantResults{
    gScalar RR;
    gScalar Pstd;
    gScalar Tstd;
    gScalar NA;
    gScalar k;
    gScalar vGreat;
    gScalar vSmall;
    gScalar small;
    gScalar great;

};

struct perfectGasResult
{
    gScalar R;
    gScalar rho;
    gScalar h;
    gScalar Cp;
    gScalar e;
    gScalar Cv;
    gScalar sp;
    //gScalar sv;
    gScalar psi;
    gScalar Z;
    gScalar CpMCv;
    gScalar alphav;

};

struct thermoResults{

    thermoResults() = default;

    thermoResults(gLabel n) :
    W(n), Y(n), R(n), Cp(n), ha(n), hs(n), hf(n), s(n), gStd(n), dCpdT(n),
    Cv(n), es(n), ea(n), K(n), Kp(n), Kc(n), dKcdTbyKc(n)
    {}

    std::vector<gScalar> W;
    std::vector<gScalar> Y;
    std::vector<gScalar> R;
    std::vector<gScalar> Cp;
    std::vector<gScalar> ha;
    std::vector<gScalar> hs;
    std::vector<gScalar> hf;
    std::vector<gScalar> s;
    std::vector<gScalar> gStd;
    std::vector<gScalar> dCpdT;
    std::vector<gScalar> Cv;
    std::vector<gScalar> es;
    std::vector<gScalar> ea;
    std::vector<gScalar> K;
    std::vector<gScalar> Kp;
    std::vector<gScalar> Kc;
    std::vector<gScalar> dKcdTbyKc;

};

struct reactionResults{

    std::vector<gScalar> Thigh;
    std::vector<gScalar> Tlow;
    std::vector<gScalar> Kc;
    std::vector<gScalar> kf;
    std::vector<gScalar> kr;
    std::vector<gScalar> omega;
    std::vector<std::vector<gScalar>> dNdtByV;
    std::vector<std::vector<gScalar>> ddNdtByVdcTp;
};

struct odeSystemResults{

    std::vector<gScalar> derivative;
    std::vector<gScalar> jacobian;

};


}