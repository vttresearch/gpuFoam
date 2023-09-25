#pragma once

#include "speciesTable.H"

#include "ReactionList.H"
#include "ReversibleReaction.H"
#include "sensibleEnthalpy.H"

#include "sutherlandTransport.H"
#include "thermo.H"
#include "validThermoType.H"

#include "IOmanip.H"
#include "janafThermo.H"
#include "perfectGas.H"
#include "specie.H"

#include "gpuPerfectGas.H"
#include "gpuReaction.H"
#include "gpuThermo.H"
#include "makeGpuReactions.H"
#include "makeGpuThermo.H"

static const char* GRI_REACTIONS =
#include "gri_reactions.h"
    ;

static const char* GRI_THERMOS =
#include "gri_thermos.h"
    ;

static const char* H2_REACTIONS =
#include "h2_reactions.h"
    ;

static const char* H2_THERMOS =
#include "h2_thermos.h"
    ;

using FoamThermoType = FoamGpu::validThermoType;

namespace TestData {

enum Mechanism { GRI, H2 };

static inline gLabel speciesCount(Mechanism m)
{
    if (m == GRI) {
        return 53;
    }
    return 10;
}

static inline gLabel equationCount(Mechanism m)
{
    return speciesCount(m) + 2;
}


static inline Foam::dictionary makeThermoDict(Mechanism m) {
    std::string t_str = [m]() {
        if (m == GRI) { return std::string(GRI_THERMOS); }
        return std::string(H2_THERMOS);
    }();

    Foam::IStringStream t_temp(t_str);
    Foam::dictionary    thermoDict(t_temp);
    return thermoDict;
}

static inline Foam::dictionary makeReactionDict(Mechanism m) {
    std::string t_str = [m]() {
        if (m == GRI) { return std::string(GRI_REACTIONS); }
        return std::string(H2_REACTIONS);
    }();

    Foam::IStringStream t_temp(t_str);
    Foam::dictionary    reactionDict(t_temp);
    return reactionDict;
}

static inline Foam::PtrList<FoamThermoType> makeCpuThermos(Mechanism m) {
    using namespace Foam;

    dictionary              thermoDict = makeThermoDict(m);
    List<word>              species    = thermoDict.lookup("species");
    PtrList<FoamThermoType> ret;

    for (auto specie : species) {
        ret.append(new FoamThermoType(specie, thermoDict.subDict(specie)));
    }
    return ret;
}

static inline Foam::speciesTable makeSpeciesTable(Mechanism m) {
    using namespace Foam;

    dictionary   thermoDict = makeThermoDict(m);
    List<word>   s_list     = thermoDict.lookup("species");
    speciesTable species(s_list);
    return species;
}

static inline std::vector<FoamGpu::gpuThermo> makeGpuThermos(Mechanism m) {

    auto thermoDict = makeThermoDict(m);
    auto cpuThermos = makeCpuThermos(m);

    std::vector<FoamGpu::gpuThermo> ret;
    for (int i = 0; i < cpuThermos.size(); ++i) {
        auto subDict = thermoDict.subDict(cpuThermos[i].name());

        auto gpuThermo = FoamGpu::makeGpuThermo(cpuThermos[i], subDict);
        ret.push_back(gpuThermo);
    }

    return ret;
}

static inline const auto makeCpuReactions(Mechanism m) {

    auto thermos      = makeCpuThermos(m);
    auto species      = makeSpeciesTable(m);
    auto reactionDict = makeReactionDict(m);

    return Foam::ReactionList<FoamThermoType>(species, thermos, reactionDict);
}

static inline std::vector<FoamGpu::gpuReaction> makeGpuReactions(Mechanism m) {

    auto                   thermos      = makeCpuThermos(m);
    auto                   thermoDict   = makeThermoDict(m);
    auto                   reactionDict = makeReactionDict(m);
    Foam::List<Foam::word> s_list       = thermoDict.lookup("species");
    Foam::speciesTable     species(s_list);

    auto gpu_thermos   = makeGpuThermos(m);
    auto cpu_reactions = makeCpuReactions(m);
    auto ret           = FoamGpu::makeGpuReactions(
        species, reactionDict, gpu_thermos, cpu_reactions);

    return ret;
}

} // namespace TestData