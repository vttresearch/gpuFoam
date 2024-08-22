#include "create_foam_inputs.H"

namespace TestData {

static const char* GRI_REACTIONS =
#include "gri_reactions.h"
    ;

static const char* GRI_THERMOS =
#include "gri_thermos.h"
    ;

static const char* YAO_REACTIONS =
#include "yao_reactions.h"
    ;

static const char* YAO_THERMOS =
#include "yao_thermos.h"
    ;

static const char* H2_REACTIONS =
#include "h2_reactions.h"
    ;

static const char* H2_THERMOS =
#include "h2_thermos.h"
    ;

Foam::dictionary makeThermoDict(TestData::Mechanism m) {
    std::string t_str = [m]() {
        if (m == TestData::GRI) { return std::string(GRI_THERMOS); }
        if (m == TestData::YAO) { return std::string(YAO_THERMOS); }
        return std::string(H2_THERMOS);
    }();

    Foam::IStringStream t_temp(t_str);
    Foam::dictionary    thermoDict(t_temp);
    return thermoDict;
}

Foam::speciesTable makeSpeciesTable(TestData::Mechanism m) {
    using namespace Foam;

    dictionary   thermoDict = makeThermoDict(m);
    List<word>   s_list     = thermoDict.lookup("species");
    speciesTable species(s_list);
    return species;
}

Foam::PtrList<FoamGpu::testThermoType1>
makeCpuThermos_h(TestData::Mechanism m) {
    using namespace Foam;

    dictionary thermoDict = makeThermoDict(m);
    List<word> species    = thermoDict.lookup("species");
    PtrList<FoamGpu::testThermoType1> ret;

    for (auto specie : species) {
        ret.append(new FoamGpu::testThermoType1(
            specie, thermoDict.subDict(specie)));
    }
    return ret;
}
Foam::PtrList<FoamGpu::testThermoType3>
makeCpuThermos_e(TestData::Mechanism m) {
    using namespace Foam;

    dictionary thermoDict = makeThermoDict(m);
    List<word> species    = thermoDict.lookup("species");
    PtrList<FoamGpu::testThermoType3> ret;

    for (auto specie : species) {
        ret.append(new FoamGpu::testThermoType1(
            specie, thermoDict.subDict(specie)));
    }
    return ret;
}
Foam::dictionary makeReactionDict(TestData::Mechanism m) {
    std::string t_str = [&]() {
        if (m == TestData::GRI) { return std::string(GRI_REACTIONS); }
        if (m == TestData::YAO) { return std::string(YAO_REACTIONS); }
        return std::string(H2_REACTIONS);
    }();

    Foam::IStringStream t_temp(t_str);
    Foam::dictionary    reactionDict(t_temp);
    return reactionDict;
}

Foam::ReactionList<FoamGpu::testThermoType1>
makeCpuReactions(TestData::Mechanism m) {

    auto thermos      = makeCpuThermos_h(m);
    auto species      = makeSpeciesTable(m);
    auto reactionDict = makeReactionDict(m);

    return Foam::ReactionList<FoamGpu::testThermoType1>(
        species, thermos, reactionDict);
}

} // namespace TestData