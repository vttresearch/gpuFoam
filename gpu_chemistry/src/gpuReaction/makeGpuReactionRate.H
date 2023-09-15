#pragma once

#include "gpuReactionRate.H"
#include "scalarField.H"
#include "speciesTable.H"

namespace FoamGpu {

static gpu::Arrhenius make_Arrhenius(const Foam::dictionary& dict) {
    gScalar A    = dict.lookup<gScalar>("A");
    gScalar beta = dict.lookup<gScalar>("beta");
    gScalar Ta   = dict.lookup<gScalar>("Ta");
    return gpu::Arrhenius(A, beta, Ta);
}

static gpuThirdBodyEfficiencies
make_thirdBodyEfficiencies(gLabel nSpecie, const Foam::dictionary& dict) {
    using namespace Foam;

    using eff_type = typename gpuThirdBodyEfficiencies::effArray;

    eff_type efficiencies{};

    if (dict.found("coeffs")) {
        List<Tuple2<word, scalar>> coeffs(dict.lookup("coeffs"));

        if (nSpecie != coeffs.size()) {
            throw std::logic_error("Number of efficiencies is not equal to the "
                                   "number of species.");
        }

        forAll(coeffs, i) { efficiencies[i] = coeffs[i].second(); }
    } else {
        scalar defaultEff = dict.lookup<scalar>("defaultEfficiency");
        for (label i = 0; i < nSpecie; ++i) { efficiencies[i] = defaultEff; }
    }

    return gpuThirdBodyEfficiencies(nSpecie, efficiencies);
}

static gpu::ThirdBodyArrhenius
make_thirdBodyArrhenius(gLabel nSpecie, const Foam::dictionary& dict) {
    auto arhenius = make_Arrhenius(dict);
    auto tbes     = make_thirdBodyEfficiencies(nSpecie, dict);
    return gpu::ThirdBodyArrhenius(arhenius, tbes);
}

static gpu::ArrheniusLindemannFallOff
make_ArrheniusLindemannFallOff(gLabel nSpecie, const Foam::dictionary& dict) {
    auto k0   = make_Arrhenius(dict.subDict("k0"));
    auto kInf = make_Arrhenius(dict.subDict("kInf"));
    auto tbes = make_thirdBodyEfficiencies(
        nSpecie, dict.subDict("thirdBodyEfficiencies"));
    gpuLindemannFallOffFunction F;
    return gpu::ArrheniusLindemannFallOff(k0, kInf, F, tbes);
}

static gpu::ArrheniusTroeFallOff
make_ArrheniusTroeFallOff(gLabel nSpecie, const Foam::dictionary& dict) {
    using namespace Foam;

    auto k0   = make_Arrhenius(dict.subDict("k0"));
    auto kInf = make_Arrhenius(dict.subDict("kInf"));

    auto tbes = make_thirdBodyEfficiencies(
        nSpecie, dict.subDict("thirdBodyEfficiencies"));
    auto                   subDict = dict.subDict("F");
    scalar                 alpha   = subDict.lookup<scalar>("alpha");
    scalar                 Tsss    = subDict.lookup<scalar>("Tsss");
    scalar                 Ts      = subDict.lookup<scalar>("Ts");
    scalar                 Tss     = subDict.lookup<scalar>("Tss");
    gpuTroeFallOffFunction F(alpha, Tsss, Ts, Tss);
    return gpu::ArrheniusTroeFallOff(k0, kInf, F, tbes);
}

static inline gpuReactionRate
make_reaction_rate(const Foam::speciesTable& species,
                   const Foam::dictionary&   dict) {
    using namespace gpu;

    const Foam::word& temp = dict.lookup("type");

    //Allow for typenames where the substring "Reaction" has been subtracted.
    //This is likely some backwards compatibility thing.
    std::string reactionTypeName = [=](){}
    {
        auto pos = temp.find("Reaction");
        if (pos != std::string::npos )
        {
            return std::string(temp.begin(), temp.begin()+pos);
        }
        return temp;
    }();
    

    if (reactionTypeName == "reversibleArrhenius") {
        return gpuReactionRate(make_Arrhenius(dict), false);
    }
    if (reactionTypeName == "irreversibleArrhenius") {
        return gpuReactionRate(make_Arrhenius(dict), true);
    }
    if (reactionTypeName == "reversibleThirdBodyArrhenius") {
        // auto rate = make_thirdBodyArrhenius(species.size(), dict);
        return gpuReactionRate(make_thirdBodyArrhenius(species.size(), dict),
                               false);
    }
    if (reactionTypeName == "reversibleArrheniusLindemannFallOff") {
        // auto rate = make_ArrheniusLindemannFallOff(species.size(), dict);
        return gpuReactionRate(
            make_ArrheniusLindemannFallOff(species.size(), dict), false);
    }
    if (reactionTypeName == "reversibleArrheniusTroeFallOff") {
        // auto rate = make_ArrheniusTroeFallOff(species.size(), dict);
        return gpuReactionRate(make_ArrheniusTroeFallOff(species.size(), dict),
                               false);
    }

    std::string msg = "Invalid reaction type: " + temp;
    throw std::logic_error(msg);
}

} // namespace FoamGpu
