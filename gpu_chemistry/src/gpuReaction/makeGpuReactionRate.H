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
make_thirdBodyEfficiencies(const Foam::speciesTable& species,
                           const Foam::dictionary&   dict) {
    using namespace Foam;

    const gLabel nSpecie = species.size();

    using eff_type = typename gpuThirdBodyEfficiencies::effArray;

    eff_type efficiencies{};

    if (nSpecie > gLabel(efficiencies.size())) {
        std::string str =
            std::string("Only up to ") +
            std::to_string(efficiencies.size()) +
            std::string(
                " species supported in ThirdBodyEfficiencies.");
        throw std::logic_error(str);
    }

    if (dict.found("coeffs")) {
        List<Tuple2<word, scalar>> coeffs(dict.lookup("coeffs"));

        if (nSpecie != coeffs.size()) {
            throw std::logic_error(
                "Number of efficiencies is not equal to the "
                "number of species.");
        }

        forAll(coeffs, i) {
            efficiencies[species[coeffs[i].first()]] =
                coeffs[i].second();
        }
    } else {
        scalar defaultEff = dict.lookup<scalar>("defaultEfficiency");
        for (label i = 0; i < nSpecie; ++i) {
            efficiencies[i] = defaultEff;
        }
    }

    return gpuThirdBodyEfficiencies(nSpecie, efficiencies);
}

static gpu::ThirdBodyArrhenius
make_thirdBodyArrhenius(const Foam::speciesTable& species,
                        const Foam::dictionary&   dict) {
    auto arhenius = make_Arrhenius(dict);
    auto tbes     = make_thirdBodyEfficiencies(species, dict);
    return gpu::ThirdBodyArrhenius(arhenius, tbes);
}

static gpu::ArrheniusLindemannFallOff
make_ArrheniusLindemannFallOff(const Foam::speciesTable& species,
                               const Foam::dictionary&   dict) {
    auto k0   = make_Arrhenius(dict.subDict("k0"));
    auto kInf = make_Arrhenius(dict.subDict("kInf"));
    auto tbes = make_thirdBodyEfficiencies(
        species, dict.subDict("thirdBodyEfficiencies"));
    gpuLindemannFallOffFunction F;
    return gpu::ArrheniusLindemannFallOff(k0, kInf, F, tbes);
}

static gpu::ArrheniusTroeFallOff
make_ArrheniusTroeFallOff(const Foam::speciesTable& species,
                          const Foam::dictionary&   dict) {
    using namespace Foam;

    auto k0   = make_Arrhenius(dict.subDict("k0"));
    auto kInf = make_Arrhenius(dict.subDict("kInf"));

    auto tbes = make_thirdBodyEfficiencies(
        species, dict.subDict("thirdBodyEfficiencies"));
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

    // Allow for typenames where the substring "Reaction" has been
    // subtracted. This is likely some backwards compatibility thing.
    std::string reactionTypeName = [=]() {
        auto pos = temp.find("Reaction");
        if (pos != std::string::npos) {
            return std::string(temp.begin(), temp.begin() + pos);
        }
        return std::string(temp.begin(), temp.end());
    }();

    if (reactionTypeName == "reversibleArrhenius") {
        return gpuReactionRate(make_Arrhenius(dict), false);
    }
    if (reactionTypeName == "irreversibleArrhenius") {
        return gpuReactionRate(make_Arrhenius(dict), true);
    }
    if (reactionTypeName == "reversibleThirdBodyArrhenius") {
        return gpuReactionRate(make_thirdBodyArrhenius(species, dict),
                               false);
    }
    if (reactionTypeName == "reversibleArrheniusLindemannFallOff") {
        return gpuReactionRate(
            make_ArrheniusLindemannFallOff(species, dict), false);
    }
    if (reactionTypeName == "reversibleArrheniusTroeFallOff") {
        return gpuReactionRate(
            make_ArrheniusTroeFallOff(species, dict), false);
    }

    std::string msg = "Invalid reaction type: " + temp;
    throw std::logic_error(msg);
}

} // namespace FoamGpu
