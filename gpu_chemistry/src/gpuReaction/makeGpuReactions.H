#pragma once

#include <map>
#include <vector>

#include "gpuReaction.H"
#include "gpuThermo.H"
#include "makeGpuReactionRate.H"


#include "ReactionList.H"
#include "dictionary.H"

#include "scalar.H"
#include "speciesTable.H"

namespace FoamGpu {

static inline void
set_reaction_thermo(gpuReaction&                      reaction,
                    std::map<std::string, gpuThermo>& thermoDatabase,
                    const std::vector<std::string>&   species) {

    // typename MulticomponentThermo::thermoType rhsThermo
    gpuThermo rhsThermo(reaction.rhs()[0].stoichCoeff *
                        (thermoDatabase[species[reaction.rhs()[0].index]]).W() *
                        (thermoDatabase[species[reaction.rhs()[0].index]]));

    for (gLabel i = 1; i < gLabel(reaction.rhs().size()); ++i) {
        rhsThermo += reaction.rhs()[i].stoichCoeff *
                     (thermoDatabase[species[reaction.rhs()[i].index]]).W() *
                     (thermoDatabase[species[reaction.rhs()[i].index]]);
    }

    gpuThermo lhsThermo(reaction.lhs()[0].stoichCoeff *
                        (thermoDatabase[species[reaction.lhs()[0].index]]).W() *
                        (thermoDatabase[species[reaction.lhs()[0].index]]));

    for (gLabel i = 1; i < gLabel(reaction.lhs().size()); ++i) {
        lhsThermo += reaction.lhs()[i].stoichCoeff *
                     (thermoDatabase[species[reaction.lhs()[i].index]]).W() *
                     (thermoDatabase[species[reaction.lhs()[i].index]]);
    }

    if (std::abs(lhsThermo.Y() - rhsThermo.Y()) > 0.1) {

        FatalErrorInFunction << "Mass imbalance for reaction "
                             << ": " << std::abs(lhsThermo.Y() - rhsThermo.Y())
                             << " kg/kmol" << exit(Foam::FatalError);
    }

    reaction.get_thermo() = (lhsThermo == rhsThermo);

    // gpuThermo::operator=(lhsThermo == rhsThermo);
}

static inline auto make_specie_coeffs(const Foam::List<Foam::specieCoeffs>& l) {

    using return_type = typename gpuReaction::specieCoeffArray;

    return_type ret{};

    if (gLabel(ret.size()) < l.size()) {
        throw std::logic_error("Too long reaction");
    }

    for (gLabel i = 0; i < l.size(); ++i) {
        ret[i].index       = l[i].index;
        ret[i].stoichCoeff = l[i].stoichCoeff;

        gScalar exp = l[i].exponent;

        if (isInteger(exp)){
           ret[i].exponent = gpuSpecieExponent(gLabel(exp));
        }
        else {
            ret[i].exponent = gpuSpecieExponent(gScalar(exp));
        }
        //ret[i].exponent = gpuSpecieExponent(gScalar(exp));

        //ret[i].exponent    = l[i].exponent;
    }
    return ret;
}

template <class ThermoType>
static std::vector<gpuReaction>
makeGpuReactions(const Foam::speciesTable&             species,
                   const Foam::dictionary&               dict,
                   const std::vector<gpuThermo>&         gpu_thermos,
                   const Foam::ReactionList<ThermoType>& cpu_reactions) {

    const Foam::dictionary& reactions(dict.subDict("reactions"));

    std::map<std::string, gpuThermo> thermoDatabase;
    for (gLabel i = 0; i < species.size(); ++i) {
        thermoDatabase[species[i]] = gpu_thermos[i];
    }

    std::vector<std::string> specieNames(species.begin(), species.end());

    std::vector<gpuReaction> ret;
    gLabel                   i = 0;
    forAllConstIter(Foam::dictionary, reactions, iter) {

        auto subDict = reactions.subDict(iter().keyword());
        auto rate    = make_reaction_rate(species, subDict);

        auto cpu_lhs = cpu_reactions[i].lhs();
        auto cpu_rhs = cpu_reactions[i].rhs();

        auto lhs = make_specie_coeffs(cpu_lhs);
        auto rhs = make_specie_coeffs(cpu_rhs);

        // Set general temperature limits from the dictionary
        gScalar Tlow = dict.lookupOrDefault<Foam::scalar>("Tlow", 0);

        gScalar Thigh = dict.lookupOrDefault<Foam::scalar>(
            "Thigh", std::numeric_limits<double>::max() / 10);

        gpuReaction reaction(gpu_thermos[0],
                             lhs,
                             rhs,
                             cpu_lhs.size(),
                             cpu_rhs.size(),
                             rate,
                             Tlow,
                             Thigh);
        set_reaction_thermo(reaction, thermoDatabase, specieNames);

        ret.push_back(reaction);

        ++i;
    }

    return ret;
}








} // namespace FoamGpu