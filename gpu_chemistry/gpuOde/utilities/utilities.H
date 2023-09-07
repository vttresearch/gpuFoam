#pragma once

#include <vector>
#include <map>

#include "gpuThermo.H"
#include "gpuReaction.H"
#include "gpuODESystem.H"
#include "gpuRosenbrock34.H"

#include "dictionary.H"
#include "scalar.H"
#include "speciesTable.H"
#include "ReactionList.H"
#include "make_reaction_rate.H"

namespace FoamGpu
{

template<class ThermoType>
static inline gpuThermo make_gpu_thermo(const ThermoType& cputhermo, const Foam::dictionary& thermoDict)
{

    gScalar W = cputhermo.W();
    gScalar Y = cputhermo.Y();

    gScalar Tlow(thermoDict.subDict("thermodynamics").lookup<Foam::scalar>("Tlow"));
    gScalar Thigh(thermoDict.subDict("thermodynamics").lookup<Foam::scalar>("Thigh"));
    gScalar Tcommon(thermoDict.subDict("thermodynamics").lookup<Foam::scalar>("Tcommon"));

    Foam::FixedList<Foam::scalar, 7> cHigh =  thermoDict.subDict("thermodynamics").lookup("highCpCoeffs");
    Foam::FixedList<Foam::scalar, 7> cLow = thermoDict.subDict("thermodynamics").lookup("lowCpCoeffs");

    using coeffArray = typename gpuThermo::coeffArray;

    coeffArray highCpCoeffs{};
    coeffArray lowCpCoeffs{};

    for (size_t i = 0; i < gpuThermo::nCoeffs_; ++i)
    {
        highCpCoeffs[i] = cputhermo.R() * cHigh[i];
        lowCpCoeffs[i]  = cputhermo.R() * cLow[i];
    }

    return gpuThermo
    (
        Y, W, Tlow, Thigh, Tcommon,
        highCpCoeffs,
        lowCpCoeffs
    );

}

template<class ThermoType>
static inline std::vector<gpuThermo> make_gpu_thermos
(
    const Foam::PtrList<ThermoType>& cpuThermos,
    const Foam:: dictionary& physicalProperties
)
{
    std::vector<gpuThermo> gpuThermos;

    for (const auto& t : cpuThermos)
    {
        const auto& specieName = t.name();
        const auto subDict = physicalProperties.subDict(specieName);
        gpuThermo gpu = make_gpu_thermo(t, subDict);
        //gpuThermo gpu(t.Y(), t.W(), physicalProperties.subDict(specieName));
        gpuThermos.push_back(gpu);
    }


    testThermo(cpuThermos, gpuThermos);
    return gpuThermos;
}





static inline void
set_reaction_thermo
(
    gpuReaction& reaction,
    std::map<std::string, gpuThermo>& thermoDatabase,
    const std::vector<std::string>& species
)
{

    //typename MulticomponentThermo::thermoType rhsThermo
    gpuThermo rhsThermo
    (
        reaction.rhs()[0].stoichCoeff
        *(thermoDatabase[species[reaction.rhs()[0].index]]).W()
        *(thermoDatabase[species[reaction.rhs()[0].index]])
    );

    for (gLabel i=1; i<gLabel(reaction.rhs().size()); ++i)
    {
        rhsThermo +=
            reaction.rhs()[i].stoichCoeff
            *(thermoDatabase[species[reaction.rhs()[i].index]]).W()
            *(thermoDatabase[species[reaction.rhs()[i].index]]);
    }

    gpuThermo lhsThermo
    (
        reaction.lhs()[0].stoichCoeff
        *(thermoDatabase[species[reaction.lhs()[0].index]]).W()
        *(thermoDatabase[species[reaction.lhs()[0].index]])
    );

    for (gLabel i=1; i<gLabel(reaction.lhs().size()); ++i)
    {
        lhsThermo +=
            reaction.lhs()[i].stoichCoeff
            *(thermoDatabase[species[reaction.lhs()[i].index]]).W()
            *(thermoDatabase[species[reaction.lhs()[i].index]]);
    }

    if (std::abs(lhsThermo.Y() - rhsThermo.Y()) > 0.1)
    {
        
        FatalErrorInFunction
            << "Mass imbalance for reaction " << ": "
            << std::abs(lhsThermo.Y() - rhsThermo.Y()) << " kg/kmol"
            << exit(Foam::FatalError);
            
    }

    reaction.get_thermo() = (lhsThermo == rhsThermo);

    //gpuThermo::operator=(lhsThermo == rhsThermo);


}


static inline auto
make_specie_coeffs(const Foam::List<Foam::specieCoeffs>& l)
{

    using return_type
        = typename gpuReaction::specieCoeffArray;

    return_type ret{};

    if (gLabel(ret.size()) < l.size())
    {
        throw std::logic_error("Too long reaction");
    }

    for (gLabel i = 0; i < l.size(); ++i)
    {
        ret[i].index = l[i].index;
        ret[i].stoichCoeff = l[i].stoichCoeff;
        ret[i].exponent = l[i].exponent;
    }
    return ret;
}



template<class ThermoType>
static std::vector<gpuReaction> make_gpu_reactions
(
    const Foam::speciesTable& species,
    const Foam::dictionary& dict,
    const std::vector<gpuThermo>& gpu_thermos,
    const Foam::ReactionList<ThermoType>& cpu_reactions
)
{

    const Foam::dictionary& reactions(dict.subDict("reactions"));

    std::map<std::string, gpuThermo> thermoDatabase;
    for(gLabel i = 0; i < species.size(); ++i)
    {
        thermoDatabase[species[i]] = gpu_thermos[i];
    }



    std::vector<std::string> specieNames(species.begin(), species.end());


    std::vector<gpuReaction> ret;
    gLabel i = 0;
    forAllConstIter(Foam::dictionary, reactions, iter)
    {

        auto subDict = reactions.subDict(iter().keyword());
        auto rate = make_reaction_rate(species, subDict);

        auto cpu_lhs = cpu_reactions[i].lhs();
        auto cpu_rhs = cpu_reactions[i].rhs();



        auto lhs = make_specie_coeffs(cpu_lhs);
        auto rhs = make_specie_coeffs(cpu_rhs);


        // Set general temperature limits from the dictionary
        gScalar Tlow = dict.lookupOrDefault<Foam::scalar>("Tlow", 0);

        gScalar Thigh =dict.lookupOrDefault<Foam::scalar>("Thigh", std::numeric_limits<double>::max() / 10);

        gpuReaction reaction
        (
            gpu_thermos[0],
            lhs,
            rhs,
            cpu_lhs.size(),
            cpu_rhs.size(),
            rate,
            Tlow,
            Thigh
        );
        set_reaction_thermo(reaction, thermoDatabase, specieNames);

        ret.push_back(reaction);

        ++i;
    }


    return ret;

}

static inline gpuRosenbrock34Inputs read_Rosenbrock34_inputs
(
    const Foam::dictionary& dict
)
{
    gpuRosenbrock34Inputs i;

    //gScalar small = std::numeric_limits<gScalar>::epsilon();

    i.absTol = dict.lookupOrDefault<Foam::scalar>("absTol", gpuSmall);
    //i.absTol = dict.lookupOrDefault<gScalar>("absTol", small);
    i.relTol = dict.lookupOrDefault<Foam::scalar>("relTol", 1e-4);
    i.maxSteps = dict.lookupOrDefault<Foam::label>("maxSteps", 10000);
    i.safeScale = dict.lookupOrDefault<Foam::scalar>("safeScale", 0.9);
    i.alphaInc = dict.lookupOrDefault<Foam::scalar>("alphaIncrease", 0.2);
    i.alphaDec = dict.lookupOrDefault<Foam::scalar>("alphaDecrease", 0.25);
    i.minScale = dict.lookupOrDefault<Foam::scalar>("minScale", 0.2);
    i.maxScale = dict.lookupOrDefault<Foam::scalar>("maxScale", 10);

    return i;
}

static inline gpuRosenbrock34<gpuODESystem> make_Rosenbrock34
(
    const gpuODESystem& ode,
    const Foam::dictionary& dict
)
{

    return gpuRosenbrock34<gpuODESystem>
    (
        ode,
        read_Rosenbrock34_inputs(dict)
    );

}



}