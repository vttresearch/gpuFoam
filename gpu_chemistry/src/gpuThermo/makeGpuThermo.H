#pragma once

#include <map>
#include <vector>

#include "host_device_vectors.H"
#include "gpuThermo.H"



#include "dictionary.H"

#include "scalar.H"
#include "speciesTable.H"

namespace FoamGpu {

template <class ThermoType>
static inline gpuThermo make_gpu_thermo(const ThermoType&       cputhermo,
                                        const Foam::dictionary& thermoDict) {

    gScalar W = cputhermo.W();
    gScalar Y = cputhermo.Y();

    gScalar Tlow(
        thermoDict.subDict("thermodynamics").lookup<Foam::scalar>("Tlow"));
    gScalar Thigh(
        thermoDict.subDict("thermodynamics").lookup<Foam::scalar>("Thigh"));
    gScalar Tcommon(
        thermoDict.subDict("thermodynamics").lookup<Foam::scalar>("Tcommon"));

    Foam::FixedList<Foam::scalar, 7> cHigh =
        thermoDict.subDict("thermodynamics").lookup("highCpCoeffs");
    Foam::FixedList<Foam::scalar, 7> cLow =
        thermoDict.subDict("thermodynamics").lookup("lowCpCoeffs");

    using coeffArray = typename gpuThermo::coeffArray;

    coeffArray highCpCoeffs{};
    coeffArray lowCpCoeffs{};

    for (size_t i = 0; i < gpuThermo::nCoeffs_; ++i) {
        highCpCoeffs[i] = cputhermo.R() * cHigh[i];
        lowCpCoeffs[i]  = cputhermo.R() * cLow[i];
    }

    return gpuThermo(Y, W, Tlow, Thigh, Tcommon, highCpCoeffs, lowCpCoeffs);
}

template <class ThermoType>
static inline std::vector<gpuThermo>
make_gpu_thermos(const Foam::PtrList<ThermoType>& cpuThermos,
                 const Foam::dictionary&          physicalProperties) {
    std::vector<gpuThermo> gpuThermos;

    for (const auto& t : cpuThermos) {
        const auto& specieName = t.name();
        const auto  subDict    = physicalProperties.subDict(specieName);
        gpuThermo   gpu        = make_gpu_thermo(t, subDict);
        // gpuThermo gpu(t.Y(), t.W(), physicalProperties.subDict(specieName));
        gpuThermos.push_back(gpu);
    }

    return gpuThermos;
}


}