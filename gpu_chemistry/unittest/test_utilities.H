#pragma once

#include <vector>
#include <array>
#include "error_handling.H"

#include "FixedList.H"
#include "IOmanip.H"
#include "janafThermo.H"
#include "specie.H"
#include "perfectGas.H"
#include "gpuThermo.H"
#include "gpuReaction.H"
#include "gpuPerfectGas.H"
#include "thermo.H"
#include "sensibleEnthalpy.H"
#include "speciesTable.H"
#include "ReactionList.H"
#include "sutherlandTransport.H"
#include "ArrheniusReactionRate.H"
#include "ReversibleReaction.H"
#include "utilities.H"

constexpr int CH4_idx = 0;
constexpr int O2_idx = 4;
constexpr int N2_idx = 47;

constexpr double CH4_val = 0.2;
constexpr double O2_val = 0.5;
constexpr double N2_val = 0.3;
constexpr double T_inf = 1000.0;
constexpr double p_inf = 1.36789e+06;


constexpr double errorTol = 1E-10;



#ifdef __NVIDIA_COMPILER__

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

template<class T>
using host_vector = thrust::host_vector<T>;

template<class T>
using device_vector = thrust::device_vector<T>;


template<class T, class R>
__global__ void on_device(T t, R* r)
{
    *r = t();
}


template<class T>
static inline Foam::scalar eval(T t)
{
    using namespace Foam;

    scalar *d_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(scalar)));
    on_device<<<1,1>>>(t, d_result);
    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
    scalar h_result;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(scalar), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaFree(d_result));
    cudaDeviceSynchronize();
    return h_result;

}

#else

template<class T>
using host_vector = std::vector<T>;

template<class T>
using device_vector = std::vector<T>;

template<class T>
static inline Foam::scalar eval(T t)
{
    return t();
}
#endif



template<class T, unsigned N>
static inline auto toArray(Foam::FixedList<T, N> a)
{

    std::array<T, N> ret{};
    for (size_t i = 0; i < N; ++i)
    {
        ret[i] = a[i];
    }
    return ret;
}



static inline auto toArray(typename FoamGpu::gpuThermo::coeffArray a)
{
    std::array<double, 7> ret{};
    for (size_t i = 0; i < 7; ++i)
    {
        ret[i] = a[i];
    }
    return ret;
}


template<class T>
static inline auto to_std_vec(const T& c)
{
    using value_type = typename T::value_type;
    host_vector<value_type> temp(c.begin(), c.end());
    return std::vector<value_type>(temp.begin(), temp.end());
}

template<class T>
static inline auto to_device_vec(const T& c)
{
    using value_type = typename T::value_type;
    host_vector<value_type> temp(c.begin(), c.end());
    return device_vector<value_type>(temp.begin(), temp.end());
}



static const char *REACTIONS =
#include "gri_reactions.h"
;

static const char *THERMOS =
#include "gri_thermos.h"
;

using cpuThermo =
        Foam::sutherlandTransport
        <
            Foam::species::thermo
            <
                Foam::janafThermo
                <
                    Foam::perfectGas
                    <
                        Foam::specie
                    >
                >,
                Foam::sensibleEnthalpy
            >
        >;

namespace FoamGpu
{






static inline Foam::dictionary get_thermo_dict()
{

    std::string t_str(THERMOS);
    Foam::IStringStream t_temp(t_str);

    Foam::dictionary thermoDict(t_temp);
    return thermoDict;
}

static inline Foam::dictionary get_reaction_dict()
{
    std::string t_str(REACTIONS);
    Foam::IStringStream t_temp(t_str);

    Foam::dictionary reactionDict(t_temp);
    return reactionDict;

}

static inline Foam::PtrList<cpuThermo> make_cpu_thermos()
{
    using namespace Foam;

    dictionary thermoDict = get_thermo_dict();
    List<word> species = thermoDict.lookup("species");
    PtrList<cpuThermo> ret;

    for (auto specie : species)
    {
        ret.append
        (
            new cpuThermo(specie, thermoDict.subDict(specie))
        );
    }
    return ret;
}

static inline Foam::speciesTable make_species_table(){
    using namespace Foam;

    dictionary thermoDict = get_thermo_dict();
    List<word> s_list = thermoDict.lookup("species");
    speciesTable species(s_list);
    return species;
}



static inline std::vector<gpuThermo> make_gpu_thermos()
{
    using namespace Foam;

    dictionary thermoDict = get_thermo_dict();
    auto cpuThermos = make_cpu_thermos();

    std::vector<gpuThermo> ret;
    for (label i = 0; i < cpuThermos.size(); ++i)
    {
        auto subDict = thermoDict.subDict(cpuThermos[i].name());

        auto gpuThermo = make_gpu_thermo
        (
            cpuThermos[i],
            subDict
        );
        ret.push_back(gpuThermo);
    }

    return ret;
}

static inline const auto make_cpu_reactions()
{
    using namespace Foam;

    auto thermos = make_cpu_thermos();
    auto species = make_species_table();
    dictionary reactionDict = get_reaction_dict();

    return ReactionList<cpuThermo>(species, thermos, reactionDict);

}

static inline std::vector<gpuReaction> make_gpu_reactions()
{
    using namespace Foam;

    auto thermos = make_cpu_thermos();
    dictionary thermoDict = get_thermo_dict();
    List<word> s_list = thermoDict.lookup("species");
    speciesTable species(s_list);
    dictionary reactionDict = get_reaction_dict();

    auto gpu_thermos = make_gpu_thermos();
    auto cpu_reactions = make_cpu_reactions();
    auto ret =
        make_gpu_reactions
        (
            species,
            reactionDict,
            gpu_thermos,
            cpu_reactions
        );

    return ret;
}

static inline double random_number(double LO, double HI){
    double r = LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI-LO)));
    return r;
}

template<class T>
static inline void fill_random(T& v, double LO = 0.0, double HI = 1.0)
{
    for (auto& e : v)
    {
        e = random_number(LO, HI);
    }
}

static inline void assign_test_condition(Foam::scalarField& f)
{
    gLabel nSpecie = 53;
    if (f.size() != nSpecie + 2){
        throw std::logic_error("Field size not gri");
    }

    for (auto& e : f)
    {
        e = 0.0;
    }
    f[CH4_idx] = CH4_val;
    f[O2_idx] = O2_val;
    f[N2_idx] = N2_val;

    f[nSpecie] = T_inf;
    f[nSpecie+1] = p_inf;
}



} //Foam








