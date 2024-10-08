#pragma once

#include "gpu_constants.H"
#include <stdexcept>
#include <vector>

namespace TestData {

enum Mechanism { GRI, H2, YAO };

static inline gLabel speciesCount(Mechanism m) {

    switch (m) {
    case Mechanism::GRI: return 53;
    case Mechanism::YAO: return 54;
    case Mechanism::H2: return 10;
    default: throw std::logic_error("Invalid mechanism");
    }
}

static inline gLabel equationCount(Mechanism m) {
    return speciesCount(m) + 2;
}

static inline gScalar pInf(Mechanism m) {

    switch (m) {
    case Mechanism::GRI: return 1.36789e+06;
    case Mechanism::YAO: return 1.36789e+06;
    case Mechanism::H2: return 202650.0;
    default: throw std::logic_error("Invalid mechanism");
    }
}

static inline gScalar TInf(Mechanism m) { return 1000.0; }

template <class T> static inline void assign_gri(T& f) {

    constexpr int CH4_idx = 0;
    constexpr int O2_idx  = 4;
    constexpr int N2_idx  = 47;

    constexpr double CH4_val = 0.2;
    constexpr double O2_val  = 0.5;
    constexpr double N2_val  = 0.3;
    double           T_inf   = TInf(Mechanism::GRI);
    double           p_inf   = pInf(Mechanism::GRI);

    gLabel nSpecie = speciesCount(GRI);

    for (auto& e : f) { e = 0.0; }
    f[CH4_idx] = CH4_val;
    f[O2_idx]  = O2_val;
    f[N2_idx]  = N2_val;

    f[nSpecie]     = T_inf;
    f[nSpecie + 1] = p_inf;
}

template <class T> static inline void assign_h2(T& f) {

    constexpr int H2_idx = 0;
    constexpr int N2_idx = 9;
    constexpr int O2_idx = 2;

    constexpr double H2_val = 0.2;
    constexpr double O2_val = 0.2;
    constexpr double N2_val = 0.6;
    double           T_inf  = TInf(Mechanism::H2);
    double           p_inf  = pInf(Mechanism::H2);

    gLabel nSpecie = speciesCount(H2);

    for (auto& e : f) { e = 0.0; }
    f[H2_idx] = H2_val;
    f[N2_idx] = N2_val;
    f[O2_idx] = O2_val;

    f[nSpecie]     = T_inf;
    f[nSpecie + 1] = p_inf;
}

template <class T> static inline void assign_yao(T& f) {

    constexpr int NC12H26_idx = 44;
    constexpr int N2_idx      = 53;
    constexpr int O2_idx      = 8;

    constexpr double NC12H26_val = 0.2;
    constexpr double O2_val      = 0.2;
    constexpr double N2_val      = 0.6;
    double           T_inf       = TInf(Mechanism::YAO);
    double           p_inf       = pInf(Mechanism::YAO);

    gLabel nSpecie = speciesCount(YAO);

    for (auto& e : f) { e = 0.0; }
    f[NC12H26_idx] = NC12H26_val;
    f[N2_idx]      = N2_val;
    f[O2_idx]      = O2_val;

    f[nSpecie]     = T_inf;
    f[nSpecie + 1] = p_inf;
}

template <class T>
static inline void assign_test_condition(T& f, Mechanism m) {
    if (gLabel(f.size()) != equationCount(m)) {
        throw std::logic_error("Field size not matching mechanism");
    }

    switch (m) {
    case Mechanism::GRI: assign_gri(f); return;
    case Mechanism::YAO: assign_yao(f); return;
    case Mechanism::H2: assign_h2(f); return;
    default: throw std::logic_error("Invalid mechanism");
    }
}

template <class T>
static inline void assign_test_concentration(T& f, Mechanism m) {
    gLabel nSpecie = speciesCount(m);
    if (gLabel(f.size()) != nSpecie) {
        throw std::logic_error(
            "Field size not matching mechanism specie count");
    }

    std::vector<gScalar> temp(nSpecie + 2);
    assign_test_condition(temp, m);

    std::copy(temp.begin(), temp.begin() + nSpecie, f.begin());
}

static inline std::vector<gScalar>
get_concentration_vector(Mechanism m) {

    gLabel               nSpecie = speciesCount(m);
    std::vector<gScalar> c(nSpecie);
    assign_test_concentration(c, m);
    return c;
}

static inline std::vector<gScalar> get_solution_vector(Mechanism m) {

    gLabel               nEqns = equationCount(m);
    std::vector<gScalar> y(nEqns);
    assign_test_condition(y, m);
    return y;
}

} // namespace TestData
