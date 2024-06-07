#pragma once

#include "mechanisms.H"

#include <utility>
#include <vector>


namespace TestData{

std::vector<std::pair<gScalar, gScalar>> Thigh_Tlow_result(Mechanism mech);
std::vector<gScalar> Kc_result(Mechanism mech);

std::vector<gScalar> omega_result(Mechanism mech);


std::vector<std::vector<gScalar>> dndtbyv_result(Mechanism mech);


std::vector<gScalar> derivative_result(Mechanism mech);



}

