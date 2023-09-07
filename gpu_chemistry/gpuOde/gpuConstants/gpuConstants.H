#pragma once

#include "cuda_host_dev.H"

#include <limits>




using gLabel = int;
using gScalar = double;


//Check OpenFOAM/OpenFOAM-gpu/etc/controlDict SI

// Avagadro number
#define gpuNA double(6.0221417930e+23)

//Boltzman constant
#define gpuk double(1.38065e-23)

#define gpuR double(gpuNA*gpuk)

#define gpuRR double(1.0E3*gpuR)

//Standard pressure
#define gpuPstd double(1e5)

//Standard temperature
#define gpuTstd double(298.15)

#define gpuLabelMax gLabel(std::numeric_limits<gLabel>::max()/10)

#define gpuVGreat double(std::numeric_limits<double>::max()/10.0)

#define gpuVSmall double(std::numeric_limits<double>::min())

#define gpuSmall double(std::numeric_limits<double>::epsilon())

#define gpuGreat double(1.0/gpuSmall)

#define gpuLogTen double(2.302585092994046)

#define SPECIE_MAX 55

#define MAX_REACTION_WIDTH 10

/*
namespace gpuConstants
{




    namespace physical
    {
        //Check OpenFOAM/OpenFOAM-gpu/etc/controlDict SI

        // Avagadro number
        constexpr double NA = 6.0221417930e+23;

        //Boltzman constant
        constexpr double k = 1.38065e-23;

        constexpr double R = NA * k;

        constexpr double RR = 1E3* R;

        //Standard pressure
        constexpr double Pstd = 1e5;

        //Standard temperature
        constexpr double Tstd = 298.15;
    }


    constexpr int gpuLabelMax =
        std::numeric_limits<gLabel>::max()/10;

    constexpr double gpuVGreat
         = std::numeric_limits<double>::max()/10;

    constexpr double gpuVSmall
        = std::numeric_limits<double>::min();

    constexpr double gpuSmall
        = std::numeric_limits<double>::epsilon();

    constexpr double gpuGreat
        = 1.0 / gpuSmall;


    constexpr double gpuLogTen = 2.302585092994046;

    //Maximum number of species in the mechanism
    constexpr int SPECIE_MAX = 100;

    //Maximum number of species involved in one side of a reaction
    constexpr int MAX_REACTION_WIDTH = 10;





}
*/