/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2021 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::gpuTroeFallOffFunction

Description
    The Troe fall-off function
scalar
SourceFiles
    gpuTroeFallOffFunctionI.H

\*---------------------------------------------------------------------------*/

#ifndef gpuTroeFallOffFunction_H
#define gpuTroeFallOffFunction_H

#include "gpuConstants.H"
#include "cuda_math.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace FoamGpu
{



/*---------------------------------------------------------------------------*\
                     Class gpuTroeFallOffFunction Declaration
\*---------------------------------------------------------------------------*/

class gpuTroeFallOffFunction
{
    // Private Data

        gScalar alpha_;
        gScalar Tsss_, Ts_, Tss_;


public:

    // Constructors

        gpuTroeFallOffFunction() = default;
        
        
        //- Construct from components
        inline CUDA_HOSTDEV gpuTroeFallOffFunction
        (
            const gScalar alpha,
            const gScalar Tsss,
            const gScalar Ts,
            const gScalar Tss
        )
        :
            alpha_(alpha),
            Tsss_(Tsss),
            Ts_(Ts),
            Tss_(Tss)
        {}



    // Member Functions


        inline CUDA_HOSTDEV gScalar operator()
        (
            const gScalar T,
            const gScalar Pr
        ) const
        {

            const gScalar logPr = log10(max(Pr, gpuSmall));

            const gScalar Fcent =
                (1.0 - alpha_)*exp(-T/Tsss_)
            + alpha_*exp(-T/Ts_)
            + exp(-Tss_/T);

            const gScalar logFcent = log10(max(Fcent, gpuSmall));

            const gScalar c = -0.4 - 0.67*logFcent;
            constexpr gScalar d = 0.14;
            const gScalar n = 0.75 - 1.27*logFcent;

            const gScalar x1 = n - d*(logPr + c);
            const gScalar x2 = (logPr + c)/x1;
            const gScalar x3 = 1.0 + x2*x2;
            const gScalar x4 = logFcent/x3;

            return pow(10.0, x4);

        }

        inline CUDA_HOSTDEV gScalar ddT
        (
            const gScalar T,
            const gScalar Pr,
            const gScalar F
        ) const
        {


            const gScalar logPr = log10(max(Pr, gpuSmall));

            const gScalar Fcent =
                (1.0 - alpha_)*exp(-T/Tsss_)
            + alpha_*exp(-T/Ts_)
            + exp(-Tss_/T);
            const gScalar dFcentdT =
            - (1.0 - alpha_)/Tsss_*exp(-T/Tsss_)
            - alpha_/Ts_*exp(-T/Ts_)
            + Tss_/(T*T)*exp(-Tss_/T);

            const gScalar logFcent = log10(max(Fcent, gpuSmall));
            const gScalar dlogFcentdT = Fcent >= gpuSmall ? dFcentdT/Fcent/gpuLogTen : 0.0;

            const gScalar c = -0.4 - 0.67*logFcent;
            const gScalar dcdT = -0.67*dlogFcentdT;
            constexpr gScalar d = 0.14;
            const gScalar n = 0.75 - 1.27*logFcent;
            const gScalar dndT = - 1.27*dlogFcentdT;

            const gScalar x1 = n - d*(logPr + c);
            const gScalar dx1dT = dndT - d*dcdT;
            const gScalar x2 = (logPr + c)/x1;
            const gScalar dx2dT = (dcdT - x2*dx1dT)/x1;
            const gScalar x3 = 1.0 + x2*x2;
            const gScalar dx3dT = 2.0*x2*dx2dT;
            const gScalar x4 = logFcent/x3;
            const gScalar dx4dT = (dlogFcentdT - x4*dx3dT)/x3;

            return gpuLogTen*F*dx4dT;

        }


        inline CUDA_HOSTDEV gScalar ddPr
        (
            const gScalar T,
            const gScalar Pr,
            const gScalar F
        ) const
        {


            const gScalar logPr = log10(max(Pr, gpuSmall));

            const gScalar dlogPrdPr = Pr >= gpuSmall ? 1/(gpuLogTen*Pr) : 0.0;

            const gScalar Fcent =
                (1.0 - alpha_)*exp(-T/Tsss_)
            + alpha_*exp(-T/Ts_)
            + exp(-Tss_/T);

            const gScalar logFcent = log10(max(Fcent, gpuSmall));

            const gScalar c = -0.4 - 0.67*logFcent;
            constexpr gScalar d = 0.14;
            const gScalar n = 0.75 - 1.27*logFcent;

            const gScalar x1 = n - d*(logPr + c);
            const gScalar dx1dPr = -d*dlogPrdPr;
            const gScalar x2 = (logPr + c)/x1;
            const gScalar dx2dPr = (dlogPrdPr - x2*dx1dPr)/x1;
            const gScalar x3 = 1.0 + x2*x2;
            const gScalar dx3dPr = 2.0*x2*dx2dPr;
            const gScalar x4 = logFcent/x3;
            const gScalar dx4dPr = -x4*dx3dPr/x3;

            return gpuLogTen*F*dx4dPr;

        }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam


#endif

// ************************************************************************* //