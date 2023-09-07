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
    Foam::gpuLindemannFallOffFunction

Description
    Lindemann fall-off function

SourceFiles
    gpuLindemannFallOffFunctionI.H

\*---------------------------------------------------------------------------*/

#ifndef gpuLindemannFallOffFunction_H
#define gpuLindemannFallOffFunction_H

#include "gpuConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace FoamGpu
{



/*---------------------------------------------------------------------------*\
                  Class gpuLindemannFallOffFunction Declaration
\*---------------------------------------------------------------------------*/

class gpuLindemannFallOffFunction
{

public:

    // Constructors

        //- Construct null
        gpuLindemannFallOffFunction() = default;



    // Member Functions


        inline CUDA_HOSTDEV gScalar operator()
        (
            const gScalar T,
            const gScalar Pr
        ) const
        {
            return 1.0;
        }

        inline CUDA_HOSTDEV gScalar ddT
        (
            const gScalar T,
            const gScalar Pr,
            const gScalar F
        ) const
        {
            return 0.0;
        }

        inline CUDA_HOSTDEV gScalar ddPr
        (
            const gScalar T,
            const gScalar Pr,
            const gScalar F
        ) const
        {
            return 0.0;
        }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam


#endif

// ************************************************************************* //