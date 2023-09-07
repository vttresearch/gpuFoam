/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Copyright (C) 2022 VTT Technical Research Centre of Finland Ltd
-------------------------------------------------------------------------------
License
    This file is a derivative work of OpenFOAM.

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
    Foam::chemistryProblem

Description
    A class holding everything required to update fields after a
    cell specific chemistry problem has been solved

SourceFiles
    gpuChemistrySolution.C

\*---------------------------------------------------------------------------*/

#ifndef gpuChemistrySolution_H
#define gpuChemistrySolution_H

#include "scalarField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{


/*---------------------------------------------------------------------------*\
                           Class problem Declaration
\*---------------------------------------------------------------------------*/


class gpuChemistrySolution
{

public:




    gpuChemistrySolution() = default;

    gpuChemistrySolution(label nSpecie) :
        celli(0),
        cpuTime(0),
        deltaTChem(0),
        Y(nSpecie)
        {}



    // Member functions

        // To satisfy the lbSolution interface


            void put(Ostream& os) const
            {
                os << celli;
                os << cpuTime;
                os << deltaTChem;
                os << Y;
            }

            void get(Istream& is)
            {
                is >> celli;
                is >> cpuTime;
                is >> deltaTChem;
                is >> Y;
            }


            bool equals(const gpuChemistrySolution& rhs) const
            {
                return false;
            }



    // Member variables


    label celli;
    scalar cpuTime;
    scalar deltaTChem;
    scalarField Y;


};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //



} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

//#include "lbProblemI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

//#ifdef NoRepository
//    #include "problem.C"
//#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
