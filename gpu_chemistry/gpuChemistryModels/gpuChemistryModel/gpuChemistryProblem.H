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
    Foam::gpuChemistryProblem

Description
    A class holding everything required to solve a cell-specific
    chemistry reaction rate.

SourceFiles
    gpuChemistryProblem.C

\*---------------------------------------------------------------------------*/

#ifndef gpuChemistryProblem_H
#define gpuChemistryProblem_H

#include "scalarField.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{


/*---------------------------------------------------------------------------*\
                           Class problem Declaration
\*---------------------------------------------------------------------------*/


class gpuChemistryProblem
{

public:




    gpuChemistryProblem() = default;

    gpuChemistryProblem(label nSpecie) :
        celli(0),
        p(0),
        T(0),
        deltaT(0),
        deltaTChem(0),
        cpuTime(0),
        Y(nSpecie)
        {}


    static constexpr bool is_problem = true;

    // Member functions

        // To satisfy the lbProblem interface

            scalar weight() const {return cpuTime;}

            void put(Ostream& os) const
            {
                os << celli;
                os << p;
                os << T;
                os << deltaT;
                os << deltaTChem;
                os << cpuTime;
                os << Y;
            }

            void get(Istream& is)
            {
                is >> celli;
                is >> p;
                is >> T;
                is >> deltaT;
                is >> deltaTChem;
                is >> cpuTime;
                is >> Y;
            }


            bool equals(const gpuChemistryProblem& rhs) const
            {
                return false;
            }



    // Member variables


    label celli;
    scalar p;
    scalar T;
    scalar deltaT;
    scalar deltaTChem;
    scalar cpuTime;
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
