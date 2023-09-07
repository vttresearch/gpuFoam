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
    Foam::gpuPerfectGas

Description
    Perfect gas equation of state:

    \verbatim
        rho = p/(R*T)
    \endverbatim

Usage
    The gas constant R used by the perfect gas equation of state is obtained
    directly from the specie or mixture molecular weight so there is no need to
    provide an equation of \c equationOfState enry in the \c mixture
    specification.

SourceFiles
    gpuPerfectGasI.H
    gpuPerfectGas.C

\*---------------------------------------------------------------------------*/

#ifndef gpuPerfectGas_H
#define gpuPerfectGas_H

#include "cuda_host_dev.H"
#include "gpuConstants.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace FoamGpu
{


/*---------------------------------------------------------------------------*\
                         Class gpuPerfectGas Declaration
\*---------------------------------------------------------------------------*/

class gpuPerfectGas
{

public:

        //- Mass fraction of this specie in mixture
        gScalar Y_;

        //- Molecular weight of specie [kg/kmol]
        gScalar molWeight_;



        inline void operator+=(const gpuPerfectGas& st)
        {

            const gScalar sumY = Y_ + st.Y_;
            if (std::abs(sumY) > gpuSmall)
            {
                molWeight_ = sumY/(Y_/molWeight_ + st.Y_/st.molWeight_);
            }

            Y_ = sumY;
        }
        inline void operator*=(gScalar s)
        {
            Y_*=s;
        }



public:

    // Constructors
            CUDA_HOSTDEV gpuPerfectGas()
                : Y_(0), molWeight_(0)
            {}

            CUDA_HOSTDEV
            gpuPerfectGas(gScalar Y, gScalar molWeight)
                :
                Y_(Y),
                molWeight_(molWeight)
            {}


        // From specie.H

            //- Molecular weight [kg/kmol]
            inline CUDA_HOSTDEV gScalar W() const
            {
                return molWeight_;
            }

            //- Mass fraction of this specie in mixture
            inline CUDA_HOSTDEV gScalar Y() const
            {
                return Y_;
            }

            //- Gas constant [J/kg/K]
            inline CUDA_HOSTDEV gScalar R() const
            {
                return gpuRR / molWeight_;
            }


        // Fundamental properties (from perfectGas.H)


            //- Return density [kg/m^3]
            inline CUDA_HOSTDEV gScalar rho(gScalar p, gScalar T) const
            {
                return p/(this->R()*T);
            }


            //- Return enthalpy contribution [J/kg]
            inline CUDA_HOSTDEV gScalar H(const gScalar p, const gScalar T) const
            {
                (void) p; (void) T;
                return 0.0;
            }

            //- Return Cp contribution [J/(kg K]
            inline CUDA_HOSTDEV gScalar Cp(gScalar p, gScalar T) const
            {
                (void) p; (void) T;
                return 0.0;
            }

            //- Return internal energy contribution [J/kg]
            inline CUDA_HOSTDEV gScalar E(const gScalar p, const gScalar T) const
            {
                (void) p; (void) T;
                return 0.0;
            }

            //- Return Cv contribution [J/(kg K]
            inline CUDA_HOSTDEV gScalar Cv(gScalar p, gScalar T) const
            {
                (void) p; (void) T;
                return 0.0;
            }

            //- Return entropy contribution to the integral of Cp/T [J/kg/K]
            inline CUDA_HOSTDEV gScalar Sp(const gScalar p, const gScalar T) const
            {
                (void) T;
                return -this->R()*std::log(p/gpuPstd);
            }

            //- Return entropy contribution to the integral of Cv/T [J/kg/K]
            inline CUDA_HOSTDEV gScalar Sv(const gScalar p, const gScalar T) const
            {
                //throw std::logic_error("Not implemented");
                (void) p; (void) T;
                return 0.0;
            }

            //- Return compressibility [s^2/m^2]
            inline CUDA_HOSTDEV gScalar psi(gScalar p, gScalar T) const
            {
                (void) p;
                return 1.0/(this->R()*T);
            }

            //- Return compression factor []
            inline CUDA_HOSTDEV gScalar Z(gScalar p, gScalar T) const
            {
                (void) p; (void) T;
                return 1.0;
            }

            //- Return (Cp - Cv) [J/(kg K]
            inline CUDA_HOSTDEV gScalar CpMCv(gScalar p, gScalar T) const
            {
                (void) p; (void) T;
                return this->R();
            }

            //- Return volumetric coefficient of thermal expansion [1/T]
            inline CUDA_HOSTDEV gScalar alphav(const gScalar p, const gScalar T) const
            {
                (void) p;
                return 1.0/T;
            }


};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

inline gpuPerfectGas operator+(const gpuPerfectGas& st1, const gpuPerfectGas& st2)
{
    const gScalar sumY = std::max(st1.Y_ + st2.Y_, gpuSmall);

    if (std::abs(sumY) > gpuSmall)
    {
        return gpuPerfectGas
        (
            sumY,
            sumY/(st1.Y_/st1.molWeight_ + st2.Y_/st2.molWeight_)
        );
    }

    return st1;

}


inline gpuPerfectGas
operator*(gScalar s, const gpuPerfectGas& st)
{
    return gpuPerfectGas
    (
        s*st.Y_,
        st.molWeight_
    );
}


inline gpuPerfectGas operator==(const gpuPerfectGas& st1, const gpuPerfectGas& st2)
{


    gScalar diffY = st2.Y_ - st1.Y_;
    if (std::abs(diffY) < gpuSmall)
    {
        diffY = gpuSmall;
    }

    const gScalar diffRW =
        st2.Y_/st2.molWeight_ - st1.Y_/st1.molWeight_;

    gScalar molWeight = gpuGreat;
    if (std::abs(diffRW) > gpuSmall)
    {
        molWeight = diffY/diffRW;
    }

    return gpuPerfectGas(diffY, molWeight);
}




} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
