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

\*---------------------------------------------------------------------------*/

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
/*
template<class ThermoType>
inline Foam::label Foam::gpuChemistryModel<ThermoType>::nSpecie() const
{
    return nSpecie_;
}

template<class ThermoType>
inline Foam::label Foam::gpuChemistryModel<ThermoType>::nEqns() const
{
    // nEqns = number of species + temperature + pressure
    return nSpecie() + 2;
}

template<class ThermoType>
inline Foam::label Foam::gpuChemistryModel<ThermoType>::nReaction() const
{
    return reactions_.size();
}

template<class ThermoType>
inline const Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::gpuChemistryModel<ThermoType>::RR(const label i) const
{
    return RR_[i];
}


template<class ThermoType>
Foam::DimensionedField<Foam::scalar, Foam::volMesh>&
Foam::gpuChemistryModel<ThermoType>::RR(const label i)
{
    return RR_[i];
}
*/

// ************************************************************************* //
