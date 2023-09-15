#pragma once

#include "gpuODESolverInputs.H"
#include "scalarField.H"

namespace FoamGpu{

static inline gpuODESolverInputs 
read_gpuODESolverInputs(const Foam::dictionary& dict)
{
    gpuODESolverInputs i;
    i.name = dict.lookup<Foam::word>("solver");

    i.absTol = dict.lookupOrDefault<Foam::scalar>("absTol", gpuSmall);
    i.relTol    = dict.lookupOrDefault<Foam::scalar>("relTol", 1e-4);
    i.maxSteps  = dict.lookupOrDefault<Foam::label>("maxSteps", 10000);

    i.safeScale = dict.lookupOrDefault<Foam::scalar>("safeScale", 0.9);
    i.alphaInc  = dict.lookupOrDefault<Foam::scalar>("alphaIncrease", 0.2);
    i.alphaDec  = dict.lookupOrDefault<Foam::scalar>("alphaDecrease", 0.25);
    i.minScale  = dict.lookupOrDefault<Foam::scalar>("minScale", 0.2);
    i.maxScale  = dict.lookupOrDefault<Foam::scalar>("maxScale", 10);


    return i;
}




}