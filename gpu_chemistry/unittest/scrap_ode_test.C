/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
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

Description

\*---------------------------------------------------------------------------*/

#include "catch.H"

#include <utility> //std::pair
#include <algorithm>


#include "argList.H"
#include "IOmanip.H"
#include "ODESystem.H"
#include "ODESolver.H"
#include "gpuODESystem.H"
#include "gpuODESolverBase.H"
#include "gpuSeulex.H"
#include "gpuRosenbrock34.H"
#include "mdspan.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

class testODE
:
    public ODESystem
{

public:

    testODE()
    {}

    label nEqns() const
    {
        return 4;
    }

    void derivatives
    (
        const scalar x,
        const scalarField& y,
        const label li,
        scalarField& dydx
    ) const
    {
        dydx[0] = -y[1];
        dydx[1] = y[0] - (1.0/x)*y[1];
        dydx[2] = y[1] - (2.0/x)*y[2];
        dydx[3] = y[2] - (3.0/x)*y[3];
    }

    void jacobian
    (
        const scalar x,
        const scalarField& y,
        const label li,
        scalarField& dfdx,
        scalarSquareMatrix& dfdy
    ) const
    {
        dfdx[0] = 0.0;
        dfdx[1] = (1.0/sqr(x))*y[1];
        dfdx[2] = (2.0/sqr(x))*y[2];
        dfdx[3] = (3.0/sqr(x))*y[3];

        dfdy(0, 0) = 0.0;
        dfdy(0, 1) = -1.0;
        dfdy(0, 2) = 0.0;
        dfdy(0, 3) = 0.0;

        dfdy(1, 0) = 1.0;
        dfdy(1, 1) = -1.0/x;
        dfdy(1, 2) = 0.0;
        dfdy(1, 3) = 0.0;

        dfdy(2, 0) = 0.0;
        dfdy(2, 1) = 1.0;
        dfdy(2, 2) = -2.0/x;
        dfdy(2, 3) = 0.0;

        dfdy(3, 0) = 0.0;
        dfdy(3, 1) = 0.0;
        dfdy(3, 2) = 1.0;
        dfdy(3, 3) = -3.0/x;
    }
};


class testGpuODESystem
{

public:

    testGpuODESystem() = default;

    label nEqns() const
    {
        return 4;
    }

    template<class Span1, class Span2>
    void derivatives
    (
        const scalar x,
        const Span1& y,
        const label li,
        Span2& dydx
    ) const
    {
        dydx[0] = -y[1];
        dydx[1] = y[0] - (1.0/x)*y[1];
        dydx[2] = y[1] - (2.0/x)*y[2];
        dydx[3] = y[2] - (3.0/x)*y[3];
    }

    template<class Span1, class Span2, class TwoDSpan>
    void jacobian
    (
        const scalar x,
        const Span1& y,
        const label li,
        Span2& dfdx,
        TwoDSpan& dfdy
    ) const
    {
        dfdx[0] = 0.0;
        dfdx[1] = (1.0/sqr(x))*y[1];
        dfdx[2] = (2.0/sqr(x))*y[2];
        dfdx[3] = (3.0/sqr(x))*y[3];

        dfdy(0, 0) = 0.0;
        dfdy(0, 1) = -1.0;
        dfdy(0, 2) = 0.0;
        dfdy(0, 3) = 0.0;

        dfdy(1, 0) = 1.0;
        dfdy(1, 1) = -1.0/x;
        dfdy(1, 2) = 0.0;
        dfdy(1, 3) = 0.0;

        dfdy(2, 0) = 0.0;
        dfdy(2, 1) = 1.0;
        dfdy(2, 2) = -2.0/x;
        dfdy(2, 3) = 0.0;

        dfdy(3, 0) = 0.0;
        dfdy(3, 1) = 0.0;
        dfdy(3, 2) = 1.0;
        dfdy(3, 3) = -3.0/x;
    }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //



static std::pair<bool, OStringStream> ODETestKernel(word solverName)
{

    OStringStream log;

    // Create the ODE system
    testODE ode;

    dictionary dict;
    dict.add("solver", solverName);

    // Create the selected ODE system solver
    autoPtr<ODESolver> odeSolver = ODESolver::New(ode, dict);

    // Initialise the ODE system fields
    scalar xStart = 1.0;
    scalarField yStart(ode.nEqns());
    yStart[0] = ::Foam::j0(xStart);
    yStart[1] = ::Foam::j1(xStart);
    yStart[2] = ::Foam::jn(2, xStart);
    yStart[3] = ::Foam::jn(3, xStart);

    // Print the evolution of the solution and the time-step
    scalarField dyStart(ode.nEqns());
    ode.derivatives(xStart, yStart, 0, dyStart);

    log << setw(10) << "relTol" << setw(12) << "dxEst";
    log << setw(13) << "dxDid" << setw(14) << "dxNext" << endl;
    log << setprecision(6);

    for (label i=0; i<15; i++)
    {
        scalar relTol = ::Foam::exp(-scalar(i + 1));

        scalar x = xStart;
        scalarField y(yStart);

        scalar dxEst = 0.6;
        scalar dxNext = dxEst;

        odeSolver->relTol() = relTol;
        odeSolver->solve(x, y, 0, dxNext);

        log << scientific << setw(13) << relTol;
        log << fixed << setw(11) << dxEst;
        log << setw(13) << x - xStart << setw(13) << dxNext
            << setw(13) << y[0] << setw(13) << y[1]
            << setw(13) << y[2] << setw(13) << y[3]
            << endl;
    }

    scalar x = xStart;
    scalar xEnd = x + 1.0;
    scalarField y(yStart);

    scalarField yEnd(ode.nEqns());
    yEnd[0] = ::Foam::j0(xEnd);
    yEnd[1] = ::Foam::j1(xEnd);
    yEnd[2] = ::Foam::jn(2, xEnd);
    yEnd[3] = ::Foam::jn(3, xEnd);

    scalar dxEst = 0.5;

    odeSolver->relTol() = 1e-4;
    odeSolver->solve(x, xEnd, y, 0, dxEst);

    log << nl << "Analytical: y(2.0) = " << yEnd << endl;
    log      << "Numerical:  y(2.0) = " << y << ", dxEst = " << dxEst << endl;

    log << "\nEnd\n" << endl;

    bool success = sum(mag(yEnd-y)) < 1E-4;


    return std::make_pair(success, log);

}

template<class System, class Solver>
static std::pair<bool, OStringStream> gpuODETestKernel(const System& ode, const Solver& odeSolver)
{

    OStringStream log;



    // Initialise the ODE system fields
    scalar xStart = 1.0;
    scalarField yStart(ode.nEqns());
    yStart[0] = ::Foam::j0(xStart);
    yStart[1] = ::Foam::j1(xStart);
    yStart[2] = ::Foam::jn(2, xStart);
    yStart[3] = ::Foam::jn(3, xStart);

    // Print the evolution of the solution and the time-step
    scalarField dyStart(ode.nEqns());
    ode.derivatives(xStart, yStart, 0, dyStart);

    log << setw(10) << "relTol" << setw(12) << "dxEst";
    log << setw(13) << "dxDid" << setw(14) << "dxNext" << endl;
    log << setprecision(6);


    for (label i=0; i<15; i++)
    {
        scalar relTol = ::Foam::exp(-scalar(i + 1));

        scalar x = xStart;
        scalarField y(yStart);

        scalar dxEst = 0.6;
        scalar dxNext = dxEst;

        odeSolver->relTol() = relTol;
        odeSolver->solve(x, y, 0, dxNext);

        log << scientific << setw(13) << relTol;
        log << fixed << setw(11) << dxEst;
        log << setw(13) << x - xStart << setw(13) << dxNext
            << setw(13) << y[0] << setw(13) << y[1]
            << setw(13) << y[2] << setw(13) << y[3]
            << endl;
    }

    scalar x = xStart;
    scalar xEnd = x + 1.0;
    scalarField y(yStart);

    scalarField yEnd(ode.nEqns());
    yEnd[0] = ::Foam::j0(xEnd);
    yEnd[1] = ::Foam::j1(xEnd);
    yEnd[2] = ::Foam::jn(2, xEnd);
    yEnd[3] = ::Foam::jn(3, xEnd);

    scalar dxEst = 0.5;

    odeSolver->relTol() = 1e-4;
    odeSolver->solve(x, xEnd, y, 0, dxEst);

    log << nl << "Analytical: y(2.0) = " << yEnd << endl;
    log      << "Numerical:  y(2.0) = " << y << ", dxEst = " << dxEst << endl;

    log << "\nEnd\n" << endl;

    bool success = sum(mag(yEnd-y)) < 1E-4;


    return std::make_pair(success, log);


}


TEST_CASE("Test cpu-ODE")
{
    /*
    SECTION("Euler")
    {
        auto pair = ODETestKernel("Euler");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("EulerSI")
    {
        auto pair = ODETestKernel("EulerSI");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }
    */


    SECTION("rodas23")
    {
        auto pair = ODETestKernel("rodas23");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("rodas34")
    {
        auto pair = ODETestKernel("rodas34");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("RKCK45")
    {
        auto pair = ODETestKernel("RKCK45");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("RKDP45")
    {
        auto pair = ODETestKernel("RKDP45");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("RKF45")
    {
        auto pair = ODETestKernel("RKF45");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("Rosenbrock12")
    {
        auto pair = ODETestKernel("Rosenbrock12");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("Rosenbrock23")
    {
        auto pair = ODETestKernel("Rosenbrock23");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("Rosenbrock34")
    {
        auto pair = ODETestKernel("Rosenbrock34");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("seulex")
    {
        auto pair = ODETestKernel("seulex");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("SIBS")
    {
        auto pair = ODETestKernel("SIBS");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("Trapezoid")
    {
        auto pair = ODETestKernel("Trapezoid");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }


}


TEST_CASE("Test gpu-ODE")
{

    SECTION("seulex")
    {
        dictionary dict;
        testGpuODESystem system;

        auto* solver = new gpuSeulex<testGpuODESystem>(system, dict);
        auto pair = gpuODETestKernel(system, solver);
        delete solver;

        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }

    SECTION("Rosenbrock34")
    {
        dictionary dict;
        testGpuODESystem system;

        auto* solver = new gpuRosenbrock34<testGpuODESystem>(system, dict);
        auto pair = gpuODETestKernel(system, solver);
        delete solver;

        CHECK(pair.first == true);

    }

    /*
    SECTION("RKF45")
    {
        auto pair = gpuODETestKernel("gpuRKF45");
        CHECK(pair.first == true);
        //Info << pair.second.str() << endl;
    }
    */
}


template<class Field>
static void init1(Field& field)
{
    for (int i = 0; i < field.size(); ++i)
    {
        field[i] = 0.1*i;
    }
}

template<class Field>
static void init2(Field& field)
{
    for (int i = 0; i < field.size(); ++i)
    {
        field[i] = -0.1*i + scalar(i);
    }
}

template<class Solver, class Field>
static void evaluate(const Solver& solver, Field& field)
{

    scalar xStart = 1.1;
    scalar xEnd = 1.2;
    label li = 0;
    scalar dxEst = 0.01;

    for (int i = 0; i < 3; ++i)
    {
        solver->solve(xStart, xEnd, field, li, dxEst);
    }

}


TEST_CASE("Compare seulex cpu/gpu ODE")
{
    // Create the ODE system
    testODE cpu_ode_system;
    testGpuODESystem gpu_ode_system;

    dictionary cpu_dict;
    cpu_dict.add("solver", "seulex");

    dictionary gpu_dict;
    gpu_dict.add("solver", "gpuSeulex");


    autoPtr<ODESolver> cpuSolver = ODESolver::New(cpu_ode_system, cpu_dict);
    //autoPtr<gpuODESolverBase> gpuSolver = gpuODESolverBase::New(gpu_ode_system, gpu_dict);


    autoPtr<gpuSeulex<testGpuODESystem>> gpuSolver
    (
        new gpuSeulex<testGpuODESystem>(gpu_ode_system, gpu_dict)
    );


    //Ensure that the systems are the same
    //CHECK(cpuSolver->nEqns() == gpuSolver->nEqns());

    label nEqns = cpuSolver->nEqns();

    SECTION("Test 1")
    {
        scalarField y_cpu(nEqns, 0);
        scalarField y_gpu(nEqns, 0);

        init1(y_cpu);
        init1(y_gpu);

        evaluate(cpuSolver, y_cpu);
        evaluate(gpuSolver, y_gpu);

        CHECK(std::abs(sum(y_cpu) - sum(y_gpu)) < 1E-7);

    }
    SECTION("Test 2")
    {
        scalarField y_cpu(nEqns, 0);
        scalarField y_gpu(nEqns, 0);

        init2(y_cpu);
        init2(y_gpu);

        evaluate(cpuSolver, y_cpu);
        evaluate(gpuSolver, y_gpu);

        CHECK(std::abs(sum(y_cpu) - sum(y_gpu)) < 1E-7);

    }



}

TEST_CASE("Compare Rosenbrock34 cpu/gpu ODE")
{
    // Create the ODE system
    testODE cpu_ode_system;
    testGpuODESystem gpu_ode_system;

    dictionary cpu_dict;
    cpu_dict.add("solver", "Rosenbrock34");

    dictionary gpu_dict;
    gpu_dict.add("solver", "Rosenbrock34");


    autoPtr<ODESolver> cpuSolver = ODESolver::New(cpu_ode_system, cpu_dict);
    //autoPtr<gpuODESolverBase> gpuSolver = gpuODESolverBase::New(gpu_ode_system, gpu_dict);


    autoPtr<gpuRosenbrock34<testGpuODESystem>> gpuSolver
    (
        new gpuRosenbrock34<testGpuODESystem>(gpu_ode_system, gpu_dict)
    );


    //Ensure that the systems are the same
    //CHECK(cpuSolver->nEqns() == gpuSolver->nEqns());

    label nEqns = cpuSolver->nEqns();

    SECTION("Test 1")
    {
        scalarField y_cpu(nEqns, 0);
        scalarField y_gpu(nEqns, 0);

        init1(y_cpu);
        init1(y_gpu);

        evaluate(cpuSolver, y_cpu);
        evaluate(gpuSolver, y_gpu);

        CHECK(std::abs(sum(y_cpu) - sum(y_gpu)) < 1E-7);

    }
    SECTION("Test 2")
    {
        scalarField y_cpu(nEqns, 0);
        scalarField y_gpu(nEqns, 0);

        init2(y_cpu);
        init2(y_gpu);

        evaluate(cpuSolver, y_cpu);
        evaluate(gpuSolver, y_gpu);

        CHECK(std::abs(sum(y_cpu) - sum(y_gpu)) < 1E-7);

    }



}



/*
TEST_CASE("Compare RKF45 cpu/gpu ODE")
{
    // Create the ODE system
    testODE cpu_ode_system;
    testGpuODESystem gpu_ode_system;

    dictionary cpu_dict;
    cpu_dict.add("solver", "RKF45");

    dictionary gpu_dict;
    gpu_dict.add("solver", "gpuRKF45");


    autoPtr<ODESolver> cpuSolver = ODESolver::New(cpu_ode_system, cpu_dict);
    autoPtr<gpuODESolverBase> gpuSolver = gpuODESolverBase::New(gpu_ode_system, gpu_dict);

    //Ensure that the systems are the same
    CHECK(cpuSolver->nEqns() == gpuSolver->nEqns());

    label nEqns = cpuSolver->nEqns();

    SECTION("Test 1")
    {
        scalarField y_cpu(nEqns, 0);
        scalarField y_gpu(nEqns, 0);

        init1(y_cpu);
        init1(y_gpu);

        evaluate(cpuSolver, y_cpu);
        evaluate(gpuSolver, y_gpu);

        CHECK(std::abs(sum(y_cpu) - sum(y_gpu)) < 1E-7);

    }
    SECTION("Test 2")
    {
        scalarField y_cpu(nEqns, 0);
        scalarField y_gpu(nEqns, 0);

        init2(y_cpu);
        init2(y_gpu);

        evaluate(cpuSolver, y_cpu);
        evaluate(gpuSolver, y_gpu);
        CHECK(std::abs(sum(y_cpu) - sum(y_gpu)) < 1E-7);

    }



}
*/

// ************************************************************************* //

