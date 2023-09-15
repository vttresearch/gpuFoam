#include "ode.H"
#include "gpuChemistryModel.H"

#include "basicChemistryModel.H"
#include "makeChemistrySolver.H"
#include "forGases.H"
//#include "forLiquids.H"
//#include "makegpuChemistrySolver.H"
#include "noChemistrySolver.H"


#define defineGpuChemistrySolvers(nullArg, ThermoPhysics)                       \
    defineChemistrySolver                                                      \
    (                                                                          \
        gpuChemistryModel,                                                      \
        ThermoPhysics                                                          \
    )

#define makeGpuChemistrySolvers(Solver, ThermoPhysics)                          \
    makeChemistrySolver                                                        \
    (                                                                          \
        Solver,                                                                \
        gpuChemistryModel,                                                      \
        ThermoPhysics                                                          \
    )





namespace Foam
{
    /*
    using cpuThermo =
        Foam::sutherlandTransport
        <
            Foam::species::thermo
            <
                Foam::janafThermo
                <
                    Foam::perfectGas
                    <
                        Foam::specie
                    >
                >,
                Foam::sensibleEnthalpy
            >
        >;

    using myChemistryModel = gpuChemistryModel<cpuThermo>;
    defineTemplateTypeNameAndDebug
    (
        myChemistryModel,
        0
    );

    addToRunTimeSelectionTable
    (
        basicChemistryModel,
        myChemistryModel,
        thermo
    );





   */
    /*
    using mySolver = noChemistrySolver<gpuChemistryModel>;
    defineTemplateTypeNameAndDebug
    (
        mySolver,
        0
    );

    addToRuntimeSelectionTable
    (
        basicChemistryModel,
        mySolver,
        thermo
    );
    */
    /*
    addTemplatedToRunTimeSelectionTable
    (
        chemistrySolver,
        mySolver,
        mySolver,
        thermo
    );
    */
    /*
    addToRunTimeSelectionTable
    (
        chemistrySolver,
        mySolver,
        thermo
    );
    */

    /*
    defineTypeNameAndDebug
    (
        gpuChemistryModel,
        0
    );

    addToRunTimeSelectionTable
    (
        basicChemistryModel,
        gpuChemistryModel,
        thermo
    );
    */

    //forCoeffGases(defineGpuChemistrySolvers, nullArg);
    //forCoeffGases(makeGpuChemistrySolvers, noChemistrySolver);




}


