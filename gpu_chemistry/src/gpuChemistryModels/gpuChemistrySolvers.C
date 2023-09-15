#include "ode.H"
#include "gpuChemistryModel.H"


#include "forGases.H"
//#include "forLiquids.H"
#include "makegpuChemistrySolver.H"
#include "noChemistrySolver.H"


namespace Foam
{

    forCoeffGases(defineLbChemistrySolvers, nullArg);
    forCoeffGases(makeLbChemistrySolvers, noChemistrySolver);

}


