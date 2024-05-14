#include "addToRunTimeSelectionTable.H"
#include "gpuChemistryModel.H"
#include "noChemistrySolver.H"

namespace Foam {

// Adding new models to runtime selection is horrible,
// the model / solver combination is chosen based on the
//  'runTimeName', the syntax of which depends on the gpuThermoType name.
using gpuModelType  = gpuChemistryModel;
using gpuSolverType = noChemistrySolver<gpuModelType>;
using gpuThermoType = typename gpuModelType::cpuThermoType;

defineTypeNameAndDebug(gpuModelType, 0);

static const word runTimeName = word(gpuSolverType::typeName_()) + "<" +
                                word(gpuModelType::typeName_()) + "<" +
                                gpuThermoType::typeName() + ">>";

defineTemplateTypeNameAndDebugWithName(gpuSolverType, runTimeName.c_str(), 0);

addToRunTimeSelectionTable(basicChemistryModel, gpuSolverType, thermo);

} // namespace Foam
