#include "catch.H"
#include "hip/hip_runtime.h"

#include "gpuThermo.H"
#include "gpuPerfectGas.H"
#include "gpuReaction.H"
#include "gpuMemoryResource.H"
#include "gpuODESystem.H"
//#include "device_vector.H"
#include "create_gpu_inputs.H"
#include "host_device_vectors.H"
#include "error_handling.H"


#include "cpu_reference_results.H"

//#include "test_utilities.H"
//#include "create_foam_inputs.H"

using namespace FoamGpu;

template<class F>
__global__ void test_on_device(F f)
{
    f();
//printf("In kernel ================ \n");
  //int i = blockIdx.x*blockDim.x + threadIdx.x;
  //if (i < n) {p[i].Y_ = 43.0;}
  //if (i < n) y[i] = a*x[i] + y[i];
}




TEST_CASE("Test device_vector"){

    std::vector<double> v1 = {1,2,3};
    device_vector<double> v2 = toDeviceVector(v1);
    std::vector<double> v3 = toStdVector(v2);

    CHECK(v3 == v1);


}

TEST_CASE("Test gpuOdeSystem"){



    TestData::Mechanism mech = TestData::Mechanism::H2;

    gLabel nSpecie = TestData::speciesCount(mech);
    gLabel nEqns = TestData::equationCount(mech);
    gLabel nCells = 1;

    std::vector<gpuThermo> h_thermos = TestData::makeGpuThermos(mech);
    std::vector<gpuReaction> h_reactions = TestData::makeGpuReactions(mech);
    auto ode_inputs = TestData::makeGpuODEInputs("Rosenbrock23", mech);


    auto thermos = toDeviceVector(h_thermos);
    auto reactions = toDeviceVector(h_reactions);

    gpuODESystem system
    (
        nEqns,
        reactions.size(),
        thermos.data(),
        reactions.data()
    );


    std::vector<gScalar> Y_h(nEqns);

    assign_test_condition(Y_h, mech);

    device_vector<gScalar> Y = toDeviceVector(Y_h);
    device_vector<gScalar> dY(nEqns);

    gpuMemoryResource memory(nCells, nSpecie);
    auto Buffers = toDeviceVector(splitToBuffers(memory));


    auto buffers = make_mdspan(Buffers, extents<1>{1});
    auto        y = make_mdspan(Y, extents<1>{nEqns});
    auto        dy = make_mdspan(dY, extents<1>{nEqns});

    auto f = [=] (){
        system.derivatives(y, dy, buffers[0]);
    };

    hipLaunchKernelGGL(test_on_device, dim3(1),
                        dim3(1), 0, 0, f);


    std::vector<gScalar> dY_res = toStdVector(dY);
    std::vector<gScalar> dY_cor = TestData::derivative_result(mech);

    CHECK
    (
        dY_res == dY_cor
    );



}

TEST_CASE("Test gpuTests")
{

    //thrust::device_vector<double> data(1, 0.0);



    TestData::Mechanism mech = TestData::Mechanism::H2;

    std::vector<gpuThermo> h_thermos = TestData::makeGpuThermos(mech);
    std::vector<gpuReaction> h_reactions = TestData::makeGpuReactions(mech);
    auto ode_inputs = TestData::makeGpuODEInputs("Rosenbrock23", mech);


    auto d_thermos = toDeviceVector(h_thermos);
    auto d_reactions = toDeviceVector(h_reactions);

    //gpuODESystem()


    CHECK(1 == 1);


    //gpuThermo t1 = h_thermos[0];

    //thrust::device_vector<gpuThermo> d_thermos(h_thermos.size(), t1);

    /*
    device_vector<double> data(1);
    double* ptr = data.data();


    hipLaunchKernelGGL(test_on_device, dim3(1),
                        dim3(1), 0, 0, ptr);

    gpuErrorCheck(hipGetLastError());
    gpuErrorCheck(hipDeviceSynchronize());

    */




    //OH parameters

    //gpuPerfectGas eos(0.1, 17.00737);




    //CHECK(1 == 1);



}