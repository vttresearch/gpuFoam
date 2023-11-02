#define CATCH_CONFIG_ENABLE_BENCHMARKING
//#define CATCH_CONFIG_MAIN
#include "catch.H"

#include "host_device_vectors.H"
#include "test_utilities.H"
#include "create_gpu_inputs.H"
#include "ludecompose.H"
#include "gpuODESystem.H"
#include "makeGpuOdeSolver.H"



static constexpr TestData::Mechanism mech = TestData::GRI;

TEST_CASE("LU")
{



    const gLabel nEqns = TestData::equationCount(mech);



    std::vector<gScalar> vals(nEqns * nEqns);


    fill_random(vals);

    device_vector<gScalar> matrix(vals.begin(), vals.end());

    device_vector<gLabel> pivot(nEqns, 0);
    device_vector<gScalar> v(nEqns, 0);
    device_vector<gScalar> source(nEqns, 0);

    auto m_span = make_mdspan(matrix, extents<2>{nEqns, nEqns});
    auto p_span = make_mdspan(pivot, extents<1>{nEqns});
    auto v_span = make_mdspan(v, extents<1>{nEqns});
    auto s_span = make_mdspan(source, extents<1>{nEqns});

    auto op1 = [=] __device__() {
        FoamGpu::LUDecompose(m_span, p_span, v_span);
        return p_span(4);
    };

    auto op2 = [=] __device__ (){
        FoamGpu::LUBacksubstitute(m_span, p_span, s_span);
        return s_span(5);
    };


    BENCHMARK("WARMUP"){
        return eval(op1);
    };

    BENCHMARK("LUdecompose"){

        return eval(op1);
    };

    BENCHMARK("LUbacksubstitute"){

        return eval(op2);
    };




}


TEST_CASE("gpuODESystem"){

    using namespace FoamGpu;

    auto gpu_thermos = toDeviceVector(makeGpuThermos(mech));
    auto gpu_reactions = toDeviceVector(makeGpuReactions(mech));

    const gLabel nCells = 1;
    const gLabel nSpecie = TestData::speciesCount(mech);
    const gLabel nEqns = TestData::equationCount(mech);


    gpuODESystem system
    (
        nEqns,
        gLabel(gpu_reactions.size()),
        make_raw_pointer(gpu_thermos.data()),
        make_raw_pointer(gpu_reactions.data())
    );

    std::vector<gScalar> vals(nEqns);
    fill_random(vals);
    device_vector<gScalar> y(vals.begin(), vals.end());

    device_vector<gScalar> dy(nEqns);
    device_vector<gScalar> J(nEqns * nEqns);

    memoryResource_t memory(nCells, nSpecie);
    auto buffers = toDeviceVector(splitToBuffers(memory));

    auto op1 = [        =,
                buffers = make_mdspan(buffers, extents<1>{1}),
                y       = make_mdspan(y, extents<1>{nEqns}),
                dy      = make_mdspan(dy, extents<1>{nEqns})
                ]__device__() {
        system.derivatives(y, dy, buffers[0]);
        return dy(5);
    };

    auto op2 =
        [        =,
         buffers = make_mdspan(buffers, extents<1>{nCells}),
         y       = make_mdspan(y, extents<1>{nEqns}),
         dy      = make_mdspan(dy, extents<1>{nEqns}),
         J = make_mdspan(J, extents<2>{nEqns, nEqns})
         ] __device__() {
            system.jacobian(y, dy, J, buffers[0]);
            return J(3, 3);
        };

    BENCHMARK("derivatives"){
        return eval(op1);
    };

    BENCHMARK("Jacobian"){
        return eval(op2);
    };

}


TEST_CASE("gpuODESolver"){

    using namespace FoamGpu;

    auto gpu_thermos = toDeviceVector(makeGpuThermos(mech));
    auto gpu_reactions = toDeviceVector(makeGpuReactions(mech));

    const gLabel nCells = 1;
    const gLabel nSpecie = TestData::speciesCount(mech);
    const gLabel nEqns = TestData::equationCount(mech);


    gpuODESystem system
    (
        nEqns,
        gLabel(gpu_reactions.size()),
        make_raw_pointer(gpu_thermos.data()),
        make_raw_pointer(gpu_reactions.data())
    );

    std::vector<gScalar> vals(nEqns);
    assign_test_condition(vals, mech);
    //fill_random(vals);
    const device_vector<gScalar> y0(vals.begin(), vals.end());
    device_vector<gScalar> y(y0.size());

    memoryResource_t memory(nCells, nSpecie);
    auto buffers = toDeviceVector(splitToBuffers(memory));


    const gScalar xStart = 0.0;
    const gScalar xEnd = 2E-7;
    const gScalar dxTry = 1E-7;



    //i.absTol =

    SECTION("Rosenbrock12"){
        auto i = TestData::makeGpuODEInputs("Rosenbrock12", mech);
        auto solver = make_gpuODESolver(system, i);

        auto op = [        =,
                   buffers = make_mdspan(buffers, extents<1>{nCells}),
                   y0       = make_mdspan(y0, extents<1>{nEqns}),
                   y       = make_mdspan(y, extents<1>{nEqns})
                  ] __device__() {
            gScalar temp = dxTry;
            for (int i = 0; i < nEqns; ++i) {
                y(i) = y0(i);
            }

            solver.solve(xStart, xEnd, y, temp, buffers[0]);
            return temp;
        };


        BENCHMARK("Rosenbrock12"){
            return eval(op);
        };

    }



}