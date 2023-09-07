#define CATCH_CONFIG_ENABLE_BENCHMARKING
//#define CATCH_CONFIG_MAIN
#include "catch.H"

#include <vector>
#include "test_utilities.H"
#include "gpuODESystem.H"
#include "gpuRosenbrock34.H"
#include "Rosenbrock34.H"
#include "mdspan.H"

#include "mock_of_odesystem.H"
#include "mock_of_rosenbrock34.H"

TEST_CASE("gpuReaction::dNdtByV")
{

    using namespace FoamGpu;

    Foam::MockOFSystem cpu;

    auto gpu_reactions_temp = make_gpu_reactions();
    auto gpu_reactions = device_vector<gpuReaction>(gpu_reactions_temp.begin(), gpu_reactions_temp.end());

    auto cpu_reactions = make_cpu_reactions();

    
    gLabel nSpecie = make_species_table().size();

    Foam::scalarField c_cpu(nSpecie);
    fill_random(c_cpu);
    Foam::scalarField dNdtByV_cpu(nSpecie, 0.0);

    auto c_gpu = to_device_vec(c_cpu);
    auto dNdtByV_gpu = to_device_vec(dNdtByV_cpu);

    const gScalar p = 1E5;
    const gScalar T = 350;


    auto f_cpu = [&](){
        gScalar sum = 0.0;
        for (gLabel i = 0; i < cpu_reactions.size(); ++i)
        {
            cpu_reactions[i].dNdtByV(p, T, c_cpu, 0, dNdtByV_cpu,  false, Foam::List<Foam::label>{}, 0);
            sum += dNdtByV_cpu[4];
        }
        return sum;

    };

    auto f_gpu = 
    [   =,
        reactions = make_mdspan(gpu_reactions, extents<1>{gpu_reactions.size()}),
        c = make_mdspan(c_gpu, extents<1>{nSpecie}),
        dNdtByV = make_mdspan(dNdtByV_gpu, extents<1>{nSpecie})
    ]()
    {

        gScalar sum = 0.0;
        for (size_t i = 0; i < reactions.size(); ++i)
        {
            reactions[i].dNdtByV(p, T, c, dNdtByV);
            sum += dNdtByV[3];
        }
        return sum;
    };

    BENCHMARK("CPU dNdtByV")
    {
        return f_cpu();
    };

    BENCHMARK("GPU dNdtByV")
    {
        return eval(f_gpu);
    };

}

TEST_CASE("gpuReaction::ddNdtByVdcTp")
{

    using namespace FoamGpu;

    Foam::MockOFSystem cpu;

    auto gpu_reactions_temp = make_gpu_reactions();
    auto gpu_reactions = device_vector<gpuReaction>(gpu_reactions_temp.begin(), gpu_reactions_temp.end());

    auto cpu_reactions = make_cpu_reactions();

    
    gLabel nSpecie = make_species_table().size();
    

    Foam::scalarField c_cpu(nSpecie);
    fill_random(c_cpu);
    Foam::scalarField dNdtByV_cpu(nSpecie, 0.0);
    Foam::scalarField work1_cpu(nSpecie, 0.0);
    Foam::scalarField work2_cpu(nSpecie, 0.0);
    Foam::scalarSquareMatrix ret_cpu(nSpecie + 2, 0.0);

    auto c_gpu = to_device_vec(c_cpu);
    auto dNdtByV_gpu = to_device_vec(dNdtByV_cpu);
    auto work1_gpu = to_device_vec(work1_cpu);
    auto work2_gpu = to_device_vec(work2_cpu);
    device_vector<gScalar> ret_gpu((nSpecie+2) * (nSpecie+2));
    const gScalar p = 1E5;
    const gScalar T = 350;

        
    auto f_cpu = [&](){
        gScalar sum = 0.0;
        for (gLabel i = 0; i < cpu_reactions.size(); ++i)
        {
            cpu_reactions[i].ddNdtByVdcTp
            (
                p, T, c_cpu, 0, dNdtByV_cpu, ret_cpu,
                false, 
                Foam::List<Foam::label>{}, 
                0,
                nSpecie,
                work1_cpu,
                work2_cpu
                );
            sum += ret_cpu(4,4);
        }
        return sum;

    };
    
    auto f_gpu = 
    [   =,
        reactions = make_mdspan(gpu_reactions, extents<1>{gpu_reactions.size()}),
        c = make_mdspan(c_gpu, extents<1>{nSpecie}),
        dNdtByV = make_mdspan(dNdtByV_gpu, extents<1>{nSpecie}),
        work1 = make_mdspan(work1_gpu, extents<1>{nSpecie}),
        work2 = make_mdspan(work2_gpu, extents<1>{nSpecie}),
        ret = make_mdspan(ret_gpu, extents<2>{nSpecie+2, nSpecie+2})
    ]()
    {
        
        gScalar sum = 0.0;
        for (size_t i = 0; i < reactions.size(); ++i)
        {
            reactions[i].ddNdtByVdcTp
            (
                p, T, c, 0, dNdtByV,
                ret,
                0,
                nSpecie,
                work1,
                work2
            );
            sum += ret(4,4);
        }
        return sum;
    };
    
    BENCHMARK("CPU ddNdtByVdcTp")
    {
        return f_cpu();
    };
    
    BENCHMARK("GPU ddNdtByVdcTp")
    {
        return eval(f_gpu);
    };
    

}


TEST_CASE("gpuODESystem::derivatives")
{

    using namespace FoamGpu;

    Foam::MockOFSystem cpu;

    auto gpu_thermos_temp = make_gpu_thermos();
    auto gpu_reactions_temp = make_gpu_reactions();


    auto gpu_thermos = device_vector<gpuThermo>(gpu_thermos_temp.begin(), gpu_thermos_temp.end());
    auto gpu_reactions = device_vector<gpuReaction>(gpu_reactions_temp.begin(), gpu_reactions_temp.end());


    gpuODESystem gpu
    (
        cpu.nEqns(),
        gLabel(gpu_reactions.size()),
        get_raw_pointer(gpu_thermos),
        get_raw_pointer(gpu_reactions)
    );

    gLabel nSpecie = make_species_table().size();
    gLabel nEqns = cpu.nEqns();
    const gLabel li = 0;
    const gScalar time = 0.0;

    Foam::scalarField YTp_cpu(nEqns);
    fill_random(YTp_cpu);

    Foam::scalarField dYTpdt_cpu(nEqns);

    auto YTp_gpu = to_device_vec(YTp_cpu);
    auto dYTpdt_gpu = to_device_vec(dYTpdt_cpu);


    size_t n_evals = 20;

    BENCHMARK("CPU derivatives")
    {
        gScalar sum = 0.0;
        for (size_t i = 0; i < n_evals; ++i)
        {
            cpu.derivatives(time, YTp_cpu, li, dYTpdt_cpu);
            sum += dYTpdt_cpu[6];
        }
        return sum;
    };


    auto YTp = make_mdspan(YTp_gpu, extents<1>{nEqns});
    auto dYTpdt = make_mdspan(dYTpdt_gpu, extents<1>{nEqns});

    auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));
    auto f =
    [
        =,
        YTp = make_mdspan(YTp_gpu, extents<1>{nEqns}),
        dYTpdt = make_mdspan(dYTpdt_gpu, extents<1>{nEqns}),
        buffer = make_mdspan(buffer, extents<1>{1})
    ]()
    {
        gScalar sum = 0.0;
        for (size_t i = 0; i < n_evals; ++i)
        {
            gpu.derivatives(time, YTp, li, dYTpdt, buffer[0]);
            sum += dYTpdt[6];
        }
        return sum;
    };

    BENCHMARK("GPU derivatives")
    {
        eval(f);
    };


}



TEST_CASE("gpuODESystem::Jacobian")
{

    using namespace FoamGpu;

    Foam::MockOFSystem cpu;

    auto gpu_thermos_temp = make_gpu_thermos();
    auto gpu_reactions_temp = make_gpu_reactions();


    auto gpu_thermos = device_vector<gpuThermo>(gpu_thermos_temp.begin(), gpu_thermos_temp.end());
    auto gpu_reactions = device_vector<gpuReaction>(gpu_reactions_temp.begin(), gpu_reactions_temp.end());


    gpuODESystem gpu
    (
        cpu.nEqns(),
        gLabel(gpu_reactions.size()),
        get_raw_pointer(gpu_thermos),
        get_raw_pointer(gpu_reactions)
    );

    gLabel nSpecie = make_species_table().size();
    gLabel nEqns = cpu.nEqns();

    SECTION("jacobian")
    {
        const gLabel li = 0;

        const gScalar time = 0.1;

        Foam::scalarSquareMatrix J_cpu(nEqns, 0.1);
        device_vector<gScalar> J_gpu(J_cpu.size(), 0.2);

        Foam::scalarField YTp_cpu(nEqns, 0);
        Foam::scalarField dYTpdt_cpu(nEqns, 0);
        assign_test_condition(YTp_cpu);

        //This needs to be sme for both
        auto YTp_gpu = to_device_vec(YTp_cpu);
        auto dYTpdt_gpu = to_device_vec(dYTpdt_cpu);


        size_t n_evals = 10;

        BENCHMARK("CPU jacobian")
        {

            gScalar sum = 0.0;
            for (size_t i = 0; i < n_evals; ++i)
            {
                cpu.jacobian(time, YTp_cpu, li, dYTpdt_cpu, J_cpu);
                sum += J_cpu(3,3);
            }
            return sum;
        };



        auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));
        auto f =
        [
            =,
            YTp = make_mdspan(YTp_gpu, extents<1>{nEqns}),
            dYTpdt = make_mdspan(dYTpdt_gpu, extents<1>{nEqns}),
            J = make_mdspan(J_gpu, extents<2>{nEqns, nEqns}),
            buffer = make_mdspan(buffer, extents<1>{1})
        ]
        ()
        {
            gScalar sum = 0.0;
            for (size_t i = 0; i < n_evals; ++i)
            {
                gpu.jacobian(time, YTp, li, dYTpdt, J, buffer[0]);
                sum += J(3,3);
            }
            return sum;
        };

        BENCHMARK("GPU jacobian")
        {
            return  eval(f);
        };




    }
}

/*

TEST_CASE("gpuRosenBrock::solve() single cell")
{

    using namespace FoamGpu;

    Foam::MockOFSystem cpu_system;

    auto gpu_thermos_temp = make_gpu_thermos();
    auto gpu_reactions_temp = make_gpu_reactions();


    auto gpu_thermos = device_vector<gpuThermo>(gpu_thermos_temp.begin(), gpu_thermos_temp.end());
    auto gpu_reactions = device_vector<gpuReaction>(gpu_reactions_temp.begin(), gpu_reactions_temp.end());


    gpuODESystem gpu_system
    (
        cpu_system.nEqns(),
        gLabel(gpu_reactions.size()),
        get_raw_pointer(gpu_thermos),
        get_raw_pointer(gpu_reactions)
    );

    gLabel nSpecie = make_species_table().size();
    gLabel nEqns = cpu_system.nEqns();


    Foam::dictionary nulldict;
    Foam::Rosenbrock34 cpu(cpu_system, nulldict);
    gpuRosenbrock34<gpuODESystem> gpu = make_Rosenbrock34(gpu_system, nulldict);

    SECTION("solve(xStart, xEnd, y, li, dxTry) gri values")
    {
        const gScalar xStart = 0.;
        const gScalar xEnd = 1E-5; //1E-5;
        const gLabel li = 0;
        const gScalar dxTry = 1E-7;

        Foam::scalarField y_cpu(nEqns, 0.0);

        assign_test_condition(y_cpu);

        auto y_gpu = to_device_vec(y_cpu);

        device_vector<gScalar> J(nEqns*nEqns);

        auto buffer = to_device_vec(host_vector<gpuBuffer>(1, gpuBuffer(nSpecie)));
        auto f = [
            gpu = gpu,
            xStart = xStart,
            xEnd = xEnd,
            y = make_mdspan(y_gpu, extents<1>{nEqns}),
            li = li,
            dxTry = dxTry,
            J = make_mdspan(J, extents<2>{nEqns, nEqns}),
            buffer = make_mdspan(buffer, extents<1>{1})
        ]()
        {
            gScalar dxTry_temp = dxTry;
            gpu.solve(xStart, xEnd, y, li, dxTry_temp, J, buffer[0]);
            return dxTry_temp;
        };

        auto f2 =
        [
            &cpu=cpu,
            xStart = xStart,
            xEnd = xEnd,
            &y = y_cpu,
            li = li,
            dxTry = dxTry
        ]()
        {

            gScalar dxTry_temp = dxTry;
            cpu.solve(xStart, xEnd, y, li, dxTry_temp);
            return dxTry_temp;
        };


        BENCHMARK("CPU SINGLE CELL")
        {
            return f2();
        };

        BENCHMARK("GPU SINGLE CELL")
        {
            return eval(f);
        };

    }

}
*/