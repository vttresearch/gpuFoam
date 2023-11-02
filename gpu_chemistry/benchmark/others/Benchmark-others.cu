#define CATCH_CONFIG_ENABLE_BENCHMARKING
//#define CATCH_CONFIG_MAIN
#include "catch.H"

#include "host_device_vectors.H"
#include "test_utilities.H"
#include "ludecompose.H"

TEST_CASE("LU")
{

    int size = 30;
    int n_times = 20;
    SECTION("LUdecompose")
    {

        std::vector<gScalar> vals(size*size);


        fill_random(vals);

        device_vector<gScalar> matrix(vals.begin(), vals.end());

        device_vector<gLabel> pivot(size, 0);
        device_vector<gScalar> v(size, 0);
        device_vector<gScalar> source(size, 0);

        auto m_span = make_mdspan(matrix, extents<2>{size, size});
        auto p_span = make_mdspan(pivot, extents<1>{size});
        auto v_span = make_mdspan(v, extents<1>{size});
        auto s_span = make_mdspan(source, extents<1>{size});

        auto op = [=] __device__() {
            gLabel ret = 0;

            for (int i = 0; i < n_times; ++i) {
                FoamGpu::LUDecompose(m_span, p_span, v_span);

                for(int j = 0; j < size; ++j){
                    ret += p_span(j);
                }

            }

            return ret;
        };

        auto op2 = [=] __device__ (){
            gScalar ret = 0.0;

            for (int i = 0; i < n_times; ++i) {

                FoamGpu::LUBacksubstitute(m_span, p_span, s_span);

                for(int j = 0; j < size; ++j){
                    ret += s_span(j);
                }

            }
            return ret;

        };


        BENCHMARK("WARMUP"){
            return eval(op);
        };

        BENCHMARK("LUdecompose"){

            return eval(op);
        };

        BENCHMARK("LUbacksubstitute"){

            return eval(op2);
        };


    }

}

/*
TEST_CASE("Jacobian"){





}
*/