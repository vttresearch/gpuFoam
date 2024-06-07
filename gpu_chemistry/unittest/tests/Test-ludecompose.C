#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "catch.H"
#include "mdspan.H"
#include "test_utilities.H"
#include "ludecompose.H"
#include "scalarMatrices.H"

#include <tuple>
#include <vector>

//#include "gsl_functions.hpp"

inline auto call_lu_gpu(const std::vector<gScalar>& m_vals, const std::vector<gScalar>& s_vals)
{

    gLabel size = std::sqrt(m_vals.size());

    device_vector<gScalar> matrix(m_vals.begin(), m_vals.end());
    device_vector<gLabel> pivot(size, 0);
    device_vector<gScalar> v(size, 0);
    device_vector<gScalar> source(s_vals.begin(), s_vals.end());

    auto m_span = make_mdspan(matrix, extents<2>{size, size});
    auto p_span = make_mdspan(pivot, extents<1>{size});
    auto v_span = make_mdspan(v, extents<1>{size});

    eval
    (
        [=](){FoamGpu::LUDecompose(m_span, p_span, v_span); return 0;}
    );

    auto s_span = make_mdspan(source, extents<1>{size});

    eval
    (
        [=](){FoamGpu::LUBacksubstitute(m_span, p_span, s_span); return 0;}
    );


    auto m_ret = toStdVector(matrix);
    auto p_ret = toStdVector(pivot);
    auto s_ret = toStdVector(source);

    return std::make_tuple(m_ret, p_ret, s_ret);

}




inline auto call_lu_cpu(const std::vector<gScalar>& m_vals, const std::vector<gScalar> s_vals)
{

    gLabel size = std::sqrt(m_vals.size());

    Foam::scalarSquareMatrix matrix(size, 0);
    std::copy(m_vals.begin(), m_vals.end(), matrix.v());
    Foam::List<Foam::label> pivot(size, 0);
    Foam::scalarField source(size);
    std::copy(s_vals.begin(), s_vals.end(), source.begin());

    gLabel sign;
    Foam::LUDecompose(matrix, pivot, sign);
    Foam::LUBacksubstitute(matrix, pivot, source);


    auto m_ret = std::vector<gScalar>(matrix.v(), matrix.v() + size*size);
    auto p_ret = std::vector<gLabel>(pivot.begin(), pivot.end());
    auto s_ret = std::vector<gScalar>(source.begin(), source.end());

    return std::make_tuple(m_ret, p_ret, s_ret);

}



TEST_CASE("Test ludecompose")
{
    using namespace FoamGpu;

    for (int i = 3; i < 50; ++i)
    {
        int size = i;


        std::vector<gScalar> vals(size*size);
        fill_random(vals);
        std::vector<gScalar> source(size, 1);


        auto [m_gpu, p_gpu, s_gpu] = call_lu_gpu(vals, source);
        auto [m_cpu, p_cpu, s_cpu] = call_lu_cpu(vals, source);

        REQUIRE_THAT
        (
            m_gpu,
            Catch::Matchers::Approx(m_cpu).epsilon(errorTol)
        );
        REQUIRE_THAT
        (
            p_gpu,
            Catch::Matchers::Approx(p_cpu).epsilon(errorTol)
        );
        REQUIRE_THAT
        (
            s_gpu,
            Catch::Matchers::Approx(s_cpu).epsilon(errorTol)
        );

    }
}









/*

TEST_CASE("Test gsl tutorial constant pivot")
{
    const std::vector<gScalar> m_vals = { 0.18, 0.60, 0.57, 0.96,
                        0.41, 0.24, 0.99, 0.58,
                        0.14, 0.30, 0.97, 0.66,
                        0.51, 0.13, 0.19, 0.85 };

    gLabel size = std::sqrt(m_vals.size());


    Foam::scalarSquareMatrix foam_matrix(size, 0);
    std::copy(m_vals.begin(), m_vals.end(), foam_matrix.v());
    Foam::List<Foam::label> foam_pivot(size, 0);
    gLabel sign;
    Foam::LUDecompose(foam_matrix, foam_pivot, sign);

    std::vector<gScalar> gsl_matrix(m_vals.begin(), m_vals.end());
    gsl_permutation *gsl_pivot = gsl_permutation_alloc(size);
    GSL::LUDecompose(gsl_matrix, gsl_pivot);

    std::vector<gScalar> gsl_solution(size);

    SECTION("Test 1"){
        const std::vector<gScalar> s_vals(size, 1);

        Foam::scalarField foam_source(size);
        std::copy(s_vals.begin(), s_vals.end(), foam_source.begin());

        std::vector<gScalar> gsl_source(s_vals.begin(), s_vals.end());



        Foam::LUBacksubstitute(foam_matrix, foam_pivot, foam_source);
        GSL::LUBacksubstitute(gsl_matrix, gsl_pivot, gsl_source, gsl_solution);


        REQUIRE_THAT
        (
            gsl_solution,
            Catch::Matchers::Approx(toStdVector(foam_source)).epsilon(errorTol)
        );


    }

    SECTION("Test 2"){
        const std::vector<gScalar> s_vals = [&](){

            std::vector<gScalar> ret(size);
            fill_random(ret);
            return ret;

        }();

        Foam::scalarField foam_source(size);
        std::copy(s_vals.begin(), s_vals.end(), foam_source.begin());

        std::vector<gScalar> gsl_source(s_vals.begin(), s_vals.end());



        Foam::LUBacksubstitute(foam_matrix, foam_pivot, foam_source);
        GSL::LUBacksubstitute(gsl_matrix, gsl_pivot, gsl_source, gsl_solution);


        REQUIRE_THAT
        (
            gsl_solution,
            Catch::Matchers::Approx(toStdVector(foam_source)).epsilon(errorTol)
        );

    }


}

TEST_CASE("Benchmark gsl"){

    SECTION("N = 10"){
        int size = 10;

        const std::vector<gScalar> m_vals = [=](){
            std::vector<gScalar> ret(size*size);
            fill_random(ret);
            return ret;
        }();

        const std::vector<gScalar> s_vals(size, 1);

        BENCHMARK_ADVANCED("gsl")(Catch::Benchmark::Chronometer meter) {
            std::vector<gScalar> matrix(m_vals.begin(), m_vals.end());
            gsl_permutation *pivot= gsl_permutation_alloc(size);
            std::vector<gScalar> result(size);
            std::vector<gScalar> source(s_vals.begin(), s_vals.end());
            meter.measure([&] {
                GSL::LUDecompose(matrix, pivot);
                GSL::LUBacksubstitute(matrix, pivot, source, result);
                return result[0] + result[4];
            });
            gsl_permutation_free(pivot);
        };


        BENCHMARK_ADVANCED("foamGpu")(Catch::Benchmark::Chronometer meter) {
            std::vector<gScalar> matrix(m_vals.begin(), m_vals.end());
            std::vector<gLabel> pivot(size);
            std::vector<gScalar> v(size);
            std::vector<gScalar> source(s_vals.begin(), s_vals.end());
            meter.measure([&] {
                auto m_span = make_mdspan(matrix, extents<2>{size, size});
                auto p_span = make_mdspan(pivot, extents<1>{size});
                auto v_span = make_mdspan(v, extents<1>{size});
                auto s_span = make_mdspan(source, extents<1>{size});
                FoamGpu::LUDecompose(m_span, p_span, v_span);
                FoamGpu::LUBacksubstitute(m_span, p_span, s_span);
                return source[0] + source[4];
            });
        };



        BENCHMARK_ADVANCED("of")(Catch::Benchmark::Chronometer meter) {
            Foam::scalarSquareMatrix matrix(size, 0);
            std::copy(m_vals.begin(), m_vals.end(), matrix.v());

            Foam::List<Foam::label> pivot(size, 0);
            Foam::scalarField source(size);
            std::copy(s_vals.begin(), s_vals.end(), source.begin());

            meter.measure([&] {
                gLabel sign;
                Foam::LUDecompose(matrix, pivot, sign);
                Foam::LUBacksubstitute(matrix, pivot, source);
                return source[0] + source[4];
            });

        };


    }

    SECTION("N = 50"){
        int size = 50;

        const std::vector<gScalar> m_vals = [=](){
            std::vector<gScalar> ret(size*size);
            fill_random(ret);
            return ret;
        }();

        const std::vector<gScalar> s_vals(size, 1);

        BENCHMARK_ADVANCED("gsl")(Catch::Benchmark::Chronometer meter) {
            std::vector<gScalar> matrix(m_vals.begin(), m_vals.end());
            gsl_permutation *pivot= gsl_permutation_alloc(size);
            std::vector<gScalar> result(size);
            std::vector<gScalar> source(s_vals.begin(), s_vals.end());
            meter.measure([&] {
                GSL::LUDecompose(matrix, pivot);
                GSL::LUBacksubstitute(matrix, pivot, source, result);
                return result[0] + result[4];
            });
            gsl_permutation_free(pivot);
        };


        BENCHMARK_ADVANCED("foamGpu")(Catch::Benchmark::Chronometer meter) {
            std::vector<gScalar> matrix(m_vals.begin(), m_vals.end());
            std::vector<gLabel> pivot(size);
            std::vector<gScalar> v(size);
            std::vector<gScalar> source(s_vals.begin(), s_vals.end());
            meter.measure([&] {
                auto m_span = make_mdspan(matrix, extents<2>{size, size});
                auto p_span = make_mdspan(pivot, extents<1>{size});
                auto v_span = make_mdspan(v, extents<1>{size});
                auto s_span = make_mdspan(source, extents<1>{size});
                FoamGpu::LUDecompose(m_span, p_span, v_span);
                FoamGpu::LUBacksubstitute(m_span, p_span, s_span);
                return source[0] + source[4];
            });
        };



        BENCHMARK_ADVANCED("of")(Catch::Benchmark::Chronometer meter) {
            Foam::scalarSquareMatrix matrix(size, 0);
            std::copy(m_vals.begin(), m_vals.end(), matrix.v());

            Foam::List<Foam::label> pivot(size, 0);
            Foam::scalarField source(size);
            std::copy(s_vals.begin(), s_vals.end(), source.begin());

            meter.measure([&] {
                gLabel sign;
                Foam::LUDecompose(matrix, pivot, sign);
                Foam::LUBacksubstitute(matrix, pivot, source);
                return source[0] + source[4];
            });

        };


    }

    SECTION("N = 100"){
        int size = 100;

        const std::vector<gScalar> m_vals = [=](){
            std::vector<gScalar> ret(size*size);
            fill_random(ret);
            return ret;
        }();

        const std::vector<gScalar> s_vals(size, 1);

        BENCHMARK_ADVANCED("gsl")(Catch::Benchmark::Chronometer meter) {
            std::vector<gScalar> matrix(m_vals.begin(), m_vals.end());
            gsl_permutation *pivot= gsl_permutation_alloc(size);
            std::vector<gScalar> result(size);
            std::vector<gScalar> source(s_vals.begin(), s_vals.end());
            meter.measure([&] {
                GSL::LUDecompose(matrix, pivot);
                GSL::LUBacksubstitute(matrix, pivot, source, result);
                return result[0] + result[4];
            });
            gsl_permutation_free(pivot);
        };


        BENCHMARK_ADVANCED("foamGpu")(Catch::Benchmark::Chronometer meter) {
            std::vector<gScalar> matrix(m_vals.begin(), m_vals.end());
            std::vector<gLabel> pivot(size);
            std::vector<gScalar> v(size);
            std::vector<gScalar> source(s_vals.begin(), s_vals.end());
            meter.measure([&] {
                auto m_span = make_mdspan(matrix, extents<2>{size, size});
                auto p_span = make_mdspan(pivot, extents<1>{size});
                auto v_span = make_mdspan(v, extents<1>{size});
                auto s_span = make_mdspan(source, extents<1>{size});
                FoamGpu::LUDecompose(m_span, p_span, v_span);
                FoamGpu::LUBacksubstitute(m_span, p_span, s_span);
                return source[0] + source[4];
            });
        };



        BENCHMARK_ADVANCED("of")(Catch::Benchmark::Chronometer meter) {
            Foam::scalarSquareMatrix matrix(size, 0);
            std::copy(m_vals.begin(), m_vals.end(), matrix.v());

            Foam::List<Foam::label> pivot(size, 0);
            Foam::scalarField source(size);
            std::copy(s_vals.begin(), s_vals.end(), source.begin());

            meter.measure([&] {
                gLabel sign;
                Foam::LUDecompose(matrix, pivot, sign);
                Foam::LUBacksubstitute(matrix, pivot, source);
                return source[0] + source[4];
            });

        };


    }

}
*/


