
#include "catch.H"
#include "scalar.H"
#include "numeric_matrix.H"
#include "scalarMatrices.H"
#include "LUscalarMatrix.H"
#include "LLTMatrix.H"
#include "QRMatrix.H"
#include <iomanip>
#include "test_utilities.H"
//#include <thrust/device_vector.h>
//#include <thrust/device_malloc_allocator.h>
#include <algorithm>

template<class T>
using allocator = std::allocator<T>;
//using allocator = thrust::device_malloc_allocator<T>;

using gpuScalarSquareMatrix
= topaz::NumericMatrix<Foam::scalar, allocator<Foam::scalar>>;

using NumericVector
= topaz::NumericArray<Foam::scalar, allocator<Foam::scalar>>;

using gpuLabelList
= topaz::NumericArray<Foam::label, allocator<Foam::label>>;

Foam::scalarSquareMatrix toHost(const gpuScalarSquareMatrix& m){


    using namespace Foam;
    scalarSquareMatrix ret(m.m(), m.n());
    std::copy(m.begin(), m.end(), ret.v());
    return ret;

}

Foam::scalarField toHost(const NumericVector& v)
{
    using namespace Foam;
    scalarField ret(v.size(),0);
    std::copy(v.begin(), v.end(), ret.begin());
    return ret;
}

Foam::scalarSquareMatrix toHost(const Foam::scalarSquareMatrix& m){
    return m;
}


template<class T>
void print(const T& m){

    auto m2 = toHost(m);

    for (auto i = 0; i < m2.m(); ++i){
        std::cout << std::endl;
        for (auto j = 0; j < m2.n(); ++j){

            Foam::scalar to_print = 0.0;
            if (std::abs(m2(i,j)) > 1E-5){
                to_print = m2(i,j);
            }

            std::cout << std::setw(8) << std::setprecision(3) << to_print;
        }
    }

    std::cout << std::endl;


}

TEST_CASE("Constructors"){

    REQUIRE_NOTHROW(gpuScalarSquareMatrix());

    gpuScalarSquareMatrix m1(3,4);
    CHECK(m1.m() == 3);
    CHECK(m1.n() == 4);
    CHECK(m1.size() == 3*4);


    REQUIRE_NOTHROW(gpuScalarSquareMatrix(m1));

    gpuScalarSquareMatrix m2(3,4, 3.0);

    CHECK
    (
        std::all_of(m2.begin(), m2.end(), [](auto val){return val == 3.0;})
    );


}

TEST_CASE("Assignment"){


    gpuScalarSquareMatrix m(3,2);
    NumericVector v1({1,2,3,4,5,6});

    m = v1;

    CHECK(m.m() == 3);
    CHECK(m.n() == 2);
    CHECK(m(0,0) == 1);
    CHECK(m(0,1) == 2);
    CHECK(m(1,0) == 3.0);
    REQUIRE_THROWS(m(4, 0));
    REQUIRE_THROWS(m(0, 2));

    NumericVector v2({1,2,3});
    REQUIRE_THROWS(m = v2);

    //CHECK(m == std::vector<Foam::scalar>{1,2,2,2});


}



std::pair<Foam::scalarSquareMatrix, Foam::labelList>
call_of_LUDecompose(const gpuScalarSquareMatrix& m)
{
    using namespace Foam;
    auto of_m = toHost(m);
    auto size = of_m.m();
    labelList pivots(size, 0);
    LUDecompose(of_m, pivots);
    return std::make_pair(of_m, pivots);

}


Foam::scalarField
call_of_LUBacksubstitute
(
    const gpuScalarSquareMatrix& m,
    const NumericVector& v
)
{
    using namespace Foam;
    auto [of_m, pivots] = call_of_LUDecompose(m);
    scalarField ret = toHost(v);
    Foam::LUBacksubstitute(of_m, pivots, ret);
    return ret;
}



TEST_CASE("LUDecompose"){

    using namespace Foam;

    SECTION("Test 1")
    {
        int size = 3;
        std::vector<int> vals(size*size, 0);
        std::iota(vals.begin(), vals.end(), 0);


        gpuScalarSquareMatrix ORG(size, size, scalar(0));
        std::copy(vals.begin(), vals.end(), ORG.begin());


        auto m_gpu = ORG;
        gpuLabelList list_gpu(size, 0);

        eval([=](){LUDecompose(m_gpu, list_gpu); return 0;});

        auto [m_cpu, list_cpu] = call_of_LUDecompose(ORG);



        std::vector<Foam::scalar> result_gpu(m_gpu.begin(), m_gpu.end());
        std::vector<Foam::scalar> result_cpu(m_cpu.v(), m_cpu.v() + size*size);


        CHECK(result_gpu == result_cpu);

    }

    SECTION("Test 2")
    {
        int size = 6;
        std::vector<int> vals(size*size, 0);
        std::iota(vals.begin(), vals.end(), 0);


        gpuScalarSquareMatrix ORG(size, size, scalar(0));
        std::copy(vals.begin(), vals.end(), ORG.begin());


        auto m_gpu = ORG;
        gpuLabelList list_gpu(size, 0);

        LUDecompose(m_gpu, list_gpu);

        auto [m_cpu, list_cpu] = call_of_LUDecompose(ORG);



        std::vector<Foam::scalar> result_gpu(m_gpu.begin(), m_gpu.end());
        std::vector<Foam::scalar> result_cpu(m_cpu.v(), m_cpu.v() + size*size);


        //print(m_cpu);
        //print(m_gpu);



        CHECK(result_gpu == result_cpu);

    }

}

TEST_CASE("LUBacksubstitute")
{
    using namespace Foam;

    SECTION("Test 1")
    {
        int size = 3;
        std::vector<int> vals(size*size, 0);

        std::iota(vals.begin(), vals.end(), 0);


        gpuScalarSquareMatrix ORG(size, size, scalar(0));
        std::copy(vals.begin(), vals.end(), ORG.begin());


        auto m_gpu = ORG;
        gpuLabelList list_gpu(size, 0);
        NumericVector source_gpu(size, 1);


        auto source_cpu = call_of_LUBacksubstitute(ORG, source_gpu);

        //Careful here, need to call of with source_gpu before
        // to ensure same source for both
        LUDecompose(m_gpu, list_gpu);
        LUBacksubstitute(m_gpu, list_gpu, source_gpu);



        std::vector<Foam::scalar> result_gpu(source_gpu.begin(), source_gpu.end());
        std::vector<Foam::scalar> result_cpu(source_cpu.begin(), source_cpu.end());


        CHECK(result_gpu == result_cpu);

    }

    SECTION("Test 2")
    {
        int size = 6;
        std::vector<int> vals(size*size, 0);

        std::iota(vals.begin(), vals.end(), 0);


        gpuScalarSquareMatrix ORG(size, size, scalar(0));
        std::copy(vals.begin(), vals.end(), ORG.begin());


        auto m_gpu = ORG;
        gpuLabelList list_gpu(size, 0);
        NumericVector source_gpu(size, 1);


        auto source_cpu = call_of_LUBacksubstitute(ORG, source_gpu);

        //Careful here, need to call of with source_gpu before
        // to ensure same source for both
        LUDecompose(m_gpu, list_gpu);
        LUBacksubstitute(m_gpu, list_gpu, source_gpu);



        std::vector<Foam::scalar> result_gpu(source_gpu.begin(), source_gpu.end());
        std::vector<Foam::scalar> result_cpu(source_cpu.begin(), source_cpu.end());


        CHECK(result_gpu == result_cpu);

    }

}


