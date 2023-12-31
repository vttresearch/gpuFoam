#include "catch.H"
#include "gpuMemoryResource.H"
#include "test_utilities.H"


TEST_CASE("memoryResource"){
    using namespace FoamGpu;

    using MR_t = memoryResource_t;

    SECTION("Constructors")
    {
        REQUIRE_NOTHROW(MR_t());

        REQUIRE_NOTHROW(MR_t(10, 1));
    }

    SECTION("resize"){

        MR_t m(10, 3);
        REQUIRE_NOTHROW(m.resize(12, 1));
        REQUIRE_NOTHROW(m.resize(0, 0));
        REQUIRE_NOTHROW(m.resize(3, 5));
        CHECK(m.nCells() == 3);
        CHECK(m.nEqns() == 7);
        CHECK(m.nSpecie() == 5);
    }

    SECTION("splitToBuffers")
    {

        gLabel nCells = 3;
        gLabel nSpecie = 6;
        gLabel nEqns = nSpecie +2;
        MR_t mr(nCells, nSpecie);

        auto buffers_arr = toDeviceVector(splitToBuffers(mr));

        CHECK(gLabel(buffers_arr.size()) == nCells);



        auto f = [
            nCells = nCells,
            nEqns = nEqns,
            buffers = make_mdspan(buffers_arr, extents<1>{nCells})
        ]()
        {

            for (gLabel i = 0; i < nCells; ++i)
            {
                for (gLabel j = 0; j < nEqns; ++j)
                {
                    buffers[i].pivotIndices()[j] = i;
                    buffers[i].dydx0()[j] = gScalar(i);
                    buffers[i].yTemp()[j] = gScalar(i);
                    buffers[i].dydx()[j] = gScalar(i);
                    buffers[i].dfdx()[j] = gScalar(i);
                    buffers[i].k1()[j] = gScalar(i);
                    buffers[i].k2()[j] = gScalar(i);
                    buffers[i].k3()[j] = gScalar(i);
                    buffers[i].k4()[j] = gScalar(i);
                    buffers[i].err()[j] = gScalar(i);
                    buffers[i].lubuffer()[j] = gScalar(i);
                    buffers[i].c()[j] = gScalar(i);
                    buffers[i].tempField1()[j] = gScalar(i);
                    buffers[i].tempField2()[j] = gScalar(i);
                }
            }
            return buffers[0].pivotIndices()[2];
        };

        CHECK(eval(f) == 0);

    }

}
