#include "catch.H"
#include "gpuMemoryResource.H"



TEST_CASE("gpuMemoryResource"){
    using namespace FoamGpu;

    using MR_t = gpuMemoryResource<std::allocator<gLabel>, std::allocator<gScalar>>;

    SECTION("Constructors")
    {
        REQUIRE_NOTHROW(MR_t());

        REQUIRE_NOTHROW(MR_t(10, 1));
    }



}