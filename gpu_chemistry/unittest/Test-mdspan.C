#include "catch.H"
#include "mdspan.H"



TEST_CASE("make_mdspan"){


    std::vector<int> v = {1,2,3,4,5,6};

    auto s = make_mdspan(v, extents<2>(2,3));

    CHECK(s(0,0) == 1);

    //REQUIRE_THROWS(make_mdspan(v, extents<2>(2,1)));

    CHECK(std::size(s) == 2*3);

}