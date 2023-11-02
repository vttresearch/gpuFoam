#include "catch.H"
#include "mdspan.H"
#include "timer.H"
#include "test_utilities.H"

TEST_CASE("make_mdspan"){


    std::vector<int> v = {1,2,3,4,5,6};

    auto s = make_mdspan(v, extents<2>(2,3));

    CHECK(s(0,0) == 1);

    //REQUIRE_THROWS(make_mdspan(v, extents<2>(2,1)));

    CHECK(s.size() == 2*3);

}


TEST_CASE("Test timer"){

    SECTION("Constructors"){

        REQUIRE_NOTHROW(Timer());




        

        auto op = []() {
            Timer t;
            t.start("ASD");
            
            t.stop("ASD");
            t.print();
            return 0.0;

        };


        eval(op);


        


    }



}

