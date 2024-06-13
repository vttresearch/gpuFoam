#pragma once

#include <cstdlib> //rand


static inline double random_number(double LO, double HI){
    double r = LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI-LO)));
    return r;
}

template<class T>
static inline void fill_random(T& v, double LO = 0.0, double HI = 1.0)
{


    for (auto& e : v)
    {
        e = random_number(LO, HI);
    }

}

template<class T>
static inline void fill_linear(T& v){

    double dx = 1.0 / v.size();

    for (int i = 0; i < static_cast<int>(v.size()); ++i){
        v[i] = dx*i + dx;
    }



}

template<class C>
static inline void remove_negative(C& c, double tolerance){

    for (auto& e : c){
        if (e < tolerance){
            e = double(0);
        }
    }

}














