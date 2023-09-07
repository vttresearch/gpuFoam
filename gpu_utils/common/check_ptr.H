#pragma once
//#include <string_view>
//#include <string.h>
#include <stdio.h>
//#include "cuda_host_dev.H"


#define check_ptr(val, name)  if (!val) {printf("null ptr %s", name); assert(0);}

/*
template<class T>
static inline CUDA_HOSTDEV void check_ptr(T ptr, std::string_view name)
{
    //Note string view may not be null terminated and this is dangerous
    if (!ptr)
    {
        printf("Bad alloc for: %s \n", name.data());
    }

}
*/