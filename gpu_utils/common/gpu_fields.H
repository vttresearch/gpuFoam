#pragma once

#include <vector>
#include "scalar.H"
#include "volFields.H"

#include "topaz.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#ifdef __NVIDIA__COMPILER__

using gpuScalarField
= topaz::NumericArray<Foam::scalar, thrust::device_allocator<Foam::scalar>>;

using gpuVectorField
= topaz::NumericArray<Foam::vector, thrust::device_allocator<Foam::vector>>;
#else
using gpuScalarField = topaz::NumericArray<Foam::scalar, std::allocator<Foam::scalar>>;
using gpuVectorField = topaz::NumericArray<Foam::vector, std::allocator<Foam::vector>>;

#endif

namespace topaz{

//Expand topaz with mag(field)...
struct Mag {

    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& t) const
        -> decltype(Foam::mag(t)) {
        return Foam::mag(t);
    }
};


template <class T, typename = std::enable_if_t<IsRangeOrNumericArray_v<T>>>
inline CUDA_HOSTDEV auto mag(const T& t) {
    return transform(t, Mag{});
}

} //namespace topaz

namespace Foam{

static inline gpuScalarField toDevice(const volScalarField& f)
{
    volScalarField::Internal temp = f;
    gpuScalarField ret(temp.size());
    topaz::copy(temp, ret);
    return ret;
}
static inline void toHost(const gpuScalarField& from, volScalarField& to)
{
    thrust::host_vector<scalar> temp(from.begin(), from.end());
    topaz::copy
    (
        temp,
        to.primitiveFieldRef()
    );
}

static inline gpuVectorField toDevice(const volVectorField& f)
{
    volVectorField::Internal temp = f;
    gpuVectorField ret(temp.size());
    topaz::copy(temp, ret);
    return ret;
}

} //namespace Foam