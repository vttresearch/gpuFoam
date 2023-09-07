#pragma once

#include <tuple> //std::tuple

#ifdef __NVIDIA_COMPILER__
#include <thrust/tuple.h>
#else
#include "boost/tuple/tuple.hpp"
#include <boost/fusion/include/tuple.hpp>
#endif



namespace topaz {


    #ifdef __NVIDIA_COMPILER__

        template<class... Types>
        using Tuple = thrust::tuple<Types...>;

        using thrust::get;
        //using thurst::tuple_size;

        template<class T>
        using tuple_size = thrust::tuple_size<T>;

        template< class... Types >
        inline constexpr CUDA_HOSTDEV
        auto adl_make_tuple( Types&&... args ) {
            return thrust::make_tuple(std::forward<Types>(args)...);
        }

    #else

        template<class... Types>
        using Tuple = boost::tuple<Types...>;

        template<class T>
        using tuple_size = boost::fusion::tuple_size<T>;
        //using boost::fusion::tuple_size;
        using boost::get;

        template< class... Types >
        inline constexpr auto adl_make_tuple( Types&&... args ) {
            return boost::make_tuple(std::forward<Types>(args)...);
        }

    #endif

namespace detail {

template <class Tuple_t, size_t... Is>
inline constexpr auto
to_std_tuple_impl(std::index_sequence<Is...>, const Tuple_t& tpl) {
    return std::make_tuple(get<Is>(tpl)...);
}

} // namespace detail

template <class Tuple_t>
inline constexpr auto to_std_tuple(const Tuple_t& tpl) {
    constexpr size_t N = tuple_size<Tuple_t>::value;
    return detail::to_std_tuple_impl(std::make_index_sequence<N>{}, tpl);
}


} // namespace topaz
