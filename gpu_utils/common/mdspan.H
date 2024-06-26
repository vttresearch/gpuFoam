#pragma once

#include "mdspan_impl.hpp"
#include "cuda_host_dev.H"
#include "pointer_casts.hpp"


//namespace stdex = std::experimental;
namespace stdex = std;



template <std::size_t N> using extents = stdex::dextents<std::size_t, N>;

template <class ElementType, size_t N, class Layout>
using mdspan_base = stdex::mdspan<ElementType, extents<N>, Layout>;

template <class ElementType, size_t N>
using mdspan = mdspan_base<ElementType, N, stdex::layout_right>;


/*

/// @brief Makes a multi-dimensional span of the input container
/// @tparam Container a container which has a value_type, size() and data()
/// members
/// @param c the input container
/// @param dims dimensions of the multi-dimensional span
/// @return a multi-dimensional span
template<size_t N, class Container>
static constexpr auto make_mdspan(Container& c, extents<N>& dims) {
    using value_type = typename Container::value_type;
    //extents<dims.size()> ext        = make_extents(dims);
    //runtime_assert(flat_size(ext) == std::size(c),
    //               "Dimension mismatch in make_span");
    return mdspan<value_type, extents.rank()>(std::data(c), dims);
}
*/

/// @brief Computes the total element count spanned by the extents
/// @param ext (array-like) extensions to compute the flat size of
/// @return size_t the flat size
template <class Extents>
static CUDA_HOSTDEV std::size_t flat_size(Extents ext) {

    std::size_t ret(1);
    for (size_t i = 0; i < ext.rank(); ++i) { ret *= ext.extent(i); }
    return ret;
}




/// @brief Makes a multi-dimensional span of the input container
/// @tparam Container a container which has a value_type, size() and data()
/// members
/// @param c the input container
/// @param dims dimensions of the multi-dimensional span
/// @return a multi-dimensional span
template <class Container, class Extents>
static constexpr HOST auto make_mdspan(Container& c, Extents ext) {
    using value_type = typename Container::value_type;
    //topaz::runtime_assert(flat_size(ext) == std::size(c),
    //               "Dimension mismatch in make_mdspan");
    return mdspan<value_type, ext.rank()>(make_raw_pointer(c.data()), ext);
}

/// @brief Makes a multi-dimensional span of the input container
/// @tparam Container a container which has a value_type, size() and data()
/// members
/// @param c the input container
/// @param dims dimensions of the multi-dimensional span
/// @return a multi-dimensional span
template <class Container, class Extents>
static constexpr HOST auto make_mdspan(const Container& c, Extents ext) {
    using value_type = const typename Container::value_type;
    //topaz::runtime_assert(flat_size(ext) == std::size(c),
    //               "Dimension mismatch in make_mdspan");
    return mdspan<value_type, ext.rank()>(make_raw_pointer(c.data()), ext);
}
