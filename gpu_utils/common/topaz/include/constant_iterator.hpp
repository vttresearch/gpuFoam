#pragma once

#ifdef __NVIDIA_COMPILER__
#include <thrust/iterator/constant_iterator.h>
#else
#include <boost/iterator/iterator_facade.hpp>
#endif

namespace topaz {

#ifdef __NVIDIA_COMPILER__
template <class T>
using constant_iterator = thrust::constant_iterator<T>;
#else

template <class T>
class constant_iterator;

template <class T>
using const_iterator_base =
    boost::iterator_facade<constant_iterator<T>,               // derived type
                           T,                                  // value type
                           boost::random_access_traversal_tag, // access tag
                           T,        // reference type !!!
                           ptrdiff_t // difference_type
                           >;
///
///@brief An iterator which returns a constant value when dereferenced. This
///compiles with both nvcc and gcc but is painfully slow with nvcc.
///Note! reference_type = value_type
///
///@tparam T value_type to be returned
///
template <class T>
class constant_iterator : public const_iterator_base<T> {
public:
    using parent          = const_iterator_base<T>;
    using value_type      = typename parent::value_type;
    using reference       = typename parent::reference;
    using difference_type = typename parent::difference_type;

    inline CUDA_HOSTDEV constant_iterator(const T&        value,
                                          difference_type index = 0)
        : m_value(value)
        , m_index(index) {}

private:
    friend class boost::iterator_core_access;

    inline CUDA_HOSTDEV reference dereference() const { return m_value; }

    inline CUDA_HOSTDEV bool equal(const constant_iterator<T>& other) const {
        return m_value == other.m_value && m_index == other.m_index;
    }

    inline CUDA_HOSTDEV void increment() { m_index++; }

    inline CUDA_HOSTDEV void decrement() { m_index--; }

    inline CUDA_HOSTDEV void advance(difference_type n) { m_index += n; }

    inline CUDA_HOSTDEV difference_type
    distance_to(const constant_iterator<T>& other) const {
        return static_cast<difference_type>(other.m_index - m_index);
    }

private:
    T               m_value;
    difference_type m_index;
};
#endif

} // namespace topaz