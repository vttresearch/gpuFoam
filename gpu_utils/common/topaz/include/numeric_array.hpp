#pragma once

#include "constant_range.hpp"
#include "range.hpp"
#include "traits.hpp"
#include "transform.hpp"

#ifdef __NVIDIA_COMPILER__
#include <thrust/detail/vector_base.h>
#else
#include <vector>
#endif

namespace topaz {

#ifdef __NVIDIA_COMPILER__
template <class T, class Allocator>
using vector_base_type = thrust::detail::vector_base<T, Allocator>;
#else
// TODO: This is very bad, make some own base type
template <class T, class Allocator>
using vector_base_type = std::vector<T, Allocator>;
#endif

template <class T, class Allocator>
struct NumericArray : public vector_base_type<T, Allocator> {

private:
    using parent = vector_base_type<T, Allocator>;

public:
    using size_type  = typename parent::size_type;
    using value_type = typename parent::value_type;

    static constexpr bool is_numeric_vector = true;

    inline NumericArray() = default;

    inline explicit NumericArray(size_type n)
        : parent(n) {}

    inline explicit NumericArray(size_type n, const value_type& value)
        : parent(n, value) {}

    template <class Iterator,
              typename = std::enable_if_t<IsIterator_v<Iterator>>>
    inline explicit NumericArray(Iterator begin, Iterator end)
        : parent(begin, end) {}

    inline explicit NumericArray(std::initializer_list<T> l)
        : parent(l.size()) {
        this->assign(l.begin(), l.end());
    }

    // TODO: this should maybe be marked explicit as well
    template <class Range_t,
              typename = std::enable_if_t<IsRangeOrNumericArray_v<Range_t>>>
    inline NumericArray(const Range_t& rng)
        : parent(adl_begin(rng), adl_end(rng)) {}

    template <class Range_t,
              typename = std::enable_if_t<IsRangeOrNumericArray_v<Range_t>>>
    inline NumericArray& operator=(const Range_t& rng) {
        // TODO: This should probably be calling some parallel algorithm for std
        // C++17
        this->assign(adl_begin(rng), adl_end(rng));
        return *this;
    }
};

} // namespace topaz