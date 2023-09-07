#pragma once

#include "begin_end.hpp"


namespace topaz {

template <typename Iterator>
class Range {
public:

    using iterator   = Iterator;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference  = typename std::iterator_traits<Iterator>::reference;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;

    static constexpr bool is_range = true;

    inline CUDA_HOSTDEV Range(iterator first, iterator last)
        : m_begin(first)
        , m_end(last) {}

    template <class Range_t>
    inline CUDA_HOSTDEV Range(Range_t& rng)
        : Range(adl_begin(rng), adl_end(rng)) {}

    template <class Range_t>
    inline CUDA_HOSTDEV Range(const Range_t& rng)
        : Range(adl_begin(rng), adl_end(rng)) {}

    inline CUDA_HOSTDEV iterator begin() const { return m_begin; }
    inline CUDA_HOSTDEV iterator begin() { return m_begin; }

    inline CUDA_HOSTDEV iterator end() const { return m_end; }
    inline CUDA_HOSTDEV iterator end()  { return m_end; }

    inline CUDA_HOSTDEV difference_type size() const { return end() - begin(); }

    inline CUDA_HOSTDEV bool empty() const { return begin() == end(); }

    inline CUDA_HOSTDEV reference operator[](const difference_type& i) const {
        //return static_cast<value_type>(begin()[i]);
        return begin()[i];
        //return static_cast<value_type>(begin()[i].value());
    }

private:
    iterator m_begin, m_end;
};


template <typename Iterator>
CUDA_HOSTDEV auto size(const Range<Iterator>& rng) {
    return rng.size();
}

template <typename Iterator>
CUDA_HOSTDEV auto make_range(Iterator first, Iterator last) {
    return Range<Iterator>(first, last);
}

template <class Range_t>
CUDA_HOSTDEV auto make_range(Range_t& rng) {
    using iterator = decltype(std::begin(rng));
    return Range<iterator>::type > (rng);
}

template <class Range_t>
CUDA_HOSTDEV auto make_range(const Range_t& rng) {
    using iterator = decltype(std::begin(rng));
    return Range<iterator>::type > (rng);
}

template <class Range_t, typename Size>
CUDA_HOSTDEV auto slice(Range_t& rng, Size first, Size last) {
    return make_range(adl_begin(rng) + first, adl_begin(rng) + last);
}

template <class Range_t, typename Size>
CUDA_HOSTDEV auto slice(const Range_t& rng, Size first, Size last) {
    return make_range(adl_begin(rng) + first, adl_begin(rng) + last);
}

template <class Range_t, typename Size>
CUDA_HOSTDEV auto take(Range_t& rng, Size n) {
    return slice(rng, Size(0), n);
}

template <class Range_t, typename Size>
CUDA_HOSTDEV auto take(const Range_t& rng, Size n) {
    return slice(rng, Size(0), n);
}

} // namespace topaz