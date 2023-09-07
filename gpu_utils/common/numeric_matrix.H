#pragma once

#include "scalar.H"
#include "ludecompose.H"
//#include "gpu_fields.H"

#include "mdspan.H"

namespace topaz{

template<class T, class Allocator>
struct NumericMatrix : private vector_base_type<T, Allocator>{

private:
    using parent = vector_base_type<T, Allocator>;


public:

    using iterator   = typename parent::iterator;
    using size_type  = typename parent::size_type;
    using value_type = typename parent::value_type;
    using reference  = typename parent::reference;
    using difference_type
        = typename std::iterator_traits<iterator>::difference_type;

    using pointer = typename parent::pointer;


    static constexpr bool is_numeric_vector = false;

    inline NumericMatrix() = default;

    inline explicit NumericMatrix(size_type m, size_type n)
        : parent(m*n), m_mrows(m), m_ncols(n) {}

    inline CUDA_HOSTDEV explicit NumericMatrix(size_type m, size_type n, T val)
        : parent(m*n, val), m_mrows(m), m_ncols(n) {}

    template <class Range_t,
              typename = std::enable_if_t<IsRangeOrNumericArray_v<Range_t>>>
    inline NumericMatrix& operator=(const Range_t& rng) {
        runtime_assert(adl_size(rng) == adl_size(*this),
            "Invalid size in matrix assignment.");

        // TODO: This should probably be calling some parallel algorithm for std
        // C++17
        this->assign(adl_begin(rng), adl_end(rng));
        return *this;
    }

    inline CUDA_HOSTDEV
    size_type m() const{return m_mrows;}

    inline CUDA_HOSTDEV
    size_type n() const{return m_ncols;}

    using parent::size;
    using parent::begin;
    using parent::end;
    using parent::data; //TODO: maybe requires a raw pointer cast for device

    inline CUDA_HOSTDEV
    reference operator()(const difference_type i, const difference_type j)
    {
        runtime_assert(i < difference_type(m_mrows), "Row index out of bounds");
        runtime_assert(i >= 0, "Row index out of bounds");
        runtime_assert(j < difference_type(m_ncols), "Col index out of bounds");
        runtime_assert(j >= 0, "Col index out of bounds");
        return parent::operator[](i*m_ncols + j);
        //return (this->data() + difference_type(i * m_ncols + j));
    }

    inline CUDA_HOSTDEV
    const T& operator()(const difference_type i, const difference_type j) const
    {
        runtime_assert(i < difference_type(m_mrows), "Row index out of bounds");
        runtime_assert(i >= 0, "Row index out of bounds");
        runtime_assert(j < difference_type(m_ncols), "Col index out of bounds");
        runtime_assert(j >= 0, "Col index out of bounds");
        return parent::operator[](i*m_ncols + j);
        //return (this->data() + difference_type(i * m_ncols + j));
    }

    inline CUDA_HOSTDEV auto operator[](const difference_type i)
    {
        runtime_assert(i < difference_type(m_mrows), "Row index out of bounds");
        runtime_assert(i >= 0, "Row index out of bounds");
        //return thrust::raw_pointer_cast(this->data() + difference_type(i * m_ncols));
        return this->data() + difference_type(i * m_ncols);
    }


private:
    size_type m_mrows, m_ncols;

};




template<class ET, class Allocator, class IdxArray>
static inline void LUDecompose
(
    NumericMatrix<ET, Allocator>& matrix,
    IdxArray& pivotIndices
)
{
    using namespace Foam;
    label m = matrix.m();
    std::vector<scalar> vv(m);

    auto m_span = make_mdspan(matrix, extents<2>(m,m));
    auto p_span = make_mdspan(pivotIndices, extents<1>(m));
    auto v_span = make_mdspan(vv, extents<1>(m));

    gpu::LUDecompose
    (
        m_span, p_span, v_span
    );

}









template<class ET, class Allocator, class IdxArray>
static inline void LUBacksubstitute
(
    NumericMatrix<ET, Allocator>& matrix,
    const IdxArray& pivotIndices,
    NumericArray<ET, Allocator>& sourceSol
)
{

    auto m = matrix.m();
    auto m_span = make_mdspan(matrix, extents<2>(m,m));
    auto p_span = make_mdspan(pivotIndices, extents<1>(m));
    auto s_span = make_mdspan(sourceSol, extents<1>(m));


    gpu::LUBacksubstitute
    (
        m_span, p_span, s_span
    );

}





}

