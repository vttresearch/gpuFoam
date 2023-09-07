#pragma once


#include <type_traits>
#include <iterator>

namespace topaz {

template<typename T>
struct IsScalar : std::is_arithmetic<T> {};

template<typename T>
static constexpr bool IsScalar_v = IsScalar<T>::value;

///////////////////////////////////////////////////////////////////////////////////

template <typename T, typename = void>
struct IsIterator : public std::false_type {};

template <typename T>
struct IsIterator<T,
                  typename std::enable_if<!std::is_same<
                      typename std::iterator_traits<T>::value_type,
                      void>::value>::type> : public std::true_type {};

template<typename T>
constexpr bool IsIterator_v = IsIterator<T>::value;

///////////////////////////////////////////////////////////////////////////////////
template<typename T, typename = void>
struct IsNumericVector : std::false_type {};


template<typename T>
struct IsNumericVector<T, std::enable_if_t< T::is_numeric_vector >>
: public std::true_type {};



template< typename T >
constexpr bool IsNumericVector_v = IsNumericVector<T>::value;

///////////////////////////////////////////////////////////////////////////////////


template<typename T, typename = void>
struct IsRange : std::false_type {};


template<typename T>
struct IsRange<T, std::enable_if_t< T::is_range >>
: public std::true_type {};




template< typename T >
constexpr bool IsRange_v = IsRange<T>::value;


///////////////////////////////////////////////////////////////////////////////////



template<typename T, typename = void>
struct IsRangeOrNumericArray : std::false_type {};



template<typename T>
struct IsRangeOrNumericArray<T, std::enable_if_t< (IsNumericVector_v<T>)||(IsRange_v<T>) >>
: public std::true_type {};




template< typename T >
constexpr bool IsRangeOrNumericArray_v = IsRangeOrNumericArray<T>::value;


///////////////////////////////////////////////////////////////////////////////////

template<typename T1, typename T2, typename = void>
struct BothRangesOrNumericArrays : public std::false_type {};

template<typename T1, typename T2>
struct BothRangesOrNumericArrays< T1, T2, std::enable_if_t< IsRangeOrNumericArray_v<T1> && IsRangeOrNumericArray_v<T2> > >
   : public std::true_type {};

template< typename T1, typename T2 >
constexpr bool BothRangesOrNumericArrays_v = BothRangesOrNumericArrays<T1,T2>::value;

///////////////////////////////////////////////////////////////////////////////////


template< typename T1, typename T2, typename = void >
struct SupportsBinaryExpression
   : public std::false_type {};



//Both are either expressions or fields
template< typename T1, typename T2 >
struct SupportsBinaryExpression< T1, T2, std::enable_if_t< BothRangesOrNumericArrays_v<T1, T2> > >
   : public std::true_type {};



//LHS is a scalar value
template< typename T1, typename T2 >
struct SupportsBinaryExpression< T1, T2, std::enable_if_t< IsScalar_v<T1> && IsRangeOrNumericArray_v<T2> > >
   : public std::true_type {};



//RHS is a scalar value
template< typename T1, typename T2 >
struct SupportsBinaryExpression< T1, T2, std::enable_if_t< IsRangeOrNumericArray_v<T1> && IsScalar_v<T2> > >
   : public std::true_type {};




//shorthand for getting the value
template< typename T1, typename T2 >
constexpr bool SupportsBinaryExpression_v = SupportsBinaryExpression<T1,T2>::value;
///////////////////////////////////////////////////////////////////////////////////



















} // namespace topaz
