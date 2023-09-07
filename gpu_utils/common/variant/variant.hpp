#pragma once

#include <type_traits>
#include <utility>
#include "uninitialized.h"
#include <assert.h>
#include "cuda_host_dev.H"

#ifdef __NVIDIA_COMPILER__
#include <thrust/memory.h>
#endif


namespace variant {

namespace detail {

template<bool c, typename T, typename E>
struct if_ {
    typedef E type;
};

template<typename T, typename E>
struct if_<true, T, E> {
    typedef T type;
};

//A type list
struct nil{};

template<typename H, typename T=nil>
struct cons{
    typedef H head_type;
    typedef T tail_type;
};

//Derive the length of a type list
template<typename Cons>
struct cons_length {
    typedef typename Cons::tail_type Tail;
    static const int value = 1 + cons_length<Tail>::value;
};

template<>
struct cons_length<nil> {
    static const int value = 0;
};

//Create a type list from a set of types
template<typename T, typename... Args>
struct cons_type {
    typedef cons<T, typename cons_type<Args...>::type> type;
};

template<typename T>
struct cons_type<T> {
    typedef cons<T, nil> type;
};

//We will store objects in a union.
//non-pod data types will be wrapped
//due to restrictions on what types can be
//placed in unions.
template<typename T>
union storage;

template<typename Head, typename Tail>
union storage<cons<Head, Tail> > {
    typedef Head head_type;
    typedef Tail tail_type;
    Head head;
    storage<Tail> tail;
};

template<>
union storage<nil> {};

//Wrapping works around the restrictions on datatypes that can
//be placed in a union, by wrapping objects in an
//uninitialized<T> type that is POD and therefore fits in a union.

template<typename T>
struct wrapped {
    typedef typename if_<
        std::is_pod<T>::value ||
        std::is_same<T, nil>::value,
        T,
        uninitialized<T> >::type type;
};

template<typename H, typename T>
struct wrapped<cons<H, T> > {
    typedef cons<
        typename wrapped<H>::type,
        typename wrapped<T>::type> type;
};

//Is a type wrapped?
template<typename T>
struct is_wrapped {
    static const bool value = false;
};

template<typename T>
struct is_wrapped<uninitialized<T> > {
    static const bool value = true;
};

//Unwrap a type, if it was wrapped.
template<typename T>
struct unwrapped {
    typedef T type;
};

template<typename T>
struct unwrapped<uninitialized<T> > {
    typedef T type;
};

template<typename H, typename T>
struct unwrapped<cons<H, T> > {
    typedef cons<
        typename unwrapped<H>::type,
        typename unwrapped<T>::type> type;
};

template<int val>
struct int_ {
    static const int value = val;
};

//Use the compiler to find all possible conversions and choose the best
template<typename List, int idx=0>
struct evaluator
    : public evaluator<typename List::tail_type, idx+1> {

    using evaluator<typename List::tail_type, idx+1>::loc;
    using evaluator<typename List::tail_type, idx+1>::type;

    static int_<idx> loc(typename List::head_type);
    static typename List::head_type type(typename List::head_type);
};

template<int idx>
struct evaluator<nil, idx> {
    static int_<idx> loc(nil);
    static nil type(nil);
};

template<typename T, typename List>
struct nearest_type {
    typedef decltype(evaluator<List>::loc(std::declval<T>())) loc;
    static const int value = loc::value;
    typedef decltype(evaluator<List>::type(std::declval<T>())) type;
};



//These overloads call a function on an object held in storage
//The object is unwrapped if it was wrapped due to being non-pod

template<typename Fn, typename H, typename T>
CUDA_HOSTDEV
auto unwrap_apply(Fn fn,
                  const storage<cons<H, T> >& storage) ->
    decltype(fn(std::declval<H>()))
{
    return fn(storage.head);
}


template<typename Fn, typename H, typename T>
CUDA_HOSTDEV
auto unwrap_apply(Fn fn,
                  const storage<cons<uninitialized<H>, T>>& storage) ->
    decltype(fn(std::declval<H>()))
{
    return fn(storage.head.get());
}

//Call a function on a variant
template<typename Fn, typename Cons, int R=0, int L=cons_length<Cons>::value,
         bool stop = R==L-1>
struct apply_to_variant{};

template<typename Fn, typename Head, typename Tail, int R, int L>
struct apply_to_variant<Fn, cons<Head, Tail>, R, L, false> {


    template<typename S>
    CUDA_HOSTDEV
    static auto impl(Fn fn,
                     const S& storage,
                     const char& which) ->
        decltype(fn(std::declval<typename unwrapped<typename S::head_type>::type>()))
    {
        if (R == which) {
            return unwrap_apply(fn, storage);
        } else {
            return apply_to_variant<Fn, Tail, R+1, L>::impl(
                fn, storage.tail, which);
        }
    }
};

template<typename Fn, typename Head, typename Tail, int R, int L>
struct apply_to_variant<Fn, cons<Head, Tail>, R, L, true> {
    template<typename S>
    CUDA_HOSTDEV
    static auto impl(Fn fn,
                     const S& storage,
                     const char& which) ->
        decltype(fn(std::declval<typename unwrapped<typename S::head_type>::type>()))
    {
        return unwrap_apply(fn, storage);
    }
};

} //end namespace detail

//Apply visitor to variant
template<typename Fn, typename Variant>
CUDA_HOSTDEV
auto apply_visitor(Fn fn, const Variant& v) ->
    decltype(fn(std::declval<typename Variant::cons_type::head_type>()))
{
    return detail::apply_to_variant<Fn,
                                    typename Variant::wrapped_type>::
        impl(fn, v.m_storage, v.m_which);
}

namespace detail {



//Construct or assign an object in storage
//Optionally by invoking construct() on the wrapper
//Or by assignment for POD
//If assign is true, we're doing an assignment
//If assign if false, we're copy constructing
//If wrapped is true, storage.head holds uninitialized<V>
//If wrapped if false, storage.head holds V
template<bool assign, typename Cons, typename V, int R, bool Wrapped>
struct do_storage {};

template<typename Cons, typename V, int R>
struct do_storage<false, Cons, V, R, true> {

    CUDA_HOSTDEV
    static void impl(storage<Cons>& storage, char& which, const V& value) {
        storage.head.construct(value);
        which = R;
    }
};

template<typename Cons, typename V, int R>
struct do_storage<true, Cons, V, R, true> {

    CUDA_HOSTDEV
    static void impl(storage<Cons>& storage, char& which, const V& value) {
        storage.head = value;
        which = R;
    }
};

template<bool assign, typename Cons, typename V, int R>
struct do_storage<assign, Cons, V, R, false> {

    CUDA_HOSTDEV
    static void impl(storage<Cons>& storage, char& which, const V& value) {
        storage.head = value;
        which = R;
    }
};

//This loop indexes into the storage class to find the right
//place to put the data.
template<bool assign, typename Cons, typename V, int D, int R=0, bool store=D==R>
struct iterate_do_storage {
    CUDA_HOSTDEV
    static void impl(storage<Cons>& storage, char& which, const V& value) {
        do_storage<assign, Cons, V, R, is_wrapped<typename Cons::head_type>::value>::
            impl(storage, which, value);
    }
};

template<bool assign, typename Cons, typename V, int D, int R>
struct iterate_do_storage<assign, Cons, V, D, R, false> {
    CUDA_HOSTDEV
    static void impl(storage<Cons>& storage, char& which, const V& value) {
        iterate_do_storage<assign, typename Cons::tail_type, V, D, R+1>::
            impl(storage.tail, which, value);
    }
};

template<bool assign, typename V, typename Cons>
struct initialize_storage {
    typedef typename unwrapped<Cons>::type List;
    typedef nearest_type<V, List> best_type;
    typedef typename best_type::type T;
    static const int best_idx = best_type::value;

    CUDA_HOSTDEV
    static void impl(storage<Cons>& storage, char& which,
                     const T& value) {
        iterate_do_storage<assign, Cons, T, best_idx>::impl(storage, which, value);
    }
};

//These overloads call the destructor
template<typename T>
CUDA_HOSTDEV
void destroy(T& t) {
    t.~T();
}

//Wrapped data has a destructor, call it through uninitialized
template<typename T>
CUDA_HOSTDEV
void destroy(uninitialized<T>& wrapped) {
    wrapped.destroy();
}

//Iterate through storage, call the correct destructor
template<typename List, int R=0>
struct destroy_storage{
    CUDA_HOSTDEV
    static void impl(storage<List>& s, int which) {}
};

template<typename Head, typename Tail, int R>
struct destroy_storage<cons<Head, Tail>, R> {
    CUDA_HOSTDEV
    static void impl(storage<cons<Head, Tail> >& s, int which) {
        if (R==which) {
            destroy(s.head);
        } else {
            destroy_storage<Tail, R+1>::impl(s.tail, which);
        }
    }

};


//Copy construction or assignment from another variant requires us to
//visit the other variant.  Each type in the other variant
//must be convertible to one of the types held in the destination
//variant.  This visitor exposes all the types held in the other
//variant, and calls initialize_storage for each one.

//Errors here indicate that the variant being copied or assigned from
//contains at least one type which is not convertible to
//any types in the variant being copied to.
template<bool assign, typename Cons>
struct initialize_from_variant {
    typedef storage<Cons> storage_type;
    storage_type& m_storage;
    char& m_which;
    CUDA_HOSTDEV
        initialize_from_variant(storage_type& storage, char& which) :
        m_storage(storage), m_which(which) {}

    template<typename V>
    CUDA_HOSTDEV
    void operator()(const V& value) const {
        initialize_storage<assign, V, Cons>::impl(m_storage, m_which, value);
    }
};


} //end namespace detail

template<typename T,
         typename... Args>
struct variant {
    typedef typename detail::cons_type<T, Args...>::type cons_type;
    typedef typename detail::wrapped<cons_type>::type wrapped_type;
    typedef detail::storage<wrapped_type> storage_type;
    storage_type m_storage;
    char m_which;


    CUDA_HOSTDEV
    variant() {
        detail::initialize_storage<false, T, wrapped_type>::
            impl(m_storage, m_which, T());
    }

    template<typename V>
    CUDA_HOSTDEV
    variant(const V& value) {
        detail::initialize_storage<false, V, wrapped_type>::
            impl(m_storage, m_which, value);
    }

    CUDA_HOSTDEV
    variant(const variant& value) {
        apply_visitor(detail::initialize_from_variant<false, wrapped_type>(m_storage,
                                                                           m_which),
                      value);
    }

    template<typename S, typename... SArgs>
    CUDA_HOSTDEV
    variant(const variant<S, SArgs...>& value) {
        apply_visitor(detail::initialize_from_variant<false, wrapped_type>(m_storage,
                                                                           m_which),
                      value);
    }

    CUDA_HOSTDEV
    ~variant() {
        detail::destroy_storage<wrapped_type>::impl(m_storage, m_which);
    }


    template<typename V>
    CUDA_HOSTDEV
    variant& operator=(const V& value) {
        detail::destroy_storage<wrapped_type>::impl(m_storage, m_which);
        detail::initialize_storage<true, V, wrapped_type>::impl(m_storage,
                                                                m_which,
                                                                value);
        return *this;
    }

    CUDA_HOSTDEV
    variant& operator=(const variant& value) {
        detail::destroy_storage<wrapped_type>::impl(m_storage, m_which);
        apply_visitor(detail::initialize_from_variant<true, wrapped_type>(m_storage,
                                                                          m_which),
                      value);
        return *this;
    }

    template<typename S, typename... SArgs>
    CUDA_HOSTDEV
    variant& operator=(const variant<S, SArgs...>& value) {
        detail::destroy_storage<wrapped_type>::impl(m_storage, m_which);
        apply_visitor(detail::initialize_from_variant<true, wrapped_type>(m_storage,
                                                                          m_which),
                      value);
        return *this;
    }

    #ifdef __NVIDIA_COMPILER__
    //XXX WAR Thrust presenting references during copies
    //More WAR may be necessary.

    template<typename P>
    CUDA_HOSTDEV
    variant& operator=(const thrust::reference<variant, P>& ref) {
        detail::destroy_storage<wrapped_type>::impl(m_storage, m_which);
        apply_visitor(detail::initialize_from_variant<true, wrapped_type>
                      (m_storage, m_which), thrust::raw_reference_cast(ref));
        return *this;
    }

    #endif

    CUDA_HOSTDEV
    int which() const {
        return m_which;
    }
};

namespace detail {

template<typename T>
struct getter {

    CUDA_HOSTDEV
    T operator()(T in) const {
        return in;
    }

    template<typename X>
    CUDA_HOSTDEV
    T operator()(X) const {
#ifdef __CUDA_ARCH__
        bool bad_get = false;
        assert(bad_get);
        return *static_cast<T*>(nullptr);
#else
        throw std::runtime_error("bad get");
#endif
    }

};

}

template<typename T, typename Variant>
CUDA_HOSTDEV
T get(const Variant& v) {
    return apply_visitor(detail::getter<T>(), v);
}

} //end namespace variant
