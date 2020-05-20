/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_COMBINED_REDUCER_HPP
#define KOKKOS_COMBINED_REDUCER_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_Parallel_Reduce.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_AnonymousSpace.hpp>

namespace Kokkos {
namespace Impl {

// TODO move this to a more general backporting facilities file

// acts like void for comma fold emulation
struct _fold_comma_emulation_return {};

template <class... Ts>
void emulate_fold_comma_operator(Ts&&...) noexcept {}

//==============================================================================

// Note: the index is only to avoid repeating the same base class multiple times
template <size_t Idx, class ValueType>
struct CombinedReducerValueItemImpl {
 public:
  using value_type = ValueType;

 private:
  value_type m_value;

 public:
  CombinedReducerValueItemImpl(value_type arg_value)
      : m_value(std::move(arg_value)) {}

  KOKKOS_INLINE_FUNCTION
  KOKKOS_CONSTEXPR_14 value_type& ref() & noexcept { return m_value; }
  KOKKOS_INLINE_FUNCTION
  constexpr value_type const& ref() const& noexcept { return m_value; }
  KOKKOS_INLINE_FUNCTION
  value_type volatile& ref() volatile& noexcept { return m_value; }
  KOKKOS_INLINE_FUNCTION
  value_type const volatile& ref() const volatile& noexcept { return m_value; }
};

//==============================================================================

template <class IdxSeq, class... ValueTypes>
struct CombinedReducerValueImpl;

template <size_t... Idxs, class... ValueTypes>
struct CombinedReducerValueImpl<integer_sequence<size_t, Idxs...>,
                                ValueTypes...>
    : private CombinedReducerValueItemImpl<Idxs, ValueTypes>... {
 public:
  KOKKOS_INLINE_FUNCTION
  constexpr CombinedReducerValueImpl() = default;
  KOKKOS_INLINE_FUNCTION
  constexpr CombinedReducerValueImpl(CombinedReducerValueImpl const&) = default;
  KOKKOS_INLINE_FUNCTION
  constexpr CombinedReducerValueImpl(CombinedReducerValueImpl&&) noexcept =
      default;
  KOKKOS_INLINE_FUNCTION
  KOKKOS_CONSTEXPR_14 CombinedReducerValueImpl& operator=(
      CombinedReducerValueImpl const&) = default;
  KOKKOS_INLINE_FUNCTION
  KOKKOS_CONSTEXPR_14 CombinedReducerValueImpl& operator=(
      CombinedReducerValueImpl&&) noexcept = default;
  KOKKOS_INLINE_FUNCTION
  ~CombinedReducerValueImpl() = default;

  explicit CombinedReducerValueImpl(ValueTypes... arg_values)
      : CombinedReducerValueItemImpl<Idxs, ValueTypes>(
            std::move(arg_values))... {}

  template <size_t Idx, class ValueType>
  KOKKOS_INLINE_FUNCTION ValueType& get() & noexcept {
    return this->CombinedReducerValueItemImpl<Idx, ValueType>::ref();
  }
  template <size_t Idx, class ValueType>
  KOKKOS_INLINE_FUNCTION ValueType const& get() const& noexcept {
    return this->CombinedReducerValueItemImpl<Idx, ValueType>::ref();
  }
  template <size_t Idx, class ValueType>
  KOKKOS_INLINE_FUNCTION ValueType volatile& get() volatile& noexcept {
    return this->CombinedReducerValueItemImpl<Idx, ValueType>::ref();
  }
  template <size_t Idx, class ValueType>
  KOKKOS_INLINE_FUNCTION ValueType const volatile& get() const
      volatile& noexcept {
    return this->CombinedReducerValueItemImpl<Idx, ValueType>::ref();
  }
};

//==============================================================================

// TODO Empty base optmization?
template <size_t Idx, class Reducer>
// requires Kokkos::is_reducer<Reducer>
struct CombinedReducerStorageImpl {
  using value_type = typename Reducer::value_type;

  Reducer m_reducer;

  KOKKOS_INLINE_FUNCTION
  explicit constexpr CombinedReducerStorageImpl(Reducer arg_reducer)
      : m_reducer(std::move(arg_reducer)) {}

  // Leading underscores to make it clear that this class is not intended to
  // model Reducer

  KOKKOS_INLINE_FUNCTION
  constexpr _fold_comma_emulation_return _init(value_type& val) const {
    m_reducer.init(val);
    return _fold_comma_emulation_return{};
  }

  KOKKOS_INLINE_FUNCTION constexpr _fold_comma_emulation_return _join(
      value_type& dest, value_type const& src) const {
    m_reducer.join(dest, src);
    return _fold_comma_emulation_return{};
  }

  KOKKOS_INLINE_FUNCTION constexpr _fold_comma_emulation_return _join(
      value_type volatile& dest, value_type const volatile& src) const {
    m_reducer.join(dest, src);
    return _fold_comma_emulation_return{};
  }

  KOKKOS_INLINE_FUNCTION
  constexpr value_type& _reference() const { return m_reducer.reference(); }
};

//------------------------------------------------------------------------------

struct _construct_combined_reducer_from_args_tag {};

template <class IdxSeq, class Space, class...>
struct CombinedReducerImpl;

template <size_t... Idxs, class Space, class... Reducers>
struct CombinedReducerImpl<integer_sequence<size_t, Idxs...>, Space,
                           Reducers...>
    : private CombinedReducerStorageImpl<Idxs, Reducers>... {
 public:
  using reducer = CombinedReducerImpl<integer_sequence<size_t, Idxs...>, Space,
                                      Reducers...>;
  using value_type = CombinedReducerValueImpl<integer_sequence<size_t, Idxs...>,
                                              typename Reducers::value_type...>;
  using result_view_type = Kokkos::View<value_type, Space>;

 private:
  value_type m_value;
  result_view_type m_value_view;

 public:
  template <class... ValueReferences>
  KOKKOS_FUNCTION constexpr explicit CombinedReducerImpl(
      _construct_combined_reducer_from_args_tag,
      ValueReferences&... refs) noexcept
      : CombinedReducerStorageImpl<Idxs, Reducers>(Reducers(refs))...,
        m_value(refs...),
        m_value_view(&m_value) {}

  // TODO check if this gets used (it's part of the reducer concept, but we may
  //  want to remove it from here later anyway since this is an implementation
  //  detail).
  KOKKOS_FUNCTION
  constexpr explicit CombinedReducerImpl(
      result_view_type const& arg_view) noexcept
      : CombinedReducerStorageImpl<Idxs, Reducers>(
            typename Reducers::result_view_type(
                &arg_view()
                     .template get<Idxs, typename Reducers::value_type>()))...,
        m_value(arg_view()),
        m_value_view(&m_value) {}

  // TODO technically m_value and m_value_view aren't valid in copy constructed
  //      instances of the reducer, but they're also only used in instances that
  //      don't get copy constructed.  This is a general design flaw in Kokkos
  //      reducers that doesn't really need to be solved here.

  KOKKOS_FUNCTION void join(value_type& dest,
                            value_type const& src) const noexcept {
    emulate_fold_comma_operator(
        this->CombinedReducerStorageImpl<Idxs, Reducers>::_join(
            dest.template get<Idxs, typename Reducers::value_type>(),
            src.template get<Idxs, typename Reducers::value_type>())...);
  }

  KOKKOS_FUNCTION void join(value_type volatile& dest,
                            value_type const volatile& src) const noexcept {
    emulate_fold_comma_operator(
        this->CombinedReducerStorageImpl<Idxs, Reducers>::_join(
            dest.template get<Idxs, typename Reducers::value_type>(),
            src.template get<Idxs, typename Reducers::value_type>())...);
  }

  KOKKOS_FUNCTION
  void init(value_type& dest) const noexcept {
    emulate_fold_comma_operator(
        this->CombinedReducerStorageImpl<Idxs, Reducers>::_init(
            dest.template get<Idxs, typename Reducers::value_type>())...);
  }

  // TODO figure out if we also need to call through to final

  KOKKOS_FUNCTION
  constexpr bool references_scalar() const noexcept {
    // For now, always pretend that we reference a scalar since we need to
    // block to do the write-back because the references may not be contiguous
    // in memory and the backends currently assume this and just do a single
    // deep copy back to a chunk of memory associated with the output argument
    return true;
  }

  KOKKOS_FUNCTION
  constexpr result_view_type const& view() const noexcept {
    return m_value_view;
  }

  KOKKOS_FUNCTION
  constexpr void write_value_back_to_original_references(
      Reducers const&... reducers_that_reference_original_values)
      const noexcept {
    emulate_fold_comma_operator(
        (reducers_that_reference_original_values.view()() =
             m_value.template get<Idxs, typename Reducers::value_type>())...);
  }
};

// Apparently this can't be an alias template because of a bug/unimplemented
// feature in GCC's name mangler.  But in this case, this amounts to the same
// thing.
template <class Space, class... Reducers>
struct CombinedReducer
    : CombinedReducerImpl<make_index_sequence<sizeof...(Reducers)>, Space,
                          Reducers...> {
  using base_t = CombinedReducerImpl<make_index_sequence<sizeof...(Reducers)>,
                                     Space, Reducers...>;
  using base_t::base_t;
  using reducer = CombinedReducer<Space, Reducers...>;
};

//==============================================================================

template <class IdxSeq, class Functor, class Space, class... Reducers>
struct CombinedReductionFunctorWrapperImpl;

template <size_t... Idxs, class Functor, class Space, class... Reducers>
struct CombinedReductionFunctorWrapperImpl<integer_sequence<size_t, Idxs...>,
                                           Functor, Space, Reducers...> {
 private:
  Functor m_functor;

 public:
  //------------------------------------------------------------------------------
  // <editor-fold desc="type aliases"> {{{2

  using reducer_type = CombinedReducer<Space, Reducers...>;

  // Prevent Kokkos from attempting to deduce value_type
  using value_type = typename reducer_type::value_type;

  // </editor-fold> end type aliases }}}2
  //------------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="Ctors, destructor, and assignment"> {{{2

  KOKKOS_INLINE_FUNCTION
  constexpr CombinedReductionFunctorWrapperImpl() = default;
  KOKKOS_INLINE_FUNCTION
  constexpr CombinedReductionFunctorWrapperImpl(
      CombinedReductionFunctorWrapperImpl const&) = default;
  KOKKOS_INLINE_FUNCTION
  constexpr CombinedReductionFunctorWrapperImpl(
      CombinedReductionFunctorWrapperImpl&&) noexcept = default;
  KOKKOS_INLINE_FUNCTION
  KOKKOS_CONSTEXPR_14 CombinedReductionFunctorWrapperImpl& operator=(
      CombinedReductionFunctorWrapperImpl const&) = default;
  KOKKOS_INLINE_FUNCTION
  KOKKOS_CONSTEXPR_14 CombinedReductionFunctorWrapperImpl& operator=(
      CombinedReductionFunctorWrapperImpl&&) noexcept = default;
  KOKKOS_INLINE_FUNCTION
  ~CombinedReductionFunctorWrapperImpl() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr explicit CombinedReductionFunctorWrapperImpl(Functor arg_functor)
      : m_functor(std::move(arg_functor)) {}

  // </editor-fold> end Ctors, destructor, and assignment }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="call operator"> {{{2

  template <class IndexOrMemberType>
  KOKKOS_FUNCTION void operator()(IndexOrMemberType&& arg_first,
                                  value_type& out) const {
    m_functor((IndexOrMemberType &&) arg_first,
              out.template get<Idxs, typename Reducers::value_type>()...);
  }

  // Tagged version
  template <class Tag, class IndexOrMemberType>
  KOKKOS_FUNCTION void operator()(Tag&& arg_tag, IndexOrMemberType&& arg_first,
                                  value_type& out) const {
    m_functor((Tag &&) arg_tag, (IndexOrMemberType &&) arg_first,
              out.template get<Idxs, typename Reducers::value_type>()...);
  }

  // </editor-fold> end call operator }}}2
  //----------------------------------------------------------------------------

  // These are things that need to be done if we decide to ever support
  // functor-customized join/init/final hooks with combined reducers. For now,
  // they are explicitly not supported.
  // TODO: forward join() function to user functor hook, or just ignore it?
  // TODO: forward init() function to user functor hook, or just ignore it?
  // TODO: forward final() function to user functor hook, or just ignore it?
};

//==============================================================================

template <class Functor, class Space, class... Reducers>
struct CombinedReductionFunctorWrapper
    : CombinedReductionFunctorWrapperImpl<
          make_index_sequence<sizeof...(Reducers)>, Functor, Space,
          Reducers...> {
  using base_t = CombinedReductionFunctorWrapperImpl<
      make_index_sequence<sizeof...(Reducers)>, Functor, Space, Reducers...>;
  using base_t::base_t;
};

//------------------------------------------------------------------------------
// <editor-fold desc="_make_reducer_from_arg"> {{{2

template <class Space, class Reducer>
KOKKOS_INLINE_FUNCTION constexpr typename std::enable_if<
    Kokkos::is_reducer<typename std::decay<Reducer>::type>::value,
    typename std::decay<Reducer>::type>::type
_make_reducer_from_arg(Reducer&& arg_reducer) noexcept {
  return arg_reducer;
}

// Two purposes: SFINAE-safety for the `View` case and laziness for the return
// value otherwise to prevent extra instantiations of the Kokkos::Sum template
template <class Space, class T, class Enable = void>
struct _wrap_with_kokkos_sum {
  using type = Kokkos::Sum<T, Space>;
};

template <class Space, class T>
struct _wrap_with_kokkos_sum<
    Space, T, typename std::enable_if<Kokkos::is_view<T>::value>::type> {
  using type = Kokkos::Sum<typename T::value_type, Space>;
};

// TODO better error message for the case when a const& to a scalar is passed in
template <class Space, class T>
KOKKOS_INLINE_FUNCTION constexpr typename std::enable_if<
    !Kokkos::is_reducer<typename std::decay<T>::type>::value,
    _wrap_with_kokkos_sum<Space, typename std::decay<T>::type>>::type::type
_make_reducer_from_arg(T& arg_scalar) noexcept {
  return
      typename _wrap_with_kokkos_sum<Space, typename std::decay<T>::type>::type{
          arg_scalar};
}

// This can't be an alias template because GCC doesn't know how to mangle
// decltype expressions in return statements (and, even though every compiler
// is supposed to, GCC is the only one that does dependent alias template
// substitution correctly and tries to do the mangling, aparently).
template <class Space, class ReferenceOrViewOrReducer, class = void>
struct _reducer_from_arg {
  using type = decltype(Impl::_make_reducer_from_arg<Space>(
      std::declval<ReferenceOrViewOrReducer&&>()));
};
template <class Space, class ReferenceOrViewOrReducer>
using _reducer_from_arg_t =
    typename _reducer_from_arg<Space, ReferenceOrViewOrReducer>::type;

// </editor-fold> end _make_reducer_from_arg }}}2
//------------------------------------------------------------------------------

template <class Space, class... ReferencesOrViewsOrReducers>
KOKKOS_INLINE_FUNCTION constexpr CombinedReducer<
    Space,
    typename _reducer_from_arg<Space, ReferencesOrViewsOrReducers>::type...>
make_combined_reducer(ReferencesOrViewsOrReducers&&... args) {
  //----------------------------------------
  using reducer_type = CombinedReducer<
      Space, _reducer_from_arg_t<Space, ReferencesOrViewsOrReducers>...>;
  return reducer_type(_construct_combined_reducer_from_args_tag{},
                      (ReferencesOrViewsOrReducers &&) args...);
  //----------------------------------------
}

template <class Functor, class Space, class... ReferencesOrViewsOrReducers>
KOKKOS_INLINE_FUNCTION constexpr CombinedReductionFunctorWrapper<
    Functor, Space,
    typename _reducer_from_arg<Space, ReferencesOrViewsOrReducers>::type...>
make_wrapped_combined_functor(Functor const& functor, Space,
                              ReferencesOrViewsOrReducers&&...) {
  //----------------------------------------
  return CombinedReductionFunctorWrapper<
      Functor, Space,
      _reducer_from_arg_t<Space, ReferencesOrViewsOrReducers>...>(functor);
  //----------------------------------------
}

}  // end namespace Impl

//==============================================================================
// <editor-fold desc="Overloads of parallel_reduce for multiple outputs"> {{{1

// These need to be forwarding references so that we can deduce const-ness,
// but none of them should be forwarded (and, indeed, none of them should be
// rvalue references)
template <class PolicyType, class Functor, class ReturnType1, class ReturnType2,
          class... ReturnTypes>
auto parallel_reduce(std::string const& label, PolicyType const& policy,
                     Functor const& functor, ReturnType1&& returnType1,
                     ReturnType2&& returnType2,
                     ReturnTypes&&... returnTypes) noexcept ->
    typename std::enable_if<
        Kokkos::Impl::is_execution_policy<PolicyType>::value>::type {
  // TODO static_assert that none of the ReturnType&& are r-value references?
  // This has to be const because that's currently how Kokkos detects if
  // something is a reducer in the parallel_reduce overload set (!!!)
  using space_type = typename std::decay<decltype(policy.space())>::type;
  const auto combined_reducer = Impl::make_combined_reducer<space_type>(
      returnType1, returnType2, returnTypes...);
  using combined_reducer_type = decltype(combined_reducer);

  auto combined_functor = Impl::make_wrapped_combined_functor(
      functor, policy.space(), returnType1, returnType2, returnTypes...);

  using combined_functor_type = decltype(combined_functor);

  Impl::ParallelReduceAdaptor<PolicyType, combined_functor_type,
                              combined_reducer_type>::execute(label, policy,
                                                              combined_functor,
                                                              combined_reducer);
  Impl::ParallelReduceFence<typename PolicyType::execution_space,
                            combined_reducer_type>::fence(policy.space(),
                                                          combined_reducer);
  combined_reducer.write_value_back_to_original_references(
      Impl::_make_reducer_from_arg<space_type>(returnType1),
      Impl::_make_reducer_from_arg<space_type>(returnType2),
      Impl::_make_reducer_from_arg<space_type>(returnTypes)...);
}

template <class PolicyType, class Functor, class ReturnType1, class ReturnType2,
          class... ReturnTypes>
auto parallel_reduce(PolicyType const& policy, Functor const& functor,
                     ReturnType1&& returnType1, ReturnType2&& returnType2,
                     ReturnTypes&&... returnTypes) noexcept ->
    typename std::enable_if<
        Kokkos::Impl::is_execution_policy<PolicyType>::value>::type {
  parallel_reduce("", policy, functor, std::forward<ReturnType1>(returnType1),
                  std::forward<ReturnType2>(returnType2),
                  std::forward<ReturnTypes>(returnTypes)...);
}

template <class Functor, class ReturnType1, class ReturnType2,
          class... ReturnTypes>
void parallel_reduce(std::string const& label, size_t n, Functor const& functor,
                     ReturnType1&& returnType1, ReturnType2&& returnType2,
                     ReturnTypes&&... returnTypes) noexcept {
  parallel_reduce(label, RangePolicy<Kokkos::DefaultExecutionSpace>(0, n),
                  functor, std::forward<ReturnType1>(returnType1),
                  std::forward<ReturnType2>(returnType2),
                  std::forward<ReturnTypes>(returnTypes)...);
}

template <class Functor, class ReturnType1, class ReturnType2,
          class... ReturnTypes>
void parallel_reduce(size_t n, Functor const& functor,
                     ReturnType1&& returnType1, ReturnType2&& returnType2,
                     ReturnTypes&&... returnTypes) noexcept {
  parallel_reduce("", n, functor, std::forward<ReturnType1>(returnType1),
                  std::forward<ReturnType2>(returnType2),
                  std::forward<ReturnTypes>(returnTypes)...);
}

//------------------------------------------------------------------------------
// <editor-fold desc="Team overloads"> {{{2

// Copied three times because that's the best way we have right now to match
// Impl::TeamThreadRangeBoundariesStruct,
// Impl::ThreadVectorRangeBoundariesStruct, and
// Impl::TeamVectorRangeBoundariesStruct
template <class iType, class MemberType, class Functor, class ReturnType1,
          class ReturnType2, class... ReturnTypes>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    std::string const& label,
    Impl::TeamThreadRangeBoundariesStruct<iType, MemberType> const& boundaries,
    Functor const& functor, ReturnType1&& returnType1,
    ReturnType2&& returnType2, ReturnTypes&&... returnTypes) noexcept {
  const auto combined_reducer =
      Impl::make_combined_reducer<Kokkos::AnonymousSpace>(
          returnType1, returnType2, returnTypes...);

  auto combined_functor = Impl::make_wrapped_combined_functor(
      functor, Kokkos::AnonymousSpace(), returnType1, returnType2,
      returnTypes...);

  parallel_reduce(label, boundaries, combined_functor, combined_reducer);
}

template <class iType, class MemberType, class Functor, class ReturnType1,
          class ReturnType2, class... ReturnTypes>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    std::string const& label,
    Impl::ThreadVectorRangeBoundariesStruct<iType, MemberType> const&
        boundaries,
    Functor const& functor, ReturnType1&& returnType1,
    ReturnType2&& returnType2, ReturnTypes&&... returnTypes) noexcept {
  const auto combined_reducer =
      Impl::make_combined_reducer<Kokkos::AnonymousSpace>(
          returnType1, returnType2, returnTypes...);

  auto combined_functor = Impl::make_wrapped_combined_functor(
      functor, Kokkos::AnonymousSpace(), returnType1, returnType2,
      returnTypes...);

  parallel_reduce(label, boundaries, combined_functor, combined_reducer);
}

template <class iType, class MemberType, class Functor, class ReturnType1,
          class ReturnType2, class... ReturnTypes>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    std::string const& label,
    Impl::TeamVectorRangeBoundariesStruct<iType, MemberType> const& boundaries,
    Functor const& functor, ReturnType1&& returnType1,
    ReturnType2&& returnType2, ReturnTypes&&... returnTypes) noexcept {
  const auto combined_reducer =
      Impl::make_combined_reducer<Kokkos::AnonymousSpace>(
          returnType1, returnType2, returnTypes...);

  auto combined_functor = Impl::make_wrapped_combined_functor(
      functor, Kokkos::AnonymousSpace(), returnType1, returnType2,
      returnTypes...);

  parallel_reduce(label, boundaries, combined_functor, combined_reducer);
}

// </editor-fold> end Team overloads }}}2
//------------------------------------------------------------------------------

// </editor-fold> end Overloads of parallel_reduce for multiple outputs }}}1
//==============================================================================

}  // namespace Kokkos

#endif  // KOKKOS_COMBINED_REDUCER_HPP
