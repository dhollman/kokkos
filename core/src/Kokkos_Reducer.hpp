/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#ifndef KOKKOS_REDUCER_HPP
#define KOKKOS_REDUCER_HPP

/** @file Kokkos_Reducer.hpp
 *
 *  @brief Functionality related to the generic concept of a Kokkos Reducer
 *
 *  The modern approach to scalable integration of user-customizable concepts
 *  (like the Kokkos Reducer concept) is to use separable namespace-scope
 *  customization functions (called "customization point objects," or CPOs)---
 *  think `std::begin()` instead of a monolithic customizable traits class
 *  like `std::allocator_traits`.
 *
 *  The current set of CPOs and other namespace traits that we need for the
 *  existing Kokkos Reducer functionality are described below.  We'll put these
 *  CPOs in the namespace `Kokkos::Experimental::Reducers`.
 *
 *  (Note that this is trying to describe the current state of affairs with as
 *  few changes as possible, and arrays make this kind of a mess.)
 *
 *  `Reducers::value_type<R>`
 *  -------------------------
 *
 *  (also includes convenience alias template `value_type_t`)
 *
 *  May be specialized. The default implementation contains a  member type named
 *  `type` that is defined as:
 *
 *    * `R::value_type`, if that type name is well-formed,
 *    * Otherwise, there is not member named `type`.
 *
 *  `value_type<R>::type` must be move-constructible.
 *
 *  If `typename Reducers::value_type<R>::type` is not well-formed, `R` does not
 *  meet the requirements of `Reducer`.
 *
 *  `Reducers::result_view_type<R>`
 *  -------------------------------
 *
 *  (also includes convenience alias template `result_view_type_t`)
 *
 *  May be specialized. The default implementation contains a  member type named
 *  `type` that is defined as:
 *
 *    * `R::result_view_type`, if that type name is well-formed,
 *    * Otherwise, there is not member named `type`.
 *
 *  If `typename Reducers::result_view_type<R>::type` is not well-formed, `R`
 *  does not meet the requirements of `Reducer`.
 *
 *  `auto Reducers::init(R&& r) -> Reducers::value_type_t<R>`
 *  ---------------------------------------------------------
 *
 *  TODO convert this logic to a two-argument version
 *
 *    * If the expression `((R&&)r).init()` is well-formed,
 *      convertible to `value_type_t<R>`, equivalent to `((R&&)r).init()`;
 *    * otherwise, if the unqualified function call expression `init((R&&)r)` is
 *      well-formed (in a context that doesn't include `Reducers::init`),
 *      convertible to `value_type_t<R>`, equivalent to `init((R&&)r)`;
 *    * otherwise, if `((R&&)r).init(vref)` is well-formed given `vref` of type
 *      `value_type_t<R>&`, if `value_type_t<R>` is default constructible,
 *      and if `result_view_type_t<R>::rank == 0`, equvalent to
 *      `auto v = value_type_t<R>{}; ((R&&)r).init(v); return v;`;
 *    * otherwise, if `value_type_t<R>` is default constructible and
 *      `result_view_type_t<R>::rank == 1`,  TODO finish this
 *
 *
 *
 *
 *
 *
 *
 */

namespace Kokkos {

namespace Experimental {

} // end namespace Experimental

} // end namespace Kokkos

#endif  // KOKKOS_REDUCER_HPP
