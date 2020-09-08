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

#ifndef KOKKOS_KOKKOS_HOST_GRAPHNODEKERNEL_HPP
#define KOKKOS_KOKKOS_HOST_GRAPHNODEKERNEL_HPP

#include <Kokkos_Macros.hpp>

#include <impl/Kokkos_Host_Graph_fwd.hpp>

#include <Kokkos_Graph.hpp>

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="GraphNodeKernelImpl"> {{{1

template <class ExecutionSpace>
struct GraphNodeKernelHostImpl {
  // TODO @graphs decide if this should use vtable or intrusive erasure via
  //      function pointers like in the rest of the graph interface
  virtual void execute_kernel() const = 0;
};

template <class ExecutionSpace, class Policy, class Functor>
class GraphNodeKernelImpl<ExecutionSpace, Policy, Functor,
    Kokkos::ParallelForTag>
final : public ParallelFor<Functor, Policy, ExecutionSpace>,
        public GraphNodeKernelHostImpl<ExecutionSpace> {
 public:
  using base_t = ParallelFor<Functor, Policy, ExecutionSpace>;
  using execute_kernel_vtable_base_t = GraphNodeKernelHostImpl<ExecutionSpace>;

  // TODO @graph kernel name info propagation
  template <class PolicyDeduced>
  GraphNodeKernelImpl(std::string, ExecutionSpace const&, Functor arg_functor,
                      PolicyDeduced&& arg_policy)
      : base_t(std::move(arg_functor), (PolicyDeduced &&) arg_policy),
        execute_kernel_vtable_base_t() {}

  // FIXME @graph Forward through the instance once that works in the backends
  template <class PolicyDeduced>
  GraphNodeKernelImpl(ExecutionSpace const& ex, Functor arg_functor,
                      PolicyDeduced&& arg_policy)
      : GraphNodeKernelImpl("", ex, std::move(arg_functor),
                            (PolicyDeduced &&) arg_policy) {}

  void execute_kernel() const final { this->base_t::execute(); }
};

// </editor-fold> end GraphNodeKernelImpl }}}1
//==============================================================================

template <class ExecutionSpace>
struct GraphNodeAggregateKernelHostImpl
    : GraphNodeKernelHostImpl<ExecutionSpace> {
  void execute_kernel() const final { }
};


} // end namespace Impl
} // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_HOST_GRAPHNODEKERNEL_HPP
