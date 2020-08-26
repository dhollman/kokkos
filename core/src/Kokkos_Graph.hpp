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

#ifndef KOKKOS_GRAPH_HPP
#define KOKKOS_GRAPH_HPP

#include <Kokkos_Macros.hpp>

#include <Kokkos_Graph_fwd.hpp>
#include <impl/Kokkos_GraphImpl_fwd.hpp>

// GraphAccess needs to be defined, not just declared
#include <impl/Kokkos_GraphImpl.hpp>

#include <impl/Kokkos_Utilities.hpp>  // fold emulation

#include <functional>
#include <memory>

namespace Kokkos {
namespace Experimental {

//==============================================================================
// <editor-fold desc="Graph"> {{{1

template <class ExecutionSpace>
struct KOKKOS_ATTRIBUTE_NODISCARD Graph {
 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="public member types"> {{{2

  using execution_space = ExecutionSpace;
  using graph           = Graph;
  using graph_builder   = GraphBuilder<ExecutionSpace>;

  // </editor-fold> end public member types }}}2
  //----------------------------------------------------------------------------

 private:
  //----------------------------------------------------------------------------
  // <editor-fold desc="friends"> {{{2

  friend struct Kokkos::Impl::GraphAccess;

  // </editor-fold> end friends }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="private data members"> {{{2

  using impl_t                       = Kokkos::Impl::GraphImpl<ExecutionSpace>;
  std::shared_ptr<impl_t> m_impl_ptr = nullptr;

  // </editor-fold> end private data members }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="private ctors"> {{{2

  // Note: only create_graph() uses this constructor, but we can't just make
  // that a friend instead of GraphAccess because of the way that friend
  // function template injection works.
  explicit Graph(std::shared_ptr<impl_t> arg_impl_ptr)
      : m_impl_ptr(std::move(arg_impl_ptr)) {}

  // </editor-fold> end private ctors }}}2
  //----------------------------------------------------------------------------

 public:
  ExecutionSpace const& get_execution_space() const {
    return m_impl_ptr->get_execution_space();
  }

  void submit() const& {
    KOKKOS_EXPECTS(bool(m_impl_ptr))
    (*m_impl_ptr).submit();
  }

  void submit() && {
    KOKKOS_EXPECTS(bool(m_impl_ptr))
    // The graph interface isn't thread-safe, so we can rely on this
    if (m_impl_ptr.use_count() == 1) {
      std::move(*m_impl_ptr).submit();
    } else {
      (*m_impl_ptr).submit();
    }
  }
};

// </editor-fold> end Graph }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="GraphBuilder"> {{{1

template <class ExecutionSpace>
struct GraphBuilder {
 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="public member types"> {{{2

  using execution_space = ExecutionSpace;
  using graph           = Graph<ExecutionSpace>;
  using graph_builder   = GraphBuilder;

  // </editor-fold> end public member types }}}2
  //----------------------------------------------------------------------------

 private:
  //----------------------------------------------------------------------------
  // <editor-fold desc="friends"> {{{2

  friend struct Kokkos::Impl::GraphAccess;

  // </editor-fold> end friends }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="private data members"> {{{2

  using graph_impl_t    = Kokkos::Impl::GraphImpl<ExecutionSpace>;
  using root_node_ref_t = typename graph_impl_t::root_node_impl_t::node_ref_t;
  root_node_ref_t m_root;

  // </editor-fold> end private data members }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="private ctors"> {{{2

  // Note: only create_graph() uses this constructor, but we can't just make
  // that a friend instead of GraphAccess because of the way that friend
  // function template injection works.
  explicit GraphBuilder(root_node_ref_t arg_root)
      : m_root(std::move(arg_root)) {}

  // </editor-fold> end private ctors }}}2
  //----------------------------------------------------------------------------

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="ctors, destructor, and assignment"> {{{2

  // Rule of 6 for copy constructible

  GraphBuilder() noexcept               = default;
  GraphBuilder(GraphBuilder const&)     = default;
  GraphBuilder(GraphBuilder&&) noexcept = default;
  GraphBuilder& operator=(GraphBuilder const&) = default;
  GraphBuilder& operator=(GraphBuilder&&) noexcept = default;

  ~GraphBuilder() = default;

  // </editor-fold> end ctors, destructor, and assignment }}}2
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="public accessors"> {{{2

  constexpr auto const& get_root() const { return m_root; }

  // </editor-fold> end public accessors }}}2
  //----------------------------------------------------------------------------

  template <class... PredecessorRefs>
  // constraints (not intended for subsumption, though...)
  //   ((remove_cvref_t<PredecessorRefs> is a specialization of
  //        GraphNodeRef with get_root().get_graph_impl() as its GraphImpl)
  //      && ...)
  auto when_all(PredecessorRefs&&... arg_pred_refs) const {
    // TODO @graph @desul-integration check the constraints and preconditions
    //                                once we have folded conjunctions from
    //                                desul
    auto graph_ptr_impl = get_root().get_graph_ptr();
    auto node_ptr_impl = graph_ptr_impl->create_aggregate_ptr(arg_pred_refs...);
    graph_ptr_impl->add_node(node_ptr_impl);
    KOKKOS_IMPL_FOLD_COMMA_OPERATOR(graph_ptr_impl->add_predecessor(
        node_ptr_impl, arg_pred_refs) /* ... */);
    return Kokkos::Impl::GraphAccess::make_graph_node_ref(
        graph_ptr_impl, std::move(node_ptr_impl));
  }

  //----------------------------------------------------------------------------
  // <editor-fold desc="Methods forward to their then_* analogs on root"> {{{2

  template <class... Args>
  auto parallel_for(Args&&... args) const {
    return get_root().then_parallel_for((Args &&) args...);
  }
  template <class... Args>
  auto parallel_reduce(Args&&... args) const {
    return get_root().then_parallel_reduce((Args &&) args...);
  }
  template <class... Args>
  auto parallel_scan(Args&&... args) const {
    return get_root().then_parallel_scan((Args &&) args...);
  }
  template <class... Args>
  auto deep_copy(Args&&... args) const {
    return get_root().then_deep_copy((Args &&) args...);
  }

  // </editor-fold> end Methods forward to their then_* analogs on root }}}2
  //----------------------------------------------------------------------------
};

// </editor-fold> end GraphBuilder }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="create_graph"> {{{1

template <class ExecutionSpace, class Closure>
Graph<ExecutionSpace> create_graph(ExecutionSpace ex, Closure&& arg_closure) {
  // Create a shared pointer to the graph:
  auto rv = Kokkos::Impl::GraphAccess::construct_graph(ex);
  // Create the graph builder instance:
  auto builder = Kokkos::Impl::GraphAccess::create_graph_builder(
      Kokkos::Impl::GraphAccess::create_root_ref(rv));
  // Invoke the user's graph construction closure
  ((Closure &&) arg_closure)(std::move(builder));
  // and given them back the graph
  // KOKKOS_ENSURES(rv.m_impl_ptr.use_count() == 1)
  return rv;
}

template <class ExecutionSpace = DefaultExecutionSpace,
          class Closure = Kokkos::Impl::AlwaysDeduceThisTemplateParameter>
Graph<ExecutionSpace> create_graph(Closure&& arg_closure) {
  return create_graph(ExecutionSpace{}, (Closure &&) arg_closure);
}

// </editor-fold> end create_graph }}}1
//==============================================================================

}  // end namespace Experimental
}  // namespace Kokkos

// Even though these things are separable, include them here for now so that
// the user only needs to include Kokkos_Graph.hpp to get the whole facility.
#include <Kokkos_GraphNode.hpp>

#include <impl/Kokkos_GraphNodeImpl.hpp>
#include <impl/Kokkos_GraphNodeCustomization.hpp>

#include <impl/Kokkos_Host_Graph_Impl.hpp>

#include <Cuda/Kokkos_Cuda_Graph_Impl.hpp>

#endif  // KOKKOS_GRAPH_HPP
