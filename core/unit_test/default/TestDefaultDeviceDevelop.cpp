
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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <default/TestDefaultDeviceType_Category.hpp>

#include <Kokkos_Graph.hpp>

namespace Test {

TEST(defaultdevicetype, development_test) {
  Kokkos::View<int> count{"graph_kernel_count"};
  Kokkos::View<int> bugs{"graph_kernel_bugs"};
  const auto graph = Kokkos::Experimental::create_graph([=](auto builder) {
    auto root = builder.get_root();

    auto f1 = root.then_parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(long) {
          bugs() += int(count() != 0);
          count()++;
        });
    auto f2 = f1.then_parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(long) {
          bugs() += int(count() < 1 || count() > 2);
          count()++;
        });
    auto f3 = f1.then_parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(long) {
          bugs() += int(count() < 1 || count() > 2);
          count()++;
        });
    builder.when_all(f2, f3).then_parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(long) {
          bugs() += int(count() != 3);
          count()++;
        });
  });

  for(int i = 0; i < 2; ++i) {
    Kokkos::deep_copy(graph.get_execution_space(), count, 0);
    Kokkos::deep_copy(graph.get_execution_space(), bugs, 0);
    graph.submit();
    auto count_host =
        Kokkos::create_mirror_view_and_copy(graph.get_execution_space(), count);
    auto bugs_host =
        Kokkos::create_mirror_view_and_copy(graph.get_execution_space(), bugs);
    graph.get_execution_space().fence();

    ASSERT_EQ(count_host(), 4);
    ASSERT_EQ(bugs_host(), 0);
  }
}

}  // namespace Test
