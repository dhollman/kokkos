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

#ifndef KOKKOS_IMPL_HOST_SHARED_PTR_HPP
#define KOKKOS_IMPL_HOST_SHARED_PTR_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Atomic.hpp>

#include <functional>

namespace Kokkos {

namespace Impl {
// These are outside of HostSharedPtr to avoid multiple instantiation for
// each template
struct is_unmanaged_tag {};
enum unmanaged_control_block_t {
  managed_uninitialized_sentinel = 0,
  unmanaged_sentinel             = intptr_t(~0ULL)
};
static constexpr unmanaged_control_block_t unmanaged_control_block =
    unmanaged_control_block_t::unmanaged_sentinel;
static constexpr unmanaged_control_block_t uninitialized_managed_control_block =
    unmanaged_control_block_t::managed_uninitialized_sentinel;
}  // end namespace Impl

namespace Experimental {

template <typename T>
class MaybeReferenceCountedPtr {
 public:
  using element_type = T;

 protected:
  explicit constexpr MaybeReferenceCountedPtr(std::nullptr_t)
      : m_element_ptr(nullptr),
        m_unmanaged_flag(Kokkos::Impl::uninitialized_managed_control_block) {}

  constexpr
  MaybeReferenceCountedPtr(T* element_ptr, Kokkos::Impl::is_unmanaged_tag)
      : m_element_ptr(element_ptr),
        m_unmanaged_flag(Kokkos::Impl::unmanaged_control_block) {}

  template <class Deleter>
  constexpr
  MaybeReferenceCountedPtr(T* element_ptr, const Deleter& deleter)
      : m_element_ptr(element_ptr),
        m_control(_safe_create_control_block(deleter)) {}

  // use this instead of a lambda to avoid extra template instantiations
  struct _default_deleter {
    void operator()(T* t) const { delete t; }
  };

 public:
  explicit constexpr
  KOKKOS_FUNCTION MaybeReferenceCountedPtr(
      MaybeReferenceCountedPtr&& other) noexcept
      : m_element_ptr(other.m_element_ptr), m_control(other.m_control) {
    other.m_element_ptr = nullptr;
    if (is_reference_counted()) {
      // leave the other one alone if it's unmanaged, since it needs to still
      // hold the unmanaged sentinel value in the control block slot
      other.m_unmanaged_flag =
          Kokkos::Impl::uninitialized_managed_control_block;
    }
  }

  explicit
  KOKKOS_FUNCTION MaybeReferenceCountedPtr(
      const MaybeReferenceCountedPtr& other) noexcept
      : m_element_ptr(other.m_element_ptr), m_control(other.m_control) {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    if (is_reference_counted() && bool(*this)) {
      Kokkos::atomic_add(&(m_control->m_counter), 1);
    }
#endif
  }

  KOKKOS_FUNCTION MaybeReferenceCountedPtr& operator=(
      MaybeReferenceCountedPtr&& other) noexcept {
    if (&other != this) {
      cleanup();
      m_element_ptr = other.m_element_ptr;

      // What we do with the control block depends on if this is reference
      // counted, other is reference counted, both, or neither
      if (is_reference_counted()) {
        if (other.is_reference_counted()) {
          // both are reference counted, so just transfer over the control block
          m_control = other.m_control;
        } else {
          // This is reference counted but other is not, so we need to create a
          // deleter since the pointer was previously unmanaged. Since we have
          // no other option, just create the default one:
          m_control = _safe_create_control_block(_default_deleter{});
        }
      } else if (other.is_reference_counted()) {
        // other is reference counted and this is not.
        // binding an unmanaged reference to a managed one seems sketchy, so
        // maybe we should disallow it, but the expected behavior should just be
        // that we clean up the old reference (since it's moved from) and
        // hope for the best?
        other.cleanup();
        other.m_control = nullptr;
      }
      // otherwise, both are unmanaged and we don't need to do anything to the
      // control blocks

      // in all cases, we need to set the element ptr in other to nullptr to
      // make the object act like a moved-from object. This needs to happen
      // afterwards because of the
      // !is_reference_counted() && other.is_reference_counted() case, where
      // cleanup needs to occur.
      other.m_element_ptr = nullptr;
    }
    return *this;
  }

  KOKKOS_FUNCTION MaybeReferenceCountedPtr& operator=(
      const MaybeReferenceCountedPtr& other) noexcept {
    if (&other != this) {
      cleanup();
      m_element_ptr = other.m_element_ptr;

      // What we do with the control block depends on if this is reference
      // counted, other is reference counted, both, or neither
      if (is_reference_counted()) {
        if (other.is_reference_counted()) {
          // both are reference counted, so just copy the control block pointer
          m_control = other.m_control;
        } else {
          // this is reference counted but the other one isn't, so we need
          // to create a control block and a deleter
          m_control = _safe_create_control_block(_default_deleter{});
        }
      }
      // if other is reference counted and this is not, we don't have anything
      // to do here, but this is sketchy (see comment above in the move
      // assignment operator) and maybe we should disallow it
      // Otherwise, neither is reference counted and we don't need to
      // touch the control blocks

      if (is_reference_counted() && bool(*this)) {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
        Kokkos::atomic_add(&(m_control->m_counter), 1);
#endif
      }
    }
    return *this;
  }

  KOKKOS_FUNCTION ~MaybeReferenceCountedPtr() { cleanup(); }

  KOKKOS_FUNCTION T* get() const noexcept { return m_element_ptr; }
  KOKKOS_FUNCTION T& operator*() const noexcept {
    KOKKOS_EXPECTS(bool(*this));
    return *get();
  }
  KOKKOS_FUNCTION T* operator->() const noexcept { return get(); }

  // checks if the stored pointer is not null
  KOKKOS_FUNCTION explicit operator bool() const noexcept {
    return get() != nullptr;
  }

  // checks whether the MaybeReferenceCountedPtr does reference counting
  // which implies managing the lifetime of objects
  KOKKOS_FUNCTION constexpr bool is_reference_counted() const noexcept {
    return m_unmanaged_flag != Kokkos::Impl::unmanaged_control_block;
  }

 protected:
  // Use a protected member function to avoid protected data members
  KOKKOS_FUNCTION
  int _use_count() const noexcept {
    KOKKOS_EXPECTS(is_reference_counted());
    return bool(*this) ? m_control->m_counter : 0;
  }

 private:
  KOKKOS_FUNCTION void cleanup() noexcept {
#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
    // If m_counter is set, then this instance is responsible for managing the
    // objects pointed to by m_counter and m_element_ptr.
    if (is_reference_counted() && bool(*this)) {
      int const count = Kokkos::atomic_fetch_sub(&(m_control->m_counter), 1);
      if (count == 1) {
        (m_control->m_deleter)(m_element_ptr);
        m_element_ptr = nullptr;
        delete m_control;
        m_control = nullptr;
      }
    }
#endif
  }

  struct Control {
    std::function<void(T*)> m_deleter;
    int m_counter;
  };

  template <class Deleter>
  KOKKOS_FUNCTION Control* _safe_create_control_block(Deleter const& deleter) {
    if (m_element_ptr) {
      try {
        return new Control{deleter, 1};
      } catch (...) {
        deleter(m_element_ptr);
        throw;
      }
    } else {
      return nullptr;
    }
  }

  T* m_element_ptr;
  union {
    Control* m_control;
    Kokkos::Impl::unmanaged_control_block_t m_unmanaged_flag;
  };
};

template <class T>
class HostSharedPtr : public MaybeReferenceCountedPtr<T> {
 private:
  using base_t = MaybeReferenceCountedPtr<T>;

 public:
  // Objects that are default-constructed or initialized with an (explicit)
  // nullptr are not considered reference-counted.
  HostSharedPtr() noexcept : MaybeReferenceCountedPtr<T>(nullptr) {}
  HostSharedPtr(std::nullptr_t) noexcept : HostSharedPtr() {}

  explicit HostSharedPtr(T* element_ptr)
      : MaybeReferenceCountedPtr<T>(element_ptr, [](T* const t) { delete t; }) {
  }

  template <class Deleter>
  HostSharedPtr(T* element_ptr, const Deleter& deleter)
      : MaybeReferenceCountedPtr<T>(element_ptr, deleter) {}

  int use_count() const noexcept { return this->base_t::_use_count(); }
};

template <class T>
class UnmanagedPtr : public MaybeReferenceCountedPtr<T> {
 public:
  UnmanagedPtr() noexcept : MaybeReferenceCountedPtr<T>(nullptr) {}

  explicit UnmanagedPtr(T* element_ptr) noexcept
      : MaybeReferenceCountedPtr<T>(element_ptr,
                                    Kokkos::Impl::is_unmanaged_tag{}) {}
};
}  // namespace Experimental
}  // namespace Kokkos

#endif
