// -*- c++ -*-

#ifndef _LIB_VOLK_ALLOCATOR_HPP_
#define _LIB_VOLK_ALLOCATOR_HPP_

#include <cstdlib>
#include <limits>
#include <volk/volk.h>

namespace gr {
namespace digitalhf {

// see https://en.cppreference.com/w/cpp/named_req/Allocator
template <class T>
struct volk_allocator {
  typedef T value_type;

  volk_allocator() = default;

  template <class U> constexpr volk_allocator(const volk_allocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();

    if (auto p = static_cast<T*>(volk_malloc(n*sizeof(T), volk_get_alignment())))
      return p;

    throw std::bad_alloc();
  }

  void deallocate(T* p, std::size_t) noexcept { volk_free(p); }

} ;

template <class T, class U>
bool operator==(const volk_allocator<T>&, const volk_allocator<U>&) { return true; }

template <class T, class U>
bool operator!=(const volk_allocator<T>&, const volk_allocator<U>&) { return false; }

} // namespace digitalhf
} // namespace gr
#endif // _LIB_VOLK_ALLOCATOR_HPP_
