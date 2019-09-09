// -*- C++ -*-

#include <cassert>
#include "lms.hpp"
#include <volk/volk.h>

namespace gr {
namespace digitalhf {

filter_update::sptr lms::make(float mu) {
  return filter_update::sptr(new lms(mu));
}

lms::lms(float mu)
  : _mu(mu)
  , _gain() {
}

lms::~lms() {
}

void lms::resize(size_t n) {
  if (_gain.size() == n)
    return;
  _gain.resize(n);
  std::fill_n(_gain.begin(), n, 0);
}

void lms::reset() {
  std::fill_n(_gain.begin(), _gain.size(), 0);
}

gr_complex const* lms::update(gr_complex const* beg,
                              gr_complex const* end) {
  assert(end-beg > 0);
  size_t n = end - beg;
  resize(n);
  for (size_t i=0; i<n; ++i)
    _gain[i] = _mu * std::conj(beg[i]);

  return &_gain.front();
}

} // namespace digitalhf
} // namespace gr
