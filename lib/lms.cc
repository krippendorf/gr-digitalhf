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
  , _gain()
  , _tmp() {
}

lms::~lms() {
}

void lms::resize(size_t n) {
  if (_gain.size() == n)
    return;
  _gain.resize(n);
  _tmp.resize(n);
  std::fill_n(_gain.begin(), n, 0);
}

void lms::reset() {
  std::fill_n(_gain.begin(), _gain.size(), 0);
}

gr_complex const* lms::update(gr_complex const* beg,
                              gr_complex const* end) {
  assert(end-beg > 0);
  size_t const n = end - beg;
  resize(n);
  volk_32fc_conjugate_32fc(&_tmp[0], beg, n);
  volk_32f_s32f_multiply_32f((float*)&_gain[0], (float const*)&_tmp[0], _mu, 2*n);

  return _gain.data();
}

void lms::set_parameters(std::map<std::string, float>const & p) {
  _mu = p.at("mu");
}

} // namespace digitalhf
} // namespace gr
