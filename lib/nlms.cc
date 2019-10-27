// -*- C++ -*-

#include <cassert>
#include "nlms.hpp"
#include <volk/volk.h>

namespace gr {
namespace digitalhf {

filter_update::sptr nlms::make(float mu, float delta) {
  return filter_update::sptr(new nlms(mu, delta));
}

nlms::nlms(float mu, float delta)
  : _mu(mu)
  , _delta(delta)
  , _gain()
  , _tmp()
  , _t1() {
}

nlms::~nlms() {
}

void nlms::resize(size_t n) {
  if (_gain.size() == n)
    return;
  _gain.resize(n);
  _tmp.resize(n);
  _t1.resize(n);
  std::fill_n(_gain.begin(), n, 0);
}

void nlms::reset() {
  std::fill_n(_gain.begin(), _gain.size(), 0);
}

gr_complex const* nlms::update(gr_complex const* beg,
                               gr_complex const* end) {
  assert(end-beg > 0);
  size_t const n = end - beg;
  resize(n);
  volk_32fc_conjugate_32fc(&_tmp[0], beg, n);
  volk_32fc_magnitude_squared_32f(&_t1[0], beg, n);
  float norm = _delta;
  volk_32f_accumulator_s32f(&norm, &_t1[0], n);
  volk_32f_s32f_multiply_32f((float*)&_gain[0], (float const*)&_tmp[0], std::max(0.005f, _mu/norm), 2*n);

  return _gain.data();
}

void nlms::set_parameters(std::map<std::string, float>const & p) {
  _mu    = p.at("mu");
  _delta = p.at("delta");
}

} // namespace digitalhf
} // namespace gr
