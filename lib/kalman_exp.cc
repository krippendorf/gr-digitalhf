// -*- C++ -*-

#include <cassert>
#include "kalman_exp.hpp"
#include <volk/volk.h>

namespace gr {
namespace digitalhf {

filter_update::sptr kalman_exp::make(float r, float lambda) {
  return filter_update::sptr(new kalman_exp(r, lambda));
}

kalman_exp::kalman_exp(float r, float lambda)
  : _r(r)
  , _lambda(lambda)
  , _gain()
  , _p()
  , _temp()
  , _t1() {
}

kalman_exp::~kalman_exp() {
}

void kalman_exp::resize(size_t n) {
  if (_gain.size() == n &&
      _p.size()    == n*n &&
      _temp.size() == n*n &&
      _t1.size()   == n*n)
    return;
  _gain.resize(n);
  _p.resize(n*n);
  _temp.resize(n*n);
  _t1.resize(n*n);
  reset();
}

void kalman_exp::reset() {
  size_t const n = _gain.size();
  std::fill_n(_gain.begin(), n,   0);
  std::fill_n(_p.begin(),    n*n, 0);
  for (size_t i=0; i<n; ++i)
    _p[n*i +i] = gr_complex(0.1f); // TODO?
}

gr_complex const* kalman_exp::update(gr_complex const* beg,
                                     gr_complex const* end) {
  assert(end-beg > 0);
  unsigned const n = end - beg;
  resize(n);

  // P = P/lambda
  volk_32f_s32f_multiply_32f((float*)&_p[0], (float const*)&_p[0], 1.0f/_lambda, 2*n*n);

  // gain = P*H^{\dagger}
  for (unsigned i=0; i<n; ++i) {
    _gain[i] = 0;
    volk_32fc_x2_conjugate_dot_prod_32fc(&_gain[i], &_p[n*i], beg, n);
  }

  // alpha = H*P*H^{\dagger} + R
  gr_complex alpha = 0;
  volk_32fc_x2_dot_prod_32fc(&alpha, &_gain[0], beg, n);
  alpha += _r;

  // gain = gain / real(alpha)
  volk_32f_s32f_multiply_32f((float*)&_gain[0], (float const*)&_gain[0], 1.0f/std::real(alpha), 2*n);

  // temp = 1 - G*H
  for (unsigned i=0; i<n; ++i) {
    volk_32fc_s32fc_multiply_32fc(&_temp[n*i], beg, -_gain[i], n);
    _temp[n*i+i] += 1.0f;
  }

  // T1 = temp * P
  // P  = T1 * temp^{\dagger} + G*R*G^{\dagger}
  for (unsigned i=0; i<n; ++i) {
    // P = P^T so we can use a VOLK kernel below
    for (unsigned j=0; j<n; ++j)
      std::swap(_p[n*i+j], _p[n*j+i]);

    for (unsigned j=0; j<n; ++j) {
      _t1[n*i+j] = 0.0f;
      volk_32fc_x2_dot_prod_32fc(&_t1[n*i+j], &_temp[n*i], &_p[n*j], n);
    }
    for (unsigned j=0; j<n; ++j) {
      _p[n*i+j] = 0;
      volk_32fc_x2_conjugate_dot_prod_32fc(&_p[n*i+j], &_t1[n*i], &_temp[n*j], n);
      _p[n*i+j] += _r * _gain[i]*std::conj(_gain[j]);
    }
  }

  return &_gain[0];
}

void kalman_exp::set_parameters(std::map<std::string, float> const& p) {
  _r      = p.at("r");
  _lambda = p.at("lambda");
}

} // namespace digitalhf
} // namespace gr
