// -*- C++ -*-

#include <cassert>
#include "kalman_hsu.hpp"
#include <volk/volk.h>
#include <iostream>
namespace gr {
namespace digitalhf {

filter_update::sptr kalman_hsu::make(float q=0.008f, float e=0.01f) {
  return filter_update::sptr(new kalman_hsu(q, e));
}

kalman_hsu::kalman_hsu(float q, float e)
  : _q(q)
  , _e(e)
  , _g()
  , _u()
  , _d()
  , _a()
  , _f()
  , _h() {
}

kalman_hsu::~kalman_hsu() {
}

void kalman_hsu::resize(size_t n) {
  if (_g.size() == n &&
      _u.size() == n*(n-1)/2 &&
      _d.size() == n &&
      _a.size() == n &&
      _f.size() == n &&
      _h.size() == n)
    return;
  _g.resize(n);
  _u.resize(n*(n-1)/2);
  _d.resize(n);
  _a.resize(n);
  _f.resize(n);
  _h.resize(n);
  reset();
}

void kalman_hsu::reset() {
  std::fill_n(_d.begin(), _d.size(),  1);
  std::fill_n(_u.begin(), _u.size(),  gr_complex(0));
}

gr_complex const* kalman_hsu::update(gr_complex const* beg,
                                     gr_complex const* end) {
  assert(end-beg > 0);
  unsigned const n = end - beg;
  resize(n);
  static float y=1.0;

  for (unsigned i=0; i<n; ++i)
    _g[i] /= y;

  _f[0] = std::conj(beg[0]);
  for (unsigned j=1; j<n; ++j) {
    _f[j] = _u[idx(0,j)]*std::conj(beg[0]) + std::conj(beg[j]);
    for (unsigned i=1; i<j; ++i) // TODO: -> SIMD
      _f[j] += _u[idx(i,j)]*std::conj(beg[i]);
  }

  for (unsigned j=0; j<n; ++j) { // TODO: -> SIMD
    _g[j] = _d[j]*_f[j];
  }
  _a[0] = e() + std::real(_g[0]*std::conj(_f[0]));
  for (unsigned j=1; j<n; ++j) {
    _a[j] = _a[j-1] + std::real(_g[j]*std::conj(_f[j]));
  }

  float const hq = 1 + q();
  float const ht = _a[n-1]*q();

  /*float */ y = 1/(_a[0] + ht);
  _d[0] = _d[0]*hq*(e() + ht)*y;

  for (unsigned j=1; j<n; ++j) {
    float const b = _a[j-1] + ht;
    _h[j] = -_f[j]*y;

    y = 1/(_a[j] + ht);
    _d[j] = _d[j]*hq*b*y;

    for (unsigned i=0; i<j; ++i) { // TODO -> SIMD
      gr_complex const b0 = _u[idx(i,j)];
      _u[idx(i,j)] = b0 + _h[j]*std::conj(_g[i]);
      _g[i] += _g[j]*std::conj(b0);
    }
  }

  for (unsigned i=0; i<n; ++i) {
    _d[i] = 1.0f/(10.0f+1.0f/_d[i]);// regularization
    _g[i] *= y;
  }

  return &_g[0];
}

void kalman_hsu::set_parameters(std::map<std::string, float> const& p) {
  _q = p.at("q");
  _e = p.at("E");
}

} // namespace digitalhf
} // namespace gr
