// -*- C++ -*-

#include <cassert>
#include "rls.hpp"
#include <volk/volk.h>

namespace gr {
namespace digitalhf {

filter_update::sptr rls::make(float delta, float lambda) {
  return filter_update::sptr(new rls(delta, lambda));
}

rls::rls(float delta, float lambda)
  : _delta(delta)
  , _lambda(lambda)
  , _gain()
  , _inv_corr()
  , _tmp() {
}

rls::~rls() {
}

void rls::resize(size_t n) {
  if (_gain.size() == n && _inv_corr.size() == n*n)
    return;
  _gain.resize(n);
  _inv_corr.resize(n*n);
  _pu.resize(n);
  _tmp.resize(n);
  reset();
}

void rls::reset() {
  size_t const n = _gain.size();
  std::fill_n(_gain.begin(),     n,   0);
  std::fill_n(_inv_corr.begin(), n*n, 0);
  for (size_t i=0; i<n; ++i)
    _inv_corr[n*i +i] = gr_complex(_delta, 0);
}

gr_complex const* rls::update(gr_complex const* beg,
                              gr_complex const* end) {
  assert(end-beg > 0);
  unsigned const n = end - beg;
  resize(n);

  for (unsigned i=0; i<n; ++i)
    volk_32fc_x2_dot_prod_32fc(&_pu[i], &_inv_corr[n*i], beg, n);

  gr_complex uPu = 0;
  volk_32fc_x2_conjugate_dot_prod_32fc(&uPu, &_pu[0], beg, n);

  volk_32fc_conjugate_32fc(&_tmp[0], &_pu[0], n);
  volk_32f_s32f_multiply_32f((float*)&_gain[0], (float const*)&_tmp[0], 1.0f/(_lambda + std::real(uPu)/_lambda), 2*n);
//  volk_32fc_s32fc_multiply_32fc(&_gain[0], &_tmp[0], 1/(_lambda + uPu), n);

  for (unsigned i=0; i<n; ++i) {
    unsigned const k = n*i;
    volk_32fc_s32fc_multiply_32fc(&_tmp[0], &_gain[0], -_pu[i], n);
    volk_32fc_x2_add_32fc(&_inv_corr[k], &_inv_corr[k], &_tmp[0], n);
    volk_32f_s32f_multiply_32f((float*)&_inv_corr[k], (float const*)&_inv_corr[k], 1.0f/_lambda, 2*n);
  }

  return _gain.data();
}

void rls::set_parameters(std::map<std::string, float> const& p) {
  _delta  = p.at("delta");
  _lambda = p.at("lambda");
}

} // namespace digitalhf
} // namespace gr
