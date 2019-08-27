/* -*- c++ -*- */
/*
 * Copyright 2018 hcab14@gmail.com.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cmath>

#include <gnuradio/io_signature.h>
#include <gnuradio/expj.h>
#include <gnuradio/logger.h>

#include "vector_pll_cc_impl.h"

namespace gr {
namespace digitalhf {

vector_pll_cc::sptr
vector_pll_cc::make(float samp_rate,
                    size_t order,
                    std::vector<int> freq_multipliers)
{
  return gnuradio::get_initial_sptr
    (new vector_pll_cc_impl(samp_rate,
                            order,
                            freq_multipliers));
}

/*
 * The private constructor
 */
vector_pll_cc_impl::vector_pll_cc_impl(float samp_rate,
                                       size_t order,
                                       std::vector<int> freq_multipliers)
  : gr::sync_block("vector_pll_cc",
                   gr::io_signature::make(1, 1, freq_multipliers.size() * sizeof(gr_complex)),
                   gr::io_signature::make(1, 1, freq_multipliers.size() * sizeof(gr_complex)))
  , _ts(1.0/samp_rate)
  , _order(order)
  , _freq_multipliers(freq_multipliers)
  , _control_loop(6.28/600.0, -6.28/600.0*10, 6.28/600.0*10)
  , _theta (freq_multipliers.size(), 0.0f)
  , _dtheta(freq_multipliers.size(), 0.0f)
  , _absz(freq_multipliers.size(), 1.0f)
  , _b({+0.6284830097698858,
        -0.6281540229565162})
  , _uf(0)
  , _ud(0)
{
  GR_LOG_DECLARE_LOGPTR(d_logger);
  GR_LOG_ASSIGN_LOGPTR(d_logger, "vector_pll_cc");
  for (int i=0; i<_freq_multipliers.size(); ++i) {
    std::cout << "FM: " << i <<" " << _freq_multipliers[i] << std::endl;
  }
}

vector_pll_cc_impl::~vector_pll_cc_impl()
{
}

int
vector_pll_cc_impl::work(int noutput_items,
                         gr_vector_const_void_star &input_items,
                         gr_vector_void_star &output_items)
{
  gr::thread::scoped_lock lock(d_setlock);

  size_t const vlen  = _freq_multipliers.size();
  const gr_complex *in  = (const gr_complex*)input_items[0];
  gr_complex *out = (gr_complex*)output_items[0];
  _control_loop.frequency_limit();

  std::vector<float> error(vlen);

  float const mu = 5e-2;

  for (int i=0; i<noutput_items; ++i) {
    for (int j=0; j<vlen; ++j) {
      _theta[j] += _uf * _freq_multipliers[j] * _ts + _dtheta[j];
      _theta[j]  = std::fmod(_theta[j], 2*M_PI*_order);
      out[j]     = gr_expj(-_theta[j]/_order) * in[j];
    }

    float weight=0, sum_weights=0;
    for (int j=0; j<vlen; ++j) {
      gr_complex z = in[j];
      for (size_t k=1; k<_order; k*=2)
        z *= z;
      _absz[j] = weight = 0.99*_absz[j] + 0.01*std::abs(z);
      sum_weights += weight;
      gr_complex const x = gr_expj(-_theta[j]) * z;
      error[j] = std::imag(x)/std::abs(x) * weight;
    }

    float sum_error = 0;
    for (int j=0; j<vlen; ++j) {
      error[j]   *= vlen / sum_weights;
      sum_error  += error[j];
      _dtheta[j]  = mu * error[j];
    }

    float const ud = sum_error / vlen;
    _uf += _b[0]*ud + _b[1]*_ud;
    _ud  = ud;

    in  += vlen;
    out += vlen;
  }
  return noutput_items;
}

} /* namespace digitalhf */
} /* namespace gr */
