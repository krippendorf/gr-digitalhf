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

#include "vector_early_late_cc_impl.h"

namespace gr {
namespace digitalhf {

vector_early_late_cc::sptr
vector_early_late_cc::make(unsigned sps,
                           float alpha)
{
  return gnuradio::get_initial_sptr
    (new vector_early_late_cc_impl(sps, alpha));
}

/*
 * The private constructor
 */
vector_early_late_cc_impl::vector_early_late_cc_impl(unsigned sps,
                                                     float alpha)
  : gr::block("vector_early_late_cc",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex)))
  , _sps(sps)
  , _alpha(alpha)
  , _counter(0)
  , _err(0.0f)
  , _t(0.0f)
{
  GR_LOG_DECLARE_LOGPTR(d_logger);
  GR_LOG_ASSIGN_LOGPTR(d_logger, "vector_early_late_cc");

  // -sps/2 ... -1 0 1 ... sps/2
  // [   early   ]   [  late   ]
  set_history(1 + sps);
}

vector_early_late_cc_impl::~vector_early_late_cc_impl()
{
}

int
vector_early_late_cc_impl::general_work(int noutput_items,
                                        gr_vector_int& ninput_items,
                                        gr_vector_const_void_star& input_items,
                                        gr_vector_void_star& output_items)
{
  gr::thread::scoped_lock lock(d_setlock);

  gr_complex const *in = (gr_complex const *)input_items[0];
  gr_complex *out      = (gr_complex *)output_items[0];

  int const nin = ninput_items[0] - _sps;
//  std::cout << "vector_early_late_cc_impl::general_work nin,sps= " << " " << nin << " " << _sps << " " << history() << std::endl;
  assert(nin > 1);
  if (nin < 1)
    return 0;

  int nout = 0;
  int nin_processed = 0;
  int i = history();
  for (; i<ninput_items[0]-_sps/2 && nout<noutput_items; ++i) {
    if (_counter == std::floor(_t)) {
      // evaluate early and late
      gr_complex early(0), late(0);
//      float early(0), late(0);
      for (int j=0; j<_sps; ++j) {
        early += std::real(in[i-j]);
        late  += std::real(in[i+1+j]);
      }
      // output symbol
      gr_complex _out(0);
      for (int j=0; j<_sps; ++j) {
        _out += in[i-j];
      }
//      out[nout++] = _out*(1.0f/(_sps-2));
      out[nout++] = _out*(1.0f/_sps);

      float const error = (std::real(early - late)*std::real(in[i]) +
                           std::imag(early - late)*std::imag(in[i]));
//      float const error = (std::real(early) - std::real(late))*std::real(in[i]);
      _err = (1.0f - _alpha)*_err + _alpha * error;
      _t  += float(_sps) - float(_counter) + 0.5*_err;
      std::cout << "EL: err= "<< _err << " early/late= " << std::abs(early) << " " << std::abs(late) << " t= " << _t << std::endl;
      _counter = 0;
    }
    consume(0, 1);
    _counter +=1;
  }
  return nout;
}

} /* namespace digitalhf */
} /* namespace gr */
