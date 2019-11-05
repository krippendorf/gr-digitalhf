/* -*- c++ -*- */
/*
 * Copyright 2019 hcab14@gmail.com.
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
#include <gnuradio/math.h>
#include <gnuradio/sincos.h>

#include "oqpsk_demodulator_impl.h"

namespace gr {
namespace digitalhf {

my_control_loop::~my_control_loop() {
}

void
my_control_loop::phase_wrap(float x)
{
  //  d_phase = fmodf(d_phase, x);
  while (d_phase > x)
    d_phase -= x;
  while (d_phase < -x)
    d_phase += x;
}

oqpsk_demodulator::sptr
oqpsk_demodulator::make(float fs,
                        float fd,
                        float loop_bw,
                        float df)
{
  return gnuradio::get_initial_sptr
    (new oqpsk_demodulator_impl(fs, fd, loop_bw, df));
}

/*
 * The private constructor
 */
oqpsk_demodulator_impl::oqpsk_demodulator_impl(float fs,
                                               float fd,
                                               float loop_bw,
                                               float df)
  : gr::block("oqpsk_demodulator",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(2, 2, sizeof(float)))
  , _pll_plus (loop_bw,
               (+0.5f*fd + df)/fs*2*M_PI,
               (+0.5f*fd - df)/fs*2*M_PI)
  , _pll_minus(loop_bw,
               (-0.5f*fd + df)/fs*2*M_PI,
               (-0.5f*fd - df)/fs*2*M_PI)
  , _soft_dec_x(0.0f)
  , _soft_dec_y(0.0f)
  , _soft_dec_x_last(0.0f)
  , _soft_dec_y_last(0.0f)
  , _code_cos_last(0.0f)
  , _code_sin_last(0.0f)
{
  GR_LOG_DECLARE_LOGPTR(d_logger);
  GR_LOG_ASSIGN_LOGPTR(d_logger, "oqpsk_demodulator");
}

oqpsk_demodulator_impl::~oqpsk_demodulator_impl()
{
}

int
oqpsk_demodulator_impl::general_work(int noutput_items,
                                     gr_vector_int& ninput_items,
                                     gr_vector_const_void_star &input_items,
                                     gr_vector_void_star &output_items)
{
  gr::thread::scoped_lock lock(d_setlock);
  gr_complex const* in = static_cast<gr_complex const*>(input_items[0]);
  float* out_x = static_cast<float*>(output_items[0]);
  float* out_y = static_cast<float*>(output_items[1]);

  int nout = 0;
  int i = 0;
  for (; i<ninput_items[0] && nout < noutput_items; ++i) {
    gr_complex const sample = *in++;

    float const carr_phase = 0.25f*(_pll_plus.get_phase() + _pll_minus.get_phase());
    float const code_phase = 0.25f*(_pll_plus.get_phase() - _pll_minus.get_phase());

    float code_sin(0.0f), code_cos(0.0f);
    gr::sincosf(code_phase, &code_sin, &code_cos);

    if (std::signbit(code_cos) != std::signbit(_code_cos_last)) {
      _soft_dec_x_last = _soft_dec_x;
      _soft_dec_x = 0.0f;
      *out_x++ = _soft_dec_x_last;
      *out_y++ = _soft_dec_y_last;
      nout += 1;
    }

    if (std::signbit(code_sin) != std::signbit(_code_sin_last)) {
      _soft_dec_y_last = _soft_dec_y;
      _soft_dec_y = 0.0f;
      *out_x++ = _soft_dec_x_last;
      *out_y++ = _soft_dec_y_last;
      nout += 1;
    }

    _code_cos_last = code_cos;
    _code_sin_last = code_sin;

    gr_complex const z = sample*gr_expj(-carr_phase);
    _soft_dec_x += code_cos * std::real(z);
    _soft_dec_y += code_sin * std::imag(z);

    gr_complex const z2 = sample*sample;
    float const error_plus = gr::fast_atan2f(z2 * gr_expj(-_pll_plus.get_phase()));
    _pll_plus.advance_loop(error_plus);
    _pll_plus.phase_wrap(8*M_PI);
    _pll_plus.frequency_limit();

    float const error_minus = gr::fast_atan2f(z2 * gr_expj(-_pll_minus.get_phase()));
    _pll_minus.advance_loop(error_minus);
    _pll_minus.phase_wrap(8*M_PI);
    _pll_minus.frequency_limit();
  }
  consume_each(i);
  return nout;
}

} /* namespace digitalhf */
} /* namespace gr */
