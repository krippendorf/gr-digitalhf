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

#ifndef INCLUDED_DIGITALHF_OQPSK_DEMODULATOR_IMPL_H
#define INCLUDED_DIGITALHF_OQPSK_DEMODULATOR_IMPL_H

#include <gnuradio/blocks/control_loop.h>

#include <digitalhf/oqpsk_demodulator.h>

namespace gr {
namespace digitalhf {

class my_control_loop : public blocks::control_loop {
public:
  my_control_loop(float loop_bw, float max_freq, float min_freq)
    : blocks::control_loop(loop_bw, max_freq, min_freq) {}
  virtual ~my_control_loop();

  virtual void phase_wrap(float x);
protected:
} ;

class oqpsk_demodulator_impl : public oqpsk_demodulator
{
private:
  my_control_loop _pll_plus;
  my_control_loop _pll_minus;

  float _soft_dec_x;
  float _soft_dec_y;
  float _soft_dec_x_last;
  float _soft_dec_y_last;
  float _code_cos_last;
  float _code_sin_last;
public:
  oqpsk_demodulator_impl(float fs,
                         float fd,
                         float loop_bw,
                         float df);
  virtual ~oqpsk_demodulator_impl();

  int general_work(int noutput_items,
                   gr_vector_int& ninput_items,
                   gr_vector_const_void_star &input_items,
                   gr_vector_void_star &output_items);

protected:

};

} // namespace digitalhf
} // namespace gr

#endif /* INCLUDED_DIGITALHF_OQPSK_DEMODULATOR_IMPL_H */
