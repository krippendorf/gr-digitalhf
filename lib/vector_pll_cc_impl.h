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

#ifndef INCLUDED_DIGITALHF_VECTOR_PLL_CC_IMPL_H
#define INCLUDED_DIGITALHF_VECTOR_PLL_CC_IMPL_H

#include <gnuradio/blocks/control_loop.h>

#include <digitalhf/vector_pll_cc.h>

#include <array>

namespace gr {
namespace digitalhf {

class vector_pll_cc_impl : public vector_pll_cc
{
private:
  float  _ts;     // sample time (sec)
  size_t _order;  //
  std::vector<int> _freq_multipliers;
  gr::blocks::control_loop _control_loop;
  std::vector<float> _theta;
  std::vector<float> _dtheta;
  std::vector<float> _absz;
  std::array<float, 2> _b;
  float _uf;
  float _ud;
public:
  vector_pll_cc_impl(float samp_rate,
                     size_t order,
                     std::vector<int> freq_multipliers);
  virtual ~vector_pll_cc_impl();

  int work(int noutput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

protected:

};

} // namespace digitalhf
} // namespace gr

#endif /* INCLUDED_DIGITALHF_VECTOR_PLL_CC_IMPL_H */
