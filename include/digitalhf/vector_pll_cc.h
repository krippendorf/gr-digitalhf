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


#ifndef INCLUDED_DIGITALHF_VECTOR_PLL_CC_H
#define INCLUDED_DIGITALHF_VECTOR_PLL_CC_H

#include <digitalhf/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace digitalhf {

/*!
 * \brief <+description of block+>
 * \ingroup digitalhf
 *
 */
class DIGITALHF_API vector_pll_cc : virtual public gr::sync_block
{
  public:
  typedef boost::shared_ptr<vector_pll_cc> sptr;

  /*!
   * \brief Return a shared_ptr to a new instance of digitalhf::vector_pll_cc.
   *
   * To avoid accidental use of raw pointers, digitalhf::vector_pll_cc's
   * constructor is in a private implementation
   * class. digitalhf::vector_pll_cc::make is the public interface for
   * creating new instances.
   */
  static sptr make(float samp_rate,
                   size_t order,
                   std::vector<int> freq_multipliers);

};

} // namespace digitalhf
} // namespace gr

#endif /* INCLUDED_DIGITALHF_VECTOR_PLL_CC_H */

