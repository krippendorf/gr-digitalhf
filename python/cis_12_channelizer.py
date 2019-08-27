#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 hcab14@gmail.com.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

import importlib
from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio.filter import pfb
from gnuradio import gr
import pmt
import digitalhf
import numpy as np

class cis_12_channelizer(gr.hier_block2):
    """
    docstring for block CIS-12 channelizer
    """
    def __init__(self):
        gr.hier_block2.__init__(self,
                                "cis_12_channelizer",
                                gr.io_signature(1, 1, gr.sizeof_gr_complex), # Input signature
                                gr.io_signature(2, 2, gr.sizeof_gr_complex)) # Output signature

        self.samp_rate = samp_rate = 12000.0
        self.analog_pll_refout_cc_0 = analog.pll_refout_cc(2*np.pi/1000,
                                                           2*np.pi*3270/samp_rate,
                                                           2*np.pi*3330/samp_rate)
        taps_pfb = firdes.low_pass_2(1,
                                     self.samp_rate,
                                     0.45*self.samp_rate/60,
                                     0.05*self.samp_rate/60,
                                     attenuation_dB=100,
                                     window=filter.firdes.WIN_BLACKMAN_HARRIS)
        taps_3300 = firdes.band_pass_2(1,
                                       self.samp_rate,
                                       3200,
                                       3400,
                                       100,
                                       attenuation_dB=60,
                                       window=filter.firdes.WIN_BLACKMAN_HARRIS)
        print('N=', len(taps_3300))
        self.filter_fir_filter_ccf_0 = filter.fir_filter_ccf(1,taps_3300)

        self.blocks_multiply_conjugate_cc_0 = blocks.multiply_conjugate_cc(1)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex,
                                           (len(taps_3300)-1) // 2)

        self.pfb_channelizer_ccf_0 = pfb.channelizer_ccf(
            60,
            (taps_pfb),
            1.0,
            100)
        chmap = 60-(2+np.arange(12))
        self.pfb_channelizer_ccf_0.set_channel_map((chmap.tolist()))
        self.pfb_channelizer_ccf_0.set_channel_map(([47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]))
        self.pfb_channelizer_ccf_0.declare_sample_delay(0)

        self.blocks_streams_to_vector_0 = blocks.streams_to_vector(gr.sizeof_gr_complex, 12)
        self.blocks_streams_to_vector_1 = blocks.streams_to_vector(gr.sizeof_gr_complex, 48)
        self.blocks_vector_to_stream_0 = blocks.vector_to_stream(gr.sizeof_gr_complex, 12)

        self.blocks_vector_pll_cc_0 = digitalhf.vector_pll_cc(samp_rate = 600.0,
                                                              order = 2,
                                                              freq_multipliers = [13,12,11,10,9,8,7,6,5,4,3,2])

        self.blocks_resamplers = [[] for _ in range(12)]
        for i in range(12):
            self.blocks_resamplers[i] = filter.rational_resampler_ccf(3,1)

        self.connect((self, 0),
                     (self.filter_fir_filter_ccf_0, 0),
                     (self.analog_pll_refout_cc_0, 0),
                     (self.blocks_multiply_conjugate_cc_0, 1))
        self.connect((self, 0),
                     (self.blocks_delay_0),
                     (self.blocks_multiply_conjugate_cc_0, 0))
        self.connect((self.blocks_multiply_conjugate_cc_0, 0),
                     (self.pfb_channelizer_ccf_0, 0))
        self.connect((self.blocks_multiply_conjugate_cc_0, 0), (self, 1))
        #self.connect((self.filter_fir_filter_ccf_0, 0),
        #             (self, 1))

        for i in range(12):
            self.connect((self.pfb_channelizer_ccf_0, i),
                         (self.blocks_resamplers[i], 0),
                         (self.blocks_streams_to_vector_0, i))

        self.blocks_null_sink_0 = blocks.null_sink(48*gr.sizeof_gr_complex)
        for i in range(48):
            self.connect((self.pfb_channelizer_ccf_0, 12+i), (self.blocks_streams_to_vector_1, i))
        self.connect(self.blocks_streams_to_vector_1, self.blocks_null_sink_0)

        self.connect(self.blocks_streams_to_vector_0,
                     self.blocks_vector_pll_cc_0,
                     self.blocks_vector_to_stream_0,
                     self)
        #self.connect((self,0),
        #             (self.analog_pll_refout_cc_0, 0),
        #             (self,0))
