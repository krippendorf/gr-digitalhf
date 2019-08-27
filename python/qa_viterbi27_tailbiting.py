#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2018 hcab14@mail.com.
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

import digitalhf.digitalhf_swig as digitalhf
import pmt
import numpy as np


def bit2llr(b):
    return 7.0*(1 - 2*b)

class ViterbiEncoder(object):
    def __init__(self, k, taps):
        self._state = np.zeros(k, dtype=np.uint16)
        self._taps  = taps

    def encode(self, bit):
        self._state    = np.roll(self._state, 1)
        self._state[0] = bit
        return [bit2llr(self._state.dot(tap) % 2) for tap in self._taps]

def run_test(N, data):
    decoder = data['dec'](*data['polys'])
    encoder = ViterbiEncoder(data['k'], data['taps'])
    k = data['k']
    for p,t in zip(data['polys'], data['taps']):
        _t = [b=='1' for b in np.binary_repr(p,k)]
        test = _t == t[::-1]
        if not test:
            raise Exception('inconsistent taps', _t, t[::-1])

    np.random.seed(123)
    bits = np.random.randint(2, size=N)
    M    = len(data['polys'])
    llr_encoded_bits = np.zeros(M*N, dtype=np.float64)

    for i in range(N-k+1,N):
        encoder.encode(bits[i])
    for i in range(0,N):
        llr_encoded_bits[M*i:M*(i+1)] = encoder.encode(bits[i])

    print(llr_encoded_bits[:14])
    llr_encoded_bits[3::4] = 0
    decoded_bits = np.roll(decoder.udpate(llr_encoded_bits), 0)
    print(decoded_bits.tolist())
    print(bits.tolist())
    print('quality:', decoder.quality(), 100*4/3.5*decoder.quality()/M/N)
    test = [np.all(decoded_bits == bits), abs(4/3.5*decoder.quality()-M*N)<1]
    print('test:', test)
    if not all(test):
        raise Exception(test)

def main():
    data = [{'dec'  : digitalhf.viterbi27,
             'polys': [0x6D, 0x4F],
             'k'    : 7,
             'taps' : [[1,0,1,1,0,1,1],
                       [1,1,1,1,0,0,1]]},
            {'dec'  : digitalhf.viterbi29,
             'polys': [0x11d, 0x1af],
             'k'    : 9,
             'taps' : [[1,0,1,1,1,0,0,0,1],
                       [1,1,1,1,0,1,0,1,1]]},
            {'dec'  : digitalhf.viterbi39,
             'polys': [0x127, 0x19B, 0x1ED],
             'k'    : 9,
             'taps' : [[1,1,1,0,0,1,0,0,1],
                       [1,1,0,1,1,0,0,1,1],
                       [1,0,1,1,0,1,1,1,1]]},
            {'dec'  : digitalhf.viterbi48,
             'polys': [0xB9, 0x9D, 0xD3, 0xF7],
             'k'    : 8,
             'taps' : [[1,0,0,1,1,1,0,1],
                       [1,0,1,1,1,0,0,1],
                       [1,1,0,0,1,0,1,1],
                       [1,1,1,0,1,1,1,1]]}]

    for d in data:
        run_test(45, d)

if __name__ == '__main__':
    main()
