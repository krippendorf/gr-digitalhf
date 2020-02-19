## -*- python -*-

from __future__ import print_function
import numpy as np
from . import common

from digitalhf.digitalhf_swig import viterbi27

## 192 = 6*32                        0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1 # 32
PREAMBLE = common.n_psk(8, np.array([7,0,3,4,1,1,1,0,2,6,1,5,1,7,0,3,5,4,2,2,6,1,2,2,0,4,5,4,1,2,2,6,   # 1
                                     7,0,7,0,1,1,5,4,2,6,5,1,1,7,4,7,5,4,6,6,6,1,6,6,0,4,1,0,1,2,6,2,   # 2
                                     7,4,7,4,1,5,5,0,2,2,5,5,1,3,4,3,5,0,6,2,6,5,6,2,0,0,1,4,1,6,6,6,   # 3
                                     7,0,3,4,5,5,5,4,2,6,1,5,5,3,4,7,5,4,2,2,2,5,6,6,0,4,5,4,5,6,6,2,   # 4
                                     7,4,3,0,1,5,1,4,2,2,1,1,1,3,0,7,5,0,2,6,6,5,2,6,0,0,5,0,1,6,2,2,   # 5
                                     7,4,3,0,5,1,5,0,2,2,1,1,5,7,4,3,5,0,2,6,2,1,6,2,0,0,5,0,5,2,6,6])) # 6

## ---- data scrambler -----------------------------------------------------------
class ScrambleData(object):
    """data scrambling sequence generator"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._state = 0xBAD
        self._counter = 0

    def next(self):
        if self._counter == 160:
            self.reset()
        for j in range(8):
            self._advance()
        self._counter += 1
        return self._state&7

    def _advance(self):
        msb = self._state>>11
        self._state = (self._state<<1)&4095
        if msb:
            self._state ^= 0x053
        return self._state

## ---- constellation indices ---------------------------------------------------
MODE_BPSK=0
MODE_QPSK=1


## ---- physcal layer class -----------------------------------------------------
class PhysicalLayer(object):
    """Physical layer description for MIL-STD-188-110 Appendix D = STANAG 4539"""

    def __init__(self, sps):
        """intialization"""
        self._sps     = sps
        self._frame_counter = -1
        self._constellations = [self.make_psk(2, [0,1]),
                                self.make_psk(4, [0,1,3,2])]
        self._preamble    = self.get_preamble()
        self._scr_data    = ScrambleData()
        self._intl_idx    = np.mod(17*np.arange(90, dtype=np.uint32), 90)
        self._viterbi_decoder  = viterbi27(0x6d, 0x4f)
        self._viterbi_decoder2 = viterbi27(0x67, 0x5d)

        ##self._data     = self.get_data()

    def get_constellations(self):
        return self._constellations

    def get_next_frame(self, symbols):
        """returns a tuple describing the frame:
        [0] ... known+unknown symbols and scrambling
        [1] ... modulation type after descrambling
        [2] ... a boolean indicating if the processing should continue
        [3] ... a boolean indicating if the soft decision for the unknown
                symbols are saved"""
        print('-------------------- get_frame --------------------',
              self._frame_counter)

        success = True

        if len(symbols) == 0:
            self._frame_counter = -1
            self._scr_data.reset()

        if self._frame_counter == -1:
            self._frame_counter += 1
            return [self.get_preamble(),MODE_BPSK,success,False]
        else:
            z = np.mean(symbols[-19:])
            print('CHECK:', z)
            success = np.real(z) > 0.5 and np.abs(np.imag(z)) < 0.5
            a = np.zeros(64, dtype=common.SYMB_SCRAMBLE_DTYPE)
            a['scramble']   = common.n_psk(8, np.array([self._scr_data.next() for _ in range(64)]))
            a['symb'][-19:] = a['scramble'][-19:]
            self._frame_counter += 1
            return [a,MODE_QPSK,success,success]

    def get_doppler(self, iq_samples):
        """returns a tuple
        [0] ... quality flag
        [1] ... doppler estimate (rad/symbol) if available"""
        print('-------------------- get_doppler --------------------',
              self._frame_counter,len(iq_samples))
        r = {'success':     False, ## -- quality flag
             'use_amp_est': False, ##self._frame_counter < 0,
             'doppler':     0}     ## -- doppler estimate (rad/symb)
        sps  = self._sps
        _,zp = self.get_preamble_z()
        wlen = 32
        cc   = np.correlate(iq_samples, zp[0:wlen])
        imax = np.argmax(np.abs(cc[0:wlen*sps]))
        idx  = np.arange(wlen*sps)
        pks  = [np.vdot(zp[             i*wlen*sps+idx],
                        iq_samples[imax+i*wlen*sps+idx])
                for i in range((len(iq_samples)-imax) // (wlen*sps))]
        print('get_doppler pks: ', pks, np.angle(pks))
        r['doppler'] = common.freq_est(pks)/(wlen*sps)
        print('get_doppler doppler: ', r['doppler'])
        r['success'] = True
        return r

    def set_mode(self, mode):
        pass

    def decode_soft_dec(self, soft_dec):
        print('decode_soft_dec', self._frame_counter)
        assert(len(soft_dec) == 90)
        deintl_soft_dec = np.zeros(90, dtype=np.float64)
        deintl_soft_dec[self._intl_idx] = soft_dec
        print('decode_soft_dec', deintl_soft_dec)
        decoded_bits = []
        quality = 0.0
        if self._frame_counter == 2:
            self._viterbi_decoder.reset()
            decoded_bits = self._viterbi_decoder.udpate(deintl_soft_dec)
            quality = 100.0*self._viterbi_decoder.quality()/(2*len(decoded_bits))
            print('bits=', len(decoded_bits), decoded_bits)
            print('quality={}% ({},{})'.format(quality,
                                               self._viterbi_decoder.quality(),
                                               len(decoded_bits)))
        else:
            depunct_deintl_soft_dec = np.zeros(90//3*4, dtype=np.float64)
            depunct_deintl_soft_dec[0::4] = deintl_soft_dec[0::3]
            depunct_deintl_soft_dec[1::4] = deintl_soft_dec[1::3]
            depunct_deintl_soft_dec[2::4] = deintl_soft_dec[2::3]
            depunct_deintl_soft_dec[3::4] = 0
            self._viterbi_decoder2.reset()
            decoded_bits = self._viterbi_decoder2.udpate(depunct_deintl_soft_dec)
            quality = 100.0*4/3.5*self._viterbi_decoder2.quality()/(2*len(decoded_bits))
            print('bits=', len(decoded_bits), decoded_bits)
            print('quality={}% ({},{})'.format(quality,
                                               self._viterbi_decoder2.quality(),
                                               len(decoded_bits)))
        if quality > 90:
            return decoded_bits,quality
        else:
            return [],0.0


    def get_preamble(self):
        """preamble symbols + scrambler"""
        return common.make_scr(PREAMBLE,PREAMBLE)

    def get_preamble_z(self):
        """preamble symbols for preamble correlation"""
        a = self.get_preamble()
        return 2,np.array([z for z in a['symb'] for _ in range(self._sps)])

    @staticmethod
    def make_psk(n, gray_code):
        """generates n-PSK constellation data"""
        c = np.zeros(n, dtype=common.CONST_DTYPE)
        c['points']  = common.n_psk(n,np.arange(n))
        c['symbols'] = gray_code
        return c

if __name__ == '__main__':
    print(PREAMBLE)
    z = common.n_psk(8,PREAMBLE)
    cc = [np.sum(z[0:32]*np.conj(z[32*i:32*i+32])) for i in range(6)]
    print(np.abs(cc))
    print(np.angle(cc)/np.pi*4)
    print(z)
