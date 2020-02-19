## -*- python -*-

from __future__ import print_function
import numpy as np
from . import common

PREAMBLE = common.n_psk(8, np.array([ 2,6,4,4,6,4,6,2,6,0 #  1
                                     ,2,2,0,4,6,2,2,0,2,6 #  2
                                     ,0,6,4,0,2,0,6,6,6,4 #  3
                                     ,0,6,0,6,2,2,0,4,2,4 #  4
                                     ,0,2,2,2,2,6,4,6,6,2 #  5
                                     ,4,6,4,6,2,6,0,2,4,0 #  6
                                     ,0,0,6,6,2,6,2,2,0,2 #  7
                                     ,4,4,6,4,6,0,4,0,6,6 #  8
                                     ,2,2,0,0,6,6,4,0,4,0 #  9
                                     ,0,6,6,6,4,6,4,6,0,2 # 10
                                     ,2,6,0,0,0,2,6,2,0,0 # 11
                                     ,6,2,6,0,4,6,6,4,0,6 # 12
                                     ,2,6,2,4,4,2,0,6,2,6 # 13
                                     ,0,0,4,2,4,0,6,0,4,4 # 14
                                     ,2,2,6,0,2,2,0,6,4,2 # 15
                                     ,2,4,0,6,0,4,6,4,0,2 # 16
                                     ,2,0,2,2,2,2,4,4,0,2 # 17
                                     ,6,2,2,4,6,6,6,2,6,4 # 18
                                     ,2,0,0,0,2,2,4,0,0,6 # 19
                                     ,6,4,2,0,0,0,0,2,0,4 # 20
                                     ,2,2,4])) ## 203 symbols

## ---- constellatios -----------------------------------------------------------
BPSK=np.array(list(zip(np.exp(2j*np.pi*np.arange(2)/2), [0,1])), common.CONST_DTYPE)
QPSK=np.array(list(zip(np.exp(2j*np.pi*np.arange(4)/4), [0,1,3,2])), common.CONST_DTYPE)
PSK8=np.array(list(zip(np.exp(2j*np.pi*np.arange(8)/8), [0,1,3,2,6,7,5,4])), common.CONST_DTYPE)

## ---- constellation indices ---------------------------------------------------
MODE_BPSK=0
MODE_QPSK=1
MODE_8PSK=2


## ---- physcal layer class -----------------------------------------------------
class PhysicalLayer(object):
    """Physical layer description for STANAG 4539 Appendix D"""

    def __init__(self, sps):
        """intialization"""
        self._sps = sps
        self._frame_counter = -1
        self._constellations = [BPSK, QPSK, PSK8]
        self._preamble = self.get_preamble()
        self._mode = {}
        self._mode_description = 'UNKNOWN'

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
              self._frame_counter, len(symbols))
        success = True
        if self._frame_counter == -1: ## preamble mode
            if len(symbols) == 0:
                return [self._preamble,MODE_BPSK,success,False]
            else:
                success = self.decode_preamble(symbols)
                return [self._preamble,MODE_BPSK,success,False]
        # else: ## data mode
        #     self._frame_counter += 1
        #     ##print('test:', symbols[self._mode['unknown']:], np.mean(np.real(symbols[self._mode['unknown']:])))
        #     if self._mode['known'] == 0: ## orthogonal WALSH modulation
        #         success = True
        #         for i in range(5):
        #             a = symbols[32*i:32*(i+1)]
        #             success &= np.max(np.imag(np.mean(a.reshape(8,4),0))) < 0.25
        #     elif self._frame_counter < self._num_frames_per_block-2:
        #         success = np.mean(np.real(symbols[self._mode['unknown']:])) > 0.4 or np.max(np.imag(symbols[self._mode['unknown']:])) < 0.6
        #     if not success:
        #         print('aborting: ', symbols[self._mode['unknown']:])# np.mean(np.real(symbols[self._mode['unknown']:])),
        #         #np.max(np.imag(symbols[self._mode['unknown']:])))
        #     return [self.get_next_data_frame(success),self._mode['ci'],success,success]

    def get_next_data_frame(self, success):
        # if self._frame_counter == self._num_frames_per_block:
        #     self._frame_counter = 0
        # scramble_for_frame = common.n_psk(8, np.array([self._scr_data.next()
        #                                                for _ in range(self._frame_len)]))
        # a = common.make_scr(scramble_for_frame, scramble_for_frame)
        # n_unknown = self._mode['unknown']
        # a['symb'][0:n_unknown] = 0
        # if self._mode['known'] != 0 and self._frame_counter >= self._num_frames_per_block-2:
        #     idx_d1d2 = self._frame_counter - self._num_frames_per_block + 2;
        #     a['symb'][n_unknown  :n_unknown+ 8] *= common.n_psk(2, WALSH8[self._d1d2[idx_d1d2]][:])
        #     a['symb'][n_unknown+8:n_unknown+16] *= common.n_psk(2, WALSH8[self._d1d2[idx_d1d2]][:])
        # if not success:
        #     self._frame_counter = -1
        #     self._pre_counter = -1
        # return a
        return False

    def get_doppler(self, iq_samples):
        """quality check and doppler estimation for preamble"""
        r = {'success':     False, ## -- quality flag
             'use_amp_est': False, ##self._frame_counter < 0,
             'doppler':     0}     ## -- doppler estimate (rad/symb)
        if len(iq_samples) != 0:
            sps  = self._sps
            ## find starting point
            _,zp = self.get_preamble_z()
            cc   = np.correlate(iq_samples, zp[0:40*sps])
            imax = np.argmax(np.abs(cc[0:20*sps]))
            print('imax=', imax, len(iq_samples), len(cc))
            apk = np.abs(cc[imax])
            tpk = np.abs(cc[imax+20*sps])
            print('imax=', imax, 'apk=', apk, 'tpk=', tpk)
            r['success'] = np.bool(apk > 2*tpk)
            if r['success']:
                idx = np.arange(40*sps)
                pks = [np.vdot(zp[             i*40*sps+idx],
                               iq_samples[imax+i*40*sps+idx])
                       for i in range(4)]
                r['doppler'] = common.freq_est(pks)/(40*sps)
                print('success=', r['success'], 'doppler=', r['doppler'],
                      np.abs(np.array(pks)),
                      np.angle(np.array(pks)))
        return r

    def decode_preamble(self, symbols):
        print('decode_preamble', symbols)
        mean_symb = np.mean(symbols[-40:])
        success = np.real(mean_symb) > 0.6
        print('decode_preamble', mean_symb, success)
        return success
        # data = [FROM_WALSH8[np.packbits
        #                     (np.real
        #                      (np.sum
        #                       (symbols[i:i+32].reshape((4,8)),0))<0)[0]]
        #         for i in range(0,15*32,32)]
        # print('data=',data)
        # self._pre_counter = sum([(x&3)*(1<<2*y) for (x,y) in zip(data[11:14][::-1], range(3))])
        # self._d1d2 = data[9:11]
        # print('MODE:', data[9:11])
        # self._mode = mode = MODE[data[9]][data[10]]
        # self._block_len = 11520 if mode['interleaver'][0] == 'L' else 1440
        # self._frame_len = mode['known'] + mode['unknown']
        # if mode['known'] == 0: ## orthogonal WALSH modulation
        #     self._num_frames_per_block = mode['interleaver'][1]*mode['interleaver'][2]/2*32/160
        # else:
        #     self._num_frames_per_block = self._block_len/self._frame_len
        # self._deinterleaver = Deinterleaver(mode['interleaver'][1], mode['interleaver'][2])
        # self._depuncturer   = common.Depuncturer(repeat=mode['repeat'])
        # self._viterbi_decoder = viterbi27(0x6d, 0x4f)
        # self._mode_description = 'MIL_STD_188-110A: (%d,%d) %dbps intl=%s [U=%d,K=%d]' % (data[9],data[10],
        #                                                                                   mode['bit_rate'],
        #                                                                                   mode['interleaver'][0],
        #                                                                                   mode['unknown'], mode['known'])
        # print(self._d1d2, mode, self._frame_len, self._mode_description)

    def set_mode(self, _):
        pass

    def get_mode(self):
        return self._mode_description

    def decode_soft_dec(self, soft_dec):
        print('decode_soft_dec', len(soft_dec), soft_dec.dtype)
        if self._mode['known'] == 0: ## orthogonal WALSH modulation
            n = len(soft_dec) // 32
            soft_bits = np.zeros(2*n, dtype=np.float32)
            for i in range(n):
                w = np.sum(soft_dec[32*i:32*(i+1)].reshape(4,8),0)
                b = FROM_WALSH4[np.packbits(w[0:4]>0)[0]]
                print('WALSH', i, w, b)
                abs_soft_dec = np.mean(np.abs(w))
                soft_bits[2*i]   = abs_soft_dec*(2*(b>>1)-1)
                soft_bits[2*i+1] = abs_soft_dec*(2*(b &1)-1)
            print('WALSH soft_bits=', soft_bits)
            r = self._deinterleaver.load(soft_bits)
        else:
            r = self._deinterleaver.load(soft_dec)
        print('decode_soft_dec r=', r.shape)
        if r.shape[0] == 0:
            return [],0.0
        ##print('deinterleaved bits: ', [x for x in 1*(r>0)])
        rd = self._depuncturer.process(r)
        self._viterbi_decoder.reset()
        decoded_bits = self._viterbi_decoder.udpate(rd)
        ##print('bits=', decoded_bits)
        quality = 100.0*self._viterbi_decoder.quality()/(2*len(decoded_bits))
        print('quality={}%'.format(quality))
        return decoded_bits,quality

    @staticmethod
    def get_preamble():
        """preamble symbols + scrambler"""
        return common.make_scr(PREAMBLE,PREAMBLE)
    def get_preamble_z(self):
        """preamble symbols for preamble correlation"""
        a = PhysicalLayer.get_preamble()
        return 1,np.array([z for z in a['symb']
                           for _ in range(self._sps)])

if __name__ == '__main__':
    print(PREAMBLE)
