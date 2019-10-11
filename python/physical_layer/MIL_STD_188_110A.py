## -*- python -*-

from __future__ import print_function
import numpy as np
import common
from digitalhf.digitalhf_swig import viterbi27

## ---- Walsh-8 codes -----------------------------------------------------------
WALSH8 = np.array([[0,0,0,0, 0,0,0,0],  # 0 - 000
                   [0,1,0,1, 0,1,0,1],  # 1 - 001
                   [0,0,1,1, 0,0,1,1],  # 2 - 010
                   [0,1,1,0, 0,1,1,0],  # 3 - 011
                   [0,0,0,0, 1,1,1,1],  # 4 - 100
                   [0,1,0,1, 1,0,1,0],  # 5 - 010
                   [0,0,1,1, 1,1,0,0],  # 6 - 011
                   [0,1,1,0, 1,0,0,1]], # 7 - 111
                  dtype=np.uint8)

FROM_WALSH8 = -np.ones(256, dtype=np.int8)
for i in range(8):
    FROM_WALSH8[np.packbits(WALSH8[i][:])[0]] = i

## ---- Walsh-4 codes -----------------------------------------------------------
WALSH4 = np.array([[0,0,0,0],  # 0 - 00
                   [0,1,0,1],  # 1 - 01
                   [0,1,1,0],  # 3 - 11 modified gray coding!
                   [0,0,1,1]], # 2 - 10 modified gray coding!
                  dtype=np.uint8)

FROM_WALSH4 = -np.ones(256, dtype=np.int8)
for i in range(4):
    FROM_WALSH4[np.packbits(WALSH4[i][:])[0]] = i

## ---- tri-bit codes -----------------------------------------------------------
TRIBIT = np.zeros((8,32), dtype=np.uint8)
for i in range(8):
    TRIBIT[i][:] = np.concatenate([WALSH8[i][:] for j in range(4)])

## ---- tri-bit scramble sequence for preamble ----------------------------------
TRIBIT_SCRAMBLE = np.array(
    [7,4,3,0,5,1,5,0,2,2,1,1,5,7,4,3,5,0,2,6,2,1,6,2,0,0,5,0,5,2,6,6],
    dtype=np.uint8)

## ---- preamble symbols ---------------------------------------------------------
D1=D2=C1=C2=C3=0 ## not known
PRE_SYMBOLS  = common.n_psk(2, np.concatenate(
    [TRIBIT[i][:] for i in [0,1,3,0,1,3,1,2,0,D1,D2,C1,C2,C3,0]]))
PRE_SYMBOLS[9*32:14*32] = 0

## ---- preamble scramble symbols ------------------------------------------------
PRE_SCRAMBLE = common.n_psk(8, np.concatenate([TRIBIT_SCRAMBLE for _ in range(15)]))

## ---- data scrambler -----------------------------------------------------------
class ScrambleData(object):
    """data scrambling sequence generator"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._state   = 0xBAD
        self._counter = 0

    def next(self):
        if self._counter == 160:
            self.reset()
        for _ in range(8):
            self._advance()
        self._counter += 1
        return self._state&7

    def _advance(self):
        msb = self._state>>11
        self._state = (self._state<<1)&4095
        if msb:
            self._state ^= 0x053
        return self._state

## ---- constellatios -----------------------------------------------------------
BPSK=np.array(zip(np.exp(2j*np.pi*np.arange(2)/2), [0,1]), common.CONST_DTYPE)
QPSK=np.array(zip(np.exp(2j*np.pi*np.arange(4)/4), [0,1,3,2]), common.CONST_DTYPE)
PSK8=np.array(zip(np.exp(2j*np.pi*np.arange(8)/8), [0,1,3,2,6,7,5,4]), common.CONST_DTYPE)

## ---- constellation indices ---------------------------------------------------
MODE_BPSK=0
MODE_QPSK=1
MODE_8PSK=2

## ---- mode definitions --------------------------------------------------------
MODE = [[{} for _ in range(8)] for _  in range(8)]
MODE[7][6] = {'bit_rate':4800, 'ci':MODE_8PSK, 'interleaver':['N',  1,  1], 'unknown':32,'known':16, 'nsymb': 1, 'coding_rate': 'n/a', 'repeat': 1}
MODE[5][6] = {} # reserved

MODE[7][7] = {'bit_rate':2400, 'ci':MODE_8PSK, 'interleaver':['S', 40, 72], 'unknown':32,'known':16, 'nsymb': 1, 'coding_rate': '1/2', 'repeat': 1}

#(5,7) is reserved
MODE[5][7] = {'bit_rate': 600, 'ci':MODE_BPSK, 'interleaver':['S', 40, 18], 'unknown':20,'known':20, 'nsymb': 1, 'coding_rate': '1/2', 'repeat': 1}

MODE[6][4] = {'bit_rate':2400, 'ci':MODE_8PSK, 'interleaver':['S', 40, 72], 'unknown':32,'known':16, 'nsymb': 1, 'coding_rate': '1/2', 'repeat': 1}
MODE[4][4] = {'bit_rate':2400, 'ci':MODE_8PSK, 'interleaver':['L', 40,576], 'unknown':32,'known':16, 'nsymb': 1, 'coding_rate': '1/2', 'repeat': 1}

MODE[6][5] = {'bit_rate':1200, 'ci':MODE_QPSK, 'interleaver':['S', 40, 36], 'unknown':20,'known':20, 'nsymb': 1, 'coding_rate': '1/2', 'repeat': 1}
MODE[4][5] = {'bit_rate':1200, 'ci':MODE_QPSK, 'interleaver':['L', 40,288], 'unknown':20,'known':20, 'nsymb': 1, 'coding_rate': '1/2', 'repeat': 1}

MODE[6][6] = {'bit_rate': 600, 'ci':MODE_BPSK, 'interleaver':['S', 40, 18], 'unknown':20,'known':20, 'nsymb': 1, 'coding_rate': '1/2', 'repeat': 1}
MODE[4][6] = {'bit_rate': 600, 'ci':MODE_BPSK, 'interleaver':['L', 40,144], 'unknown':20,'known':20, 'nsymb': 1, 'coding_rate': '1/2', 'repeat': 1}

MODE[6][7] = {'bit_rate': 300, 'ci':MODE_BPSK, 'interleaver':['S', 40, 18], 'unknown':20,'known':20, 'nsymb': 1, 'coding_rate': '1/4', 'repeat': 2}
MODE[4][7] = {'bit_rate': 300, 'ci':MODE_BPSK, 'interleaver':['L', 40,144], 'unknown':20,'known':20, 'nsymb': 1, 'coding_rate': '1/4', 'repeat': 2}

MODE[7][4] = {'bit_rate': 150, 'ci':MODE_BPSK, 'interleaver':['S', 40, 18], 'unknown':20,'known':20, 'nsymb': 1, 'coding_rate': '1/8', 'repeat': 4}
MODE[5][4] = {'bit_rate': 150, 'ci':MODE_BPSK, 'interleaver':['L', 40,144], 'unknown':20,'known':20, 'nsymb': 1, 'coding_rate': '1/8', 'repeat': 4}
## 75 bps othogonal WALSH modulation
MODE[7][5] = {'bit_rate':  75, 'ci':MODE_BPSK, 'interleaver':['S', 10,  9], 'unknown':160,'known': 0, 'nsymb':32, 'coding_rate': '1/2', 'repeat': 1}
MODE[5][5] = {'bit_rate':  75, 'ci':MODE_BPSK, 'interleaver':['L', 20, 36], 'unknown':160,'known': 0, 'nsymb':32, 'coding_rate': '1/2', 'repeat': 1}

## ---- deinterleaver -----------------------------------------------------------

class Deinterleaver(object):
    """deinterleave"""
    def __init__(self, rows, cols):
        self._a = np.zeros((rows, cols), dtype=np.float32)
        self._i = 0
        self._j = 0
        self._di =   9 if rows==40 else  7
        self._dj = -17 if rows==40 else -7
        self._buffer = np.zeros(0, dtype=np.float32)
        print('deinterleaver: ', rows, cols, self._di, self._dj)

    def fetch(self, a):
        pass

    def load(self, a):
        self._buffer = np.append(self._buffer, a)
        print('interleaver load', self._a.shape, a.shape, self._buffer.shape)
        if self._buffer.shape[0] < self._a.shape[0]:
            return np.zeros(0, dtype=np.float32)
        print('interleaver load buffer:', len(self._buffer),self._i,self._j)
        i = np.arange(self._a.shape[0])
        j = (self._j + self._dj*np.arange(self._a.shape[0])) % self._a.shape[1]
        self._a[i,j] = self._buffer[0:self._a.shape[0]]
        self._buffer = np.delete(self._buffer, i)
        self._j += 1
        print('interleaver load buffer:', len(self._buffer),self._i,self._j)
        if self._j == self._a.shape[1]:
            self._j = 0
            print('==================== interleaver is full! ====================')
            return np.concatenate([self._a[(self._di*i)%self._a.shape[0],j] for j in range(self._a.shape[1])])
        else:
            return np.zeros(0, dtype=np.float32)

## ---- physcal layer class -----------------------------------------------------
class PhysicalLayer(object):
    """Physical layer description for MIL-STD-188-110 Appendix A"""

    def __init__(self, sps):
        """intialization"""
        self._sps     = sps
        self._frame_counter = -1
        self._constellations = [BPSK, QPSK, PSK8]
        self._preamble    = self.get_preamble()
        self._pre_counter = -1
        self._d1d2        = [-1,-1] ## D1,D2
        self._mode        = {}
        self._scr_data    = ScrambleData()
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
              self._pre_counter, self._frame_counter)
        success = True
        if self._frame_counter == -1: ## preamble mode
            if len(symbols) == 0:
                return [self._preamble,MODE_BPSK,success,False]
            else:
                success = self.decode_preamble(symbols)
                if self._pre_counter != 0:
                    return [self._preamble,MODE_BPSK,success,False]
                else:
                    self._frame_counter = 0
                    self._scr_data.reset()
                    return [self.get_next_data_frame(success),self._mode['ci'],success,success]
        else: ## data mode
            self._frame_counter += 1
            ##print('test:', symbols[self._mode['unknown']:], np.mean(np.real(symbols[self._mode['unknown']:])))
            if self._mode['known'] == 0: ## orthogonal WALSH modulation
                success = True
                for i in range(5):
                    a = symbols[32*i:32*(i+1)]
                    success &= np.max(np.imag(np.mean(a.reshape(8,4),0))) < 0.25
            elif self._frame_counter < self._num_frames_per_block-2:
                success = np.mean(np.real(symbols[self._mode['unknown']:])) > 0.4 or np.max(np.imag(symbols[self._mode['unknown']:])) < 0.6
            if not success:
                print('aborting: ', symbols[self._mode['unknown']:])# np.mean(np.real(symbols[self._mode['unknown']:])),
                #np.max(np.imag(symbols[self._mode['unknown']:])))
            return [self.get_next_data_frame(success),self._mode['ci'],success,success]

    def get_next_data_frame(self, success):
        if self._frame_counter == self._num_frames_per_block:
            self._frame_counter = 0
        scramble_for_frame = common.n_psk(8, np.array([self._scr_data.next()
                                                       for _ in range(self._frame_len)]))
        a = common.make_scr(scramble_for_frame, scramble_for_frame)
        n_unknown = self._mode['unknown']
        a['symb'][0:n_unknown] = 0
        if self._mode['known'] != 0 and self._frame_counter >= self._num_frames_per_block-2:
            idx_d1d2 = self._frame_counter - self._num_frames_per_block + 2;
            a['symb'][n_unknown  :n_unknown+ 8] *= common.n_psk(2, WALSH8[self._d1d2[idx_d1d2]][:])
            a['symb'][n_unknown+8:n_unknown+16] *= common.n_psk(2, WALSH8[self._d1d2[idx_d1d2]][:])
        if not success:
            self._frame_counter = -1
            self._pre_counter = -1
        return a

    def get_doppler(self, iq_samples):
        """quality check and doppler estimation for preamble"""
        r = {'success':     False, ## -- quality flag
             'use_amp_est': self._frame_counter < 0,
             'doppler':     0}     ## -- doppler estimate (rad/symb)
        if len(iq_samples) != 0:
            sps  = self._sps
            zp   = np.array([z for z in PhysicalLayer.get_preamble()['symb']
                             for _ in range(sps)], dtype=np.complex64)
            ## find starting point
            _,_zp = self.get_preamble_z()
            cc   = np.correlate(iq_samples, zp[0:3*32*sps])
            imax = np.argmax(np.abs(cc[0:2*32*sps]))
            print('imax=', imax, len(iq_samples), len(cc))
            apks = np.abs(cc[(imax, imax+3*32*sps),])
            tpks = np.abs(cc[imax+3*16*sps:imax+5*16*sps])
            print('imax=', imax, 'apks=',apks,
                  np.mean(apks), np.mean(tpks))
            r['success'] = np.bool(np.mean(apks) > 5*np.mean(tpks) and apks[0]/apks[1] > 0.5 and apks[0]/apks[1] < 2.0)
            if r['success']:
                idx = np.arange(32*sps)
                pks = [np.correlate(iq_samples[imax+i*32*sps+idx],
                                    zp[             i*32*sps+idx])[0]
                       for i in range(9)]
                r['doppler'] = common.freq_est(pks)/(32*sps)
                print('success=', r['success'], 'doppler=', r['doppler'],
                      np.abs(np.array(pks)),
                      np.angle(np.array(pks)))
        return r

    def decode_preamble(self, symbols):
        data = [FROM_WALSH8[np.packbits
                            (np.real
                             (np.sum
                              (symbols[i:i+32].reshape((4,8)),0))<0)[0]]
                for i in range(0,15*32,32)]
        print('data=',data)
        self._pre_counter = sum([(x&3)*(1<<2*y) for (x,y) in zip(data[11:14][::-1], range(3))])
        self._d1d2 = data[9:11]
        print('MODE:', data[9:11])
        self._mode = mode = MODE[data[9]][data[10]]
        self._block_len = 11520 if mode['interleaver'][0] == 'L' else 1440
        self._frame_len = mode['known'] + mode['unknown']
        if mode['known'] == 0: ## orthogonal WALSH modulation
            self._num_frames_per_block = mode['interleaver'][1]*mode['interleaver'][2]/2*32/160
        else:
            self._num_frames_per_block = self._block_len/self._frame_len
        self._deinterleaver = Deinterleaver(mode['interleaver'][1], mode['interleaver'][2])
        self._depuncturer   = common.Depuncturer(repeat=mode['repeat'])
        self._viterbi_decoder = viterbi27(0x6d, 0x4f)
        self._mode_description = 'MIL_STD_188-110A: (%d,%d) %dbps intl=%s [U=%d,K=%d]' % (data[9],data[10],
                                                                                          mode['bit_rate'],
                                                                                          mode['interleaver'][0],
                                                                                          mode['unknown'], mode['known'])
        print(self._d1d2, mode, self._frame_len, self._mode_description)
        return True

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
        return common.make_scr(PRE_SCRAMBLE*PRE_SYMBOLS,
                               PRE_SCRAMBLE)
    def get_preamble_z(self):
        """preamble symbols for preamble correlation"""
        a = PhysicalLayer.get_preamble()
        return 0,np.array([z for z in a['symb'][0:3*32]
                           for _ in range(self._sps)])

if __name__ == '__main__':
    def gen_data_scramble():
        def advance(s):
            msb = s>>11
            s = (s<<1)&((1<<12)-1)
            if msb: s ^= 0x053
            return s
        a = np.zeros(160, dtype=np.uint8)
        s = 0xBAD
        for i in range(160):
            for _ in range(8): s = advance(s)
            a[i] = s&7;
        return a

    sps = 5;
    p=PhysicalLayer(sps)
    z1=np.array([x for x in PRE_SYMBOLS  for _ in range(sps)])
    z2=np.array([x for x in PRE_SCRAMBLE for _ in range(sps)])
    z=z1*z2;
    _,_z=p.get_preamble_z()
    print(all(z[0:3*32*sps]==_z[0:3*32*sps]))
    for i in range(3):
        print(i, all(z[32*sps*i:32*sps*(i+1)] == z[32*sps*(3+i):32*sps*(3+i+1)]))

    #print(np.sum(np.sum(z[0:32*5] * np.conj(z[32*5*3:32*5*4]))))
    #print(WALSH8[1][:])
    #print(sum(WALSH8[1][:]*(1<<np.array(range(7,-1,-1)))))
    #print(FROM_WALSH8)
    #print(gen_data_scramble())

    s=ScrambleData()
    #print([s.next() for _ in range(160)])
    #print([s.next() for _ in range(160)])
    #print(np.round(np.angle(PRE_SYMBOLS*PRE_SCRAMBLE)/np.pi*4))
