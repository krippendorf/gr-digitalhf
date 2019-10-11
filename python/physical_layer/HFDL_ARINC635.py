## -*- python -*-

from __future__ import print_function
import numpy as np
import common
from digitalhf.digitalhf_swig import viterbi27

## ---- constellatios -----------------------------------------------------------
BPSK=np.array(zip(np.exp(2j*np.pi*np.arange(2)/2), [0,1]), common.CONST_DTYPE)
QPSK=np.array(zip(np.exp(2j*np.pi*np.arange(4)/4), [0,1,3,2]), common.CONST_DTYPE)
PSK8=np.array(zip(np.exp(2j*np.pi*np.arange(8)/8), [0,1,3,2,6,7,5,4]), common.CONST_DTYPE)

## ---- constellation indices ---------------------------------------------------
MODE_BPSK=0
MODE_QPSK=1
MODE_8PSK=2

class LFSR(object):
    def __init__(self, init, taps):
        self._init  = np.array(init, dtype=np.bool)
        self._state = np.array(init, dtype=np.bool)
        self._taps  = np.array(taps, dtype=np.bool)

    def reset(self):
        self._state = self._init

    def next(self):
        self._state = np.concatenate([[np.sum(self._state&self._taps)&1], self._state[0:-1]])
        return self._state

LFSR_PREAMBLE = LFSR([1,1,1,1,1,1,1],
                     [1,0,0,1,0,1,1])
LFSR_M1       = LFSR([1,1,1,1,1,1,1],
                     [1,1,1,0,0,0,1])
LFSR_SCRAMBLE = LFSR([1,1,0,1,0,0,1,0,1,0,1,1,0,0,1],
                     [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
PREAMBLE = common.n_psk(2, np.array([LFSR_PREAMBLE.next()[0] for _ in range(127)]))
M1       = common.n_psk(2, np.array([LFSR_M1.next()[0]       for _ in range(127)]))
SCRAMBLE = common.n_psk(2, np.array([LFSR_SCRAMBLE.next()[0] for _ in range(120)]))
PROBE    = common.n_psk(2, np.array([0,0,0,1,0,0,1,1,0,1,0,1,1,1,1]))
T        = np.tile(PROBE, 9)

SHIFTS = [72,82,113,123,61,103,93,9]

MODES = [
    {'bps':  300, 'intl': {'type': 'S', 'cols':  54}, 'shift':  72, 'mode': MODE_BPSK, 'repeat': 2},
    {'bps':  600, 'intl': {'type': 'S', 'cols':  54}, 'shift':  82, 'mode': MODE_BPSK, 'repeat': 1},
    {'bps': 1200, 'intl': {'type': 'S', 'cols': 108}, 'shift': 113, 'mode': MODE_QPSK, 'repeat': 1},
    {'bps': 1800, 'intl': {'type': 'S', 'cols': 162}, 'shift': 123, 'mode': MODE_8PSK, 'repeat': 1},

    {'bps':  300, 'intl': {'type': 'L', 'cols': 126}, 'shift':  61, 'mode': MODE_BPSK, 'repeat': 2},
    {'bps':  600, 'intl': {'type': 'L', 'cols': 126}, 'shift': 103, 'mode': MODE_BPSK, 'repeat': 1},
    {'bps': 1200, 'intl': {'type': 'L', 'cols': 252}, 'shift':  93, 'mode': MODE_QPSK, 'repeat': 1},
    {'bps': 1800, 'intl': {'type': 'L', 'cols': 378}, 'shift':   9, 'mode': MODE_8PSK, 'repeat': 1}
]

## interleaver -> number of 45-symbol frames
NUM_FRAMES = { 'S':  72,
               'L': 168 }

## interleave -> interleaver increment
INTL_INCR  = { 'S': -17,
               'L': -23 }

class DeInterleaver(object):
    def __init__(self, ncols, di):
        self._matrix = np.zeros((40, ncols), dtype=np.float32)
        self._ncols = ncols
        self._dj    = di
        self._i = 0
        self._j = 0
        self._counter = 0

    def insert(self, v):
        for val in v:
            self._counter += 1
            self._matrix[self._i][self._j] = val
            self._i += 1
            self._j  = np.mod(self._j + self._dj, self._ncols)
            if self._i == 40:
                self._i  = 0
                self._j += 1
        print('insert: ', self._i, self._j, self._counter, 40*self._ncols)
        return self._counter == 40*self._ncols

    def fetch(self):
        r = np.zeros(40*self._ncols)
        idx = np.mod(9*np.arange(40, dtype=np.int), 40)
        for j in range(self._ncols):
            r[j*40:(j+1)*40] = self._matrix[idx,j]
        return r

## ---- physcal layer class -----------------------------------------------------
class PhysicalLayer(object):
    """Physical layer description for HFDL ARINC 635"""

    def __init__(self, sps):
        """intialization"""
        self._sps     = sps
        self._frame_counter = -1
        self._constellations = [BPSK, QPSK, PSK8]
        self._preamble    = self.get_preamble()
        self._pre_counter = -1
        self._mode        = {}
        self._viterbi_dec = viterbi27(0x6d, 0x4f)
        self._repeat      = 1
        self._mode_descr  = 'UNKNOWN'
        self._fault_counter = 0

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
        if len(symbols) == 0:
            self._frame_counter = -1
            self._fault_counter = 0
            s = self.get_preamble()
            s.resize(15+len(s))
            s['scramble'][-15:] = 1
            return [s,MODE_BPSK,success,False]

        if self._frame_counter == -1: ## preamble mode
            success,idx = self.decode_preamble(symbols)
            print('IDX= ', idx)
            if idx == 0: ## 2nd preamble frame
                s = self._preamble
                s = np.roll(s, -15)
                s['symb'][-15:] = 0
                s['scramble'][-15:] = 1
                return [s,MODE_BPSK,True,False]
            else:
                self._frame_counter = 0
                mode = MODES[idx-1]
                print('MODE=', mode)
                self._mode = mode['mode']
                self._mode_descr = "HFDL bps=%d intl=%s " % (mode['bps'], mode['intl']['type'])
                self._num_frames = NUM_FRAMES[mode['intl']['type']]
                self._deintl     = DeInterleaver(mode['intl']['cols'], INTL_INCR[mode['intl']['type']])
                self._repeat     = mode['repeat']
                self._a = self.make_data_frame(mode)
                s = np.concatenate([np.roll(M1, -(15+SHIFTS[idx-1])), T])
                a = common.make_scr(s,s)
                return [a,MODE_BPSK,True,False]

        if self._frame_counter >= 0: ## data
            print('====', self._frame_counter, self._num_frames)
            do_continue = use_soft_dec = self.get_data_frame_quality(symbols)
            if self._frame_counter == self._num_frames:
                self._frame_counter = 0
                do_continue = False
            else:
                self._frame_counter += len(self._a)/45;
            if not do_continue:
                self._frame_counter = -2
            print("SUCCESS ", do_continue, use_soft_dec)
            return [self._a, self._mode, do_continue, use_soft_dec]

    def make_data_frame(self, mode):
        s = np.zeros(180, dtype=common.SYMB_SCRAMBLE_DTYPE)
        s['scramble'][:] = 1
        for i in range(0,180,45):
            s['scramble'][i   :i+30] = SCRAMBLE[i*30/45:i*30/45+30]
            s['scramble'][i+30:i+45] = PROBE
            s['symb'    ][i+30:i+45] = PROBE
        return s

    def get_data_frame_quality(self, symbols):
        s = symbols[-15:]
        mean_s = np.mean(s)
        tests = [np.abs(mean_s) > 0.4,
                 np.real(mean_s) > np.imag(mean_s)]
        print('FRAME_QUALITY: ', s, mean_s, tests)
        if all(tests):
            self._fault_counter -= 1
        else:
            self._fault_counter += 1
        self._fault_counter = min(11, max(0, self._fault_counter))
        success = self._fault_counter < 10
        if not success:
            self._fault_counter = 0
        return success

    def get_doppler(self, iq_samples):
        """quality check and doppler estimation for preamble"""
        r = {'success': False, ## -- quality flag
             'use_amp_est': self._frame_counter < 0,
             'doppler': 0}     ## -- doppler estimate (rad/symb)
        if len(iq_samples) != 0:
            sps  = self._sps
            _,zp = self.get_preamble_z() ## length is sps*128
            cc   = np.correlate(iq_samples, zp[0:32*sps])
            imax = np.argmax(np.abs(cc[0:32*sps]))
            print('imax=', imax, len(iq_samples), len(cc))
            pks = np.zeros(4, dtype=np.complex64)
            for i in range(4):
                idx = 32*sps*i+np.arange(32*sps)
                idx = idx[idx<len(iq_samples)]
                pks[i] = np.vdot(zp[idx], iq_samples[idx])
            print('pks=', pks)
            r['success'] = True
            if r['success']:
                r['doppler'] = common.freq_est(pks)/(32*sps)
                print('success=', r['success'], 'doppler=', r['doppler'],
                      np.abs(np.array(pks)),
                      np.angle(np.array(pks)))
        return r

    def decode_preamble(self, symbols):
        st = symbols[-50:-15] ## should all be 1+0i
        print('st=', st)
        test = np.mean(np.real(st)) > 0.5 and np.max(np.imag(st)) < 0.5
        ## decide what is the next frame
        t = symbols[-15:]
        print('t=', t, np.angle(t))
        tt = np.zeros(1+len(MODES))
        tt[0] = np.abs(np.vdot(t, self._preamble['symb'][0:15]))
        for i in range(1,len(tt)):
            tt[i] = np.abs(np.vdot(t, np.roll(M1, -SHIFTS[i-1])[0:15]))
        imax = np.argmax(tt)
        test = tt[imax] / (np.sum(tt) - tt[imax]) * len(MODES)
        success = test > 3
        print('XXX ', test, tt)
        return success,imax

    def set_mode(self, _):
        pass

    def get_mode(self):
        return self._mode_descr

    def decode_soft_dec(self, soft_dec):
        is_full = self._deintl.insert(soft_dec)
        print('decode_soft_dec ', len(soft_dec), is_full, '******************************')
        if not is_full:
            return [],0.0
        r = self._deintl.fetch()
        rd = r if self._repeat == 1 else r[0::2]+r[1::2]
        self._viterbi_dec.reset()
        decoded_bits = self._viterbi_dec.udpate(rd)
        quality      = 100.0*self._viterbi_dec.quality()/(2*len(decoded_bits))
        print('qyality= ', quality, ' bits=', decoded_bits)
        if quality > 99.0:
            return np.packbits(decoded_bits),quality
        else:
            return [],quality

    @staticmethod
    def get_preamble():
        """preamble symbols + scrambler"""
        a = common.make_scr(PREAMBLE, PREAMBLE);
        #a['scramble'][:]=1
        return a
    def get_preamble_z(self):
        """preamble symbols for preamble correlation"""
        a = PhysicalLayer.get_preamble()
        ## add one more symbol to get a 128 symbol sequence
        z = np.array([z for z in a['symb'] for _ in range(self._sps)])
        return 3,np.concatenate([z, z[0:2*self._sps]])

if __name__ == '__main__':

    p0=np.array([0,1,0,1,1,0,1,1,1,0,1,1,1,1,0,0,0,1,1,1,0,1,0,0,0,1,0,1,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,1,1,0,0,1,1,0,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,1,0,1,0,1,0,0,1,1,0,1,1,0,1,0,0,1,0,1,0,0,0,0,1,0,1,1,0,0,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1])
    # p = make_preamble()
    # print(len(p), p)
    # print(len(p0), p0)
    # print(np.all(p==p0))

    m1='011 1011 0111 1010 0010 1100 1011 1110 0010 0000 0110 0110 1100 0111 0011 1010 1110 0001 0011 0000 0101 0101 1010 0111 1001 00001 1010 1000 0111 1111'
    m1='011101101111010001011001011111000100000011001101100011100111010111000010011000001010101101001111001'
    # m1=make_m1()
    # l= [72,82,113,123,61,103,93,9]
    # print(m1)
    # for i in l:
    #     print(np.roll(m1,-i)[:31])
    # scr='13' '1B' 'C4' '25' '0F' '8C' '15' 'EF' 'CD' '6A' 'EC' '99' '6E' '23' '68'
    # scr=make_scr()
    print(np.concatenate([PREAMBLE,[PREAMBLE[0]]]))
    ##print(M1)
    #print(scr)
#     300  72  61
#     600  82 103
#    1200 113  93
#    1800 123   9

