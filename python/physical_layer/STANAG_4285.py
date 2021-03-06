## -*- python -*-

import numpy as np

from . import common
from digitalhf.digitalhf_swig import viterbi27

class Deinterleaver(object):
    "S4285 deinterleaver"
    def __init__(self, incr):
        self._dtype = np.float32
        ## incr = 12 -> L
        ## incr =  1 -> S
        self._buf = [np.zeros(incr*(31-i) + 1, dtype=self._dtype)
                     for i in range(32)]

    def push(self, a):
        assert(len(a) == 32)
        for i in range(32):
            self._buf[i][0] = a[i]
            self._buf[i] = np.roll(self._buf[i],1)
        return self.fetch()

    def fetch(self):
        return np.array([self._buf[(9*i)%32][0] for i in range(32)],
                        dtype=self._dtype)


## ---- constellatios -----------------------------------------------------------
BPSK=np.array(list(zip(np.exp(2j*np.pi*np.arange(2)/2), [0,1])), common.CONST_DTYPE)
QPSK=np.array(list(zip(np.exp(2j*np.pi*np.arange(4)/4), [0,1,3,2])), common.CONST_DTYPE)
PSK8=np.array(list(zip(np.exp(2j*np.pi*np.arange(8)/8), [0,1,3,2,6,7,5,4])), common.CONST_DTYPE)

## ---- constellation indices ---------------------------------------------------
MODE_BPSK=0
MODE_QPSK=1
MODE_8PSK=2

MODES = { ## [BPS]['const'] [BPS]['punct'] [BPS]['repeat']
    '2400': {'const': MODE_8PSK, 'repeat': 1, 'deintl_multiple': 4},
    '1200': {'const': MODE_QPSK, 'repeat': 1, 'deintl_multiple': 2},
     '600': {'const': MODE_BPSK, 'repeat': 1, 'deintl_multiple': 1},
     '300': {'const': MODE_BPSK, 'repeat': 2, 'deintl_multiple': 1},
     '150': {'const': MODE_BPSK, 'repeat': 4, 'deintl_multiple': 1},
      '75': {'const': MODE_BPSK, 'repeat': 8, 'deintl_multiple': 1}
}

DEINTERLEAVER_INCR = { 'S': 1, 'L': 12 }

class PhysicalLayer(object):
    """Physical layer description for STANAG 4285"""

    def __init__(self, sps):
        """intialization"""
        self._sps     = sps
        self._frame_counter = 0
        self._is_first_frame = True
        self._constellations = [BPSK,QPSK,PSK8]
        self._preamble = self.get_preamble()
        self._data     = self.get_data()
        self._viterbi_decoder = viterbi27(0x6d, 0x4f)
        self._mode_description = None
        self._frame_counter = -1
        self._fault_counter = 0

    def set_mode(self, mode):
        """set modulation and interleaver: 'BPS/S' or 'BPS/L'"""
        self._mode_description = mode
        bps,intl = mode.split('/')
        self._mode          = MODES[bps]['const']
        self._deinterleaver = Deinterleaver(DEINTERLEAVER_INCR[intl] * MODES[bps]['deintl_multiple'])
        self._depuncturer   = common.Depuncturer(repeat = MODES[bps]['repeat'])
        self._repeat        = MODES[bps]['repeat']
        self._fault_counter = 0

    def get_mode(self):
        return self._mode_description

    def get_constellations(self):
        return self._constellations

    def get_next_frame(self, symbols):
        """returns a tuple describing the frame:
        [0] ... known+unknown symbols and scrambling
        [1] ... modulation type after descrambling
        [2] ... a boolean indicating if the processing should continue
        [3] ... a boolean indicating if the soft decision for the unknown symbols are saved"""
        if len(symbols) == 0: ## 1st preamble
            self._frame_counter = 0

        success,frame_description = True,[]
        if (self._frame_counter%2) == 0: ## current frame is a data frame
            frame_description = [self._preamble,MODE_BPSK,success,False]
        else: ## current frame is a preamble frame
            idx = range(30,80)
            idx = range(50)
            z = symbols[idx]*np.conj(self._preamble['symb'][idx])
            mean_z = np.mean(z)
            if np.sum(np.real(z)<0) < 30 and np.real(mean_z) > np.abs(np.imag(mean_z)) and np.real(mean_z) > 0.3:
                self._fault_counter -= 1
            else:
                self._fault_counter += 1
            self._fault_counter = min(11, max(0, self._fault_counter))
            success = self._fault_counter < 10
            if not success:
                self._frame_counter = -2
                self._fault_counter = 0
            frame_description = [self._data,self._mode,success,True]

        self._frame_counter += 1
        return frame_description

    def get_doppler(self, iq_samples):
        r = {'success': False, ## -- quality flag
             'use_amp_est': self._frame_counter < 0,
             'doppler': 0}     ## -- doppler estimate (rad/symb)
        if len(iq_samples) == 0:
            return r

        sps  = self._sps
        zp   = np.array([x for x in self._preamble['symb'][9:40]
                         for _ in range(sps)], dtype=np.complex64)
        cc   = np.correlate(iq_samples, zp)
        imax = np.argmax(np.abs(cc[0:18*sps]))
        pks  = cc[(imax,imax+31*sps),]
        tpks = cc[imax+15*sps:imax+16*sps]

        r['success'] = bool(np.mean(np.abs(pks)) > 5*np.mean(np.abs(tpks)))
        r['doppler'] = np.diff(np.unwrap(np.angle(pks)))[0]/31/self._sps if r['success'] else 0
        return r

    def get_preamble_z(self):
        """preamble symbols for preamble correlation"""
        a = PhysicalLayer.get_preamble()
        return 2,np.array([z for z in a['symb'][0:31] for _ in range(self._sps)])

    def decode_soft_dec(self, soft_dec):
        n = len(soft_dec)
        quality_correction = 4.0/3.5 if n==384 else 1.0
        deintl = lambda x: np.concatenate([self._deinterleaver.push(x[i:i+32]) for i in range(0,len(x),32)])
        rd = self._derepeat(deintl(self._depuncture(soft_dec)))
        decoded_bits = self._viterbi_decoder.udpate(rd.tolist())
        quality      = 100.0*self._viterbi_decoder.quality()/(2*len(decoded_bits))*quality_correction
        return decoded_bits,quality

    def _derepeat(self, soft_dec):
        if self._repeat == 1:
            return soft_dec
        n = len(soft_dec)
        m = n//(2*self._repeat)
        u = soft_dec.reshape(m, 2*self._repeat)
        for i in range(1,self._repeat):
            u[:,0] += u[:,2*i]
            u[:,1] += u[:,2*i+1]
        return np.reshape(u[:,0:2], 2*m)

    def _depuncture(self, soft_dec):
        if len(soft_dec) != 384:
            return soft_dec
        else:
            u = np.zeros(512, dtype=soft_dec.dtype)
            u[0::4] = soft_dec[0::3]
            u[1::4] = soft_dec[1::3]
            u[2::4] = soft_dec[2::3]
            u[3::4] = 0.0 ## puncture
            return u

    @staticmethod
    def get_preamble():
        """preamble symbols + scrambler(=1)"""
        state = np.array([1,1,0,1,0], dtype=np.bool)
        taps  = np.array([0,0,1,0,1], dtype=np.bool)
        p = np.zeros(80, dtype=np.uint8)
        for i in range(80):
            p[i]  = state[-1]
            state = np.concatenate(([np.sum(state&taps)&1], state[0:-1]))
        a = np.zeros(80, common.SYMB_SCRAMBLE_DTYPE)
        ## BPSK modulation
        constellation = PhysicalLayer.make_psk(2,range(2))['points']
        a['symb']     = constellation[p,]
        a['scramble'] = 1
        return a

    @staticmethod
    def get_data():
        """data symbols + scrambler; for unknown symbols 'symb'=0"""
        state = np.array([1,1,1,1,1,1,1,1,1], dtype=np.bool)
        taps  = np.array([0,0,0,0,1,0,0,0,1], dtype=np.bool)
        p = np.zeros(176, dtype=np.uint8)
        for i in range(176):
            p[i] = np.sum(state[-3:]*[4,2,1])
            for _ in range(3):
                state = np.concatenate(([np.sum(state&taps)&1], state[0:-1]))
        a = np.zeros(176, common.SYMB_SCRAMBLE_DTYPE)
        ## 8PSK modulation
        constellation = PhysicalLayer.make_psk(8,range(8))['points']
        a['scramble'] = constellation[p,]
        known_symbols = np.mod(range(176),48)>=32
        a['symb'][known_symbols] = a['scramble'][known_symbols]
        return a

    @staticmethod
    def make_psk(n, gray_code):
        """generates n-PSK constellation data"""
        c = np.zeros(n, common.CONST_DTYPE)
        c['points']  = np.exp(2*np.pi*1j*np.arange(n)/n)
        c['symbols'] = gray_code
        return c
