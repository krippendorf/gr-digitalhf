## -*- python -*-

from __future__ import print_function
import numpy as np
import common
from digitalhf.digitalhf_swig import viterbi27

## ---- constellations -----------------------------------------------------------
BPSK=np.array(zip(np.exp(2j*np.pi*np.arange(2)/2), [0,1]), common.CONST_DTYPE)
QPSK=np.array(zip(np.exp(2j*np.pi*np.arange(4)/4), [0,1,3,2]), common.CONST_DTYPE)
PSK8=np.array(zip(np.exp(2j*np.pi*np.arange(8)/8), [1,0,2,3,7,6,4,5]), common.CONST_DTYPE)
QAM16=np.array(
    zip([+0.866025+0.500000j,  0.500000+0.866025j,  1.000000+0.000000j,  0.258819+0.258819j,
         -0.500000+0.866025j,  0.000000+1.000000j, -0.866025+0.500000j, -0.258819+0.258819j,
         +0.500000-0.866025j,  0.000000-1.000000j,  0.866025-0.500000j,  0.258819-0.258819j,
         -0.866025-0.500000j, -0.500000-0.866025j, -1.000000+0.000000j, -0.258819-0.258819j],
        range(16)), common.CONST_DTYPE)
QAM32=np.array(
    zip([+0.866380+0.499386j,  0.984849+0.173415j,  0.499386+0.866380j,  0.173415+0.984849j,
         +0.520246+0.520246j,  0.520246+0.173415j,  0.173415+0.520246j,  0.173415+0.173415j,
         -0.866380+0.499386j, -0.984849+0.173415j, -0.499386+0.866380j, -0.173415+0.984849j,
         -0.520246+0.520246j, -0.520246+0.173415j, -0.173415+0.520246j, -0.173415+0.173415j,
         +0.866380-0.499386j,  0.984849-0.173415j,  0.499386-0.866380j,  0.173415-0.984849j,
         +0.520246-0.520246j,  0.520246-0.173415j,  0.173415-0.520246j,  0.173415-0.173415j,
         -0.866380-0.499386j, -0.984849-0.173415j, -0.499386-0.866380j, -0.173415-0.984849j,
         -0.520246-0.520246j, -0.520246-0.173415j, -0.173415-0.520246j, -0.173415-0.173415j],
        range(32)), common.CONST_DTYPE)
QAM64=np.array(
    zip([+1.000000+0.000000j,  0.822878+0.568218j,  0.821137+0.152996j,  0.932897+0.360142j,
         +0.000000-1.000000j,  0.822878-0.568218j,  0.821137-0.152996j,  0.932897-0.360142j,
         +0.568218+0.822878j,  0.588429+0.588429j,  0.588429+0.117686j,  0.588429+0.353057j,
         +0.568218-0.822878j,  0.588429-0.588429j,  0.588429-0.117686j,  0.588429-0.353057j,
         +0.152996+0.821137j,  0.117686+0.588429j,  0.117686+0.117686j,  0.117686+0.353057j,
         +0.152996-0.821137j,  0.117686-0.588429j,  0.117686-0.117686j,  0.117686-0.353057j,
         +0.360142+0.932897j,  0.353057+0.588429j,  0.353057+0.117686j,  0.353057+0.353057j,
         +0.360142-0.932897j,  0.353057-0.588429j,  0.353057-0.117686j,  0.353057-0.353057j,
         +0.000000+1.000000j, -0.822878+0.568218j, -0.821137+0.152996j, -0.932897+0.360142j,
         -1.000000+0.000000j, -0.822878-0.568218j, -0.821137-0.152996j, -0.932897-0.360142j,
         -0.568218+0.822878j, -0.588429+0.588429j, -0.588429+0.117686j, -0.588429+0.353057j,
         -0.568218-0.822878j, -0.588429-0.588429j, -0.588429-0.117686j, -0.588429-0.353057j,
         -0.152996+0.821137j, -0.117686+0.588429j, -0.117686+0.117686j, -0.117686+0.353057j,
         -0.152996-0.821137j, -0.117686-0.588429j, -0.117686-0.117686j, -0.117686-0.353057j,
         -0.360142+0.932897j, -0.353057+0.588429j, -0.353057+0.117686j, -0.353057+0.353057j,
         -0.360142-0.932897j, -0.353057-0.588429j, -0.353057-0.117686j, -0.353057-0.353057j],
        range(64)), common.CONST_DTYPE)

## for test
QAM64p = QAM64[(3,24,56,35,39,60,28,7),]
QAM64p['symbols'] = range(8) ## not used

## ---- Walsh-4 codes ----------------------------------------------------------
WALSH4 = np.array([[0,0,0,0],  # 0 - 00
                   [0,1,0,1],  # 1 - 01
                   [0,0,1,1],  # 2 - 10
                   [0,1,1,0]], # 3 - 11
                  dtype=np.uint8)
FROM_WALSH4 = -np.ones(256, dtype=np.int8)
for i in range(4):
    FROM_WALSH4[np.packbits(WALSH4[i][:])[0]] = i

## ---- constellation indices ---------------------------------------------------
MODE_BPSK   = 0
MODE_QPSK   = 1
MODE_8PSK   = 2
MODE_16QAM  = 3
MODE_32QAM  = 4
MODE_64QAM  = 5
MODE_64QAMp = 6

## ---- data scrambler -----------------------------------------------------------
class ScrambleData(object):
    """data scrambling sequence generator"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._state = np.array([0,0,0,0,0,0,0,0,1], dtype=np.bool)
        self._taps =  np.array([0,0,0,0,1,0,0,0,1], dtype=np.bool)

    def next(self, num_bits):
        r = np.packbits(self._state[1:])[0]&((1<<num_bits)-1)
        for _ in range(num_bits):
            self._advance()
        return r

    def _advance(self):
        self._state = np.concatenate(([self._state.dot(self._taps)&1],
                                      self._state[0:-1]))


class ScrambleDataP(object):
    """data scrambling sequence generator"""
    def __init__(self):
        self._i = 0
        state = np.array([0,0,0,0,0,0,0,0,0,0,0,1], dtype=np.uint8)
        taps =  np.array([1,1,0,0,1,0,1,0,0,0,0,0], dtype=np.uint8)
        n = 10000
        m = len(state)
        sequence = np.zeros(n, dtype=np.uint8)
        sequence[0:m] = state
        for i in range(m,n):
            sequence[i] = sequence[i-m:i].dot(taps)&1
        idx = np.arange(160, dtype=np.uint32)
        self._seq = 4*sequence[3530+idx] + 2*sequence[4042+idx] + sequence[4796+idx]

    def reset(self):
        self._i = 0

    def get_seq(self):
        return self._seq

    def next(self):
        if self._i == 160:
            self._i = 0
        s = self._seq[self._i]
        self._i += 1
        return s

## ---- preamble definitions  ---------------------------------------------------
## 184 = 8*23
PREAMBLE=common.n_psk(8, np.array(
    [1,5,1,3,6,1,3,1,1,6,3,7,7,3,5,4,3,6,6,4,5,4,0,
     2,2,2,6,0,7,5,7,4,0,7,5,7,1,6,1,0,5,2,2,6,2,3,
     6,0,0,5,1,4,2,2,2,3,4,0,6,2,7,4,3,3,7,2,0,2,6,
     4,4,1,7,6,2,0,6,2,3,6,7,4,3,6,1,3,7,4,6,5,7,2,
     0,1,1,1,4,4,0,0,5,7,7,4,7,3,5,4,1,6,5,6,6,4,6,
     3,4,3,0,7,1,3,4,7,0,1,4,3,3,3,5,1,1,1,4,6,1,0,
     6,0,1,3,1,4,1,7,7,6,3,0,0,7,2,7,2,0,2,6,1,1,1,
     2,7,7,5,3,3,6,0,5,3,3,1,0,7,1,1,0,3,0,4,0,7,3]))

BARKER_13 = [0,4,0,4,0,0,4,4,0,0,0,0,0]
MP_PLUS   = [0,0,0,0,0,2,4,6,0,4,0,4,0,6,4,2,0,0,0,0,0,2,4,6,0,4,0,4,0,6,4] ## length 31
MP_MINUS  = [4,4,4,4,4,6,0,2,4,0,4,0,4,2,0,6,4,4,4,4,4,6,0,2,4,0,4,0,4,2,0] ## length 31

## 103 = 31 + 1 + 3*13 + 1 + 31
REINSERTED_PREAMBLE = common.n_psk(8, np.array(MP_PLUS + [2,] + 3 * BARKER_13 + [6,] + MP_MINUS))
HFXL_PREAMBLE       = common.n_psk(8, np.array(7 * BARKER_13 + MP_PLUS))

## length 31 mini-probes
MINI_PROBE=[common.n_psk(8, np.array(MP_PLUS)),  ## sign = + (0)
            common.n_psk(8, np.array(MP_MINUS))] ## sign = - (1)

## ---- di-bits ----------------------------------------------------------------
TO_DIBIT=[(0,0),(0,1),(1,1),(1,0)]

## ---- rate -------------------------------------------------------------------
TO_RATE={(0,0,0): {'baud': '--------', 'bits_per_symbol': 0},  ## reserved
         (0,0,1): {'baud': '3200 bps', 'bits_per_symbol': 2, 'ci': MODE_QPSK},
         (0,1,0): {'baud': '4800 bps', 'bits_per_symbol': 3, 'ci': MODE_8PSK},
         (0,1,1): {'baud': '6400 bps', 'bits_per_symbol': 4, 'ci': MODE_16QAM},
         (1,0,0): {'baud': '8000 bps', 'bits_per_symbol': 5, 'ci': MODE_32QAM},
         (1,0,1): {'baud': '9600 bps', 'bits_per_symbol': 6, 'ci': MODE_64QAM},
         (1,1,0): {'baud':'12800 bps', 'bits_per_symbol': 6, 'ci': MODE_64QAM},
         (1,1,1): {'baud':     'HFXL', 'bits_per_symbol': 0}}  ## reserved - used by THALES HFXL

## ---- interleaver ------------------------------------------------------------
TO_INTERLEAVER={(0,0,0): {'frames': -1, 'id': '--', 'name': 'illegal'},
                (0,0,1): {'frames':  1, 'id': 'US', 'name': 'Ultra Short'},
                (0,1,0): {'frames':  3, 'id': 'VS', 'name': 'Very Short'},
                (0,1,1): {'frames':  9, 'id':  'S', 'name': 'Short'},
                (1,0,0): {'frames': 18, 'id':  'M', 'name': 'Medium'},
                (1,0,1): {'frames': 36, 'id':  'L', 'name': 'Long'},
                (1,1,0): {'frames': 72, 'id': 'VL', 'name': 'Very Long'},
                (1,1,1): {'frames': -1, 'id': '--', 'name': 'illegal'}}

MP_COUNTER=[(0,0,1), ## 1st
            (0,1,0), ## 2nd
            (0,1,1), ## 3rd
            (1,0,0)] ## 4th

## ---- interleaver size
INTL_SIZE = { ## 1 3 9 18 36 72
    '--------': {'US':    0, 'VS':    0, 'S':     0, 'M':     0, 'L':     0, 'VL':      0},
    '3200 bps': {'US':  512, 'VS': 1536, 'S':  4608, 'M':  9216, 'L': 18432, 'VL':  36864},
    '4800 bps': {'US':  768, 'VS': 2304, 'S':  6912, 'M': 13824, 'L': 27648, 'VL':  55296},
    '6400 bps': {'US': 1024, 'VS': 3072, 'S':  9216, 'M': 18432, 'L': 36864, 'VL':  73728},
    '8000 bps': {'US': 1280, 'VS': 3840, 'S': 11520, 'M': 23040, 'L': 46080, 'VL':  92160},
    '9600 bps': {'US': 1536, 'VS': 4608, 'S': 13824, 'M': 27648, 'L': 55296, 'VL': 110592}
}

## ---- interleaver increment
INTL_INCR = { ## 1 3 9 18 36 72
    '--------': {'US':   0, 'VS':   0, 'S':    0, 'M':    0, 'L':     0, 'VL':     0},
    '3200 bps': {'US':  97, 'VS': 229, 'S':  805, 'M': 1393, 'L':  3281, 'VL':  6985},
    '4800 bps': {'US': 145, 'VS': 361, 'S': 1045, 'M': 2089, 'L':  5137, 'VL': 10273},
    '6400 bps': {'US': 189, 'VS': 481, 'S': 1393, 'M': 3281, 'L':  6985, 'VL': 11141},
    '8000 bps': {'US': 201, 'VS': 601, 'S': 1741, 'M': 3481, 'L':  8561, 'VL': 14441},
    '9600 bps': {'US': 229, 'VS': 805, 'S': 2089, 'M': 5137, 'L': 10273, 'VL': 17329}
}

## ---- HFXL ----
TO_HFXL_MOD = {
    (0,0,0,0): MODE_BPSK,
    (0,0,0,1): MODE_BPSK,

    (1,0,0,0): MODE_QPSK,
    (1,0,0,1): MODE_QPSK,

    (0,1,1,0): MODE_QPSK,
    (0,1,1,1): MODE_QPSK,

    (0,1,0,0): MODE_8PSK,
    (0,1,0,1): MODE_8PSK,

    (0,0,1,0): MODE_32QAM,

    (1,1,0,0): MODE_16QAM,
    (1,1,0,1): MODE_16QAM
}

## ---- deinterleaver+depuncturer
class DeIntl_DePunct(object):
    """deinterleave"""
    def __init__(self, size, incr):
        self._size    = size
        self._i       = 0
        self._array   = np.zeros(size, dtype=np.float64)
        self._idx     = np.mod(incr*np.arange(size, dtype=np.int32), size)
        print('deinterleaver: ', size, incr, self._idx[0:100])

    def fetch(self, a):
        pass

    def load(self, a):
        n = len(a)
        i = self._i
        if i==0:
            self._array[:] = 0
        print('deinterleaver load buffer:', i,len(self._array),n)
        assert(i+n <= self._size)
        self._array[i:i+n] = a
        self._i += n
        result = np.zeros(0, dtype=np.float64)
        if self._i == self._size:
            print('deinterleaver: ', self._idx[0:100])
            print('==== TEST ====', self._array)
            #tmp = np.zeros(self._size, dtype=np.float32)
            tmp = self._array[self._idx]
            result = np.zeros(self._size*6//4, dtype=np.float64)
            assert(len(result[0::6]) == len(tmp[0::4]))
            assert(len(result[1::6]) == len(tmp[1::4]))
            assert(len(result[2::6]) == len(tmp[2::4]))
            assert(len(result[5::6]) == len(tmp[3::4]))
            result[0::6] = tmp[0::4]
            result[1::6] = tmp[1::4]
            result[2::6] = tmp[2::4]
            result[3::6] = 0
            result[4::6] = 0
            result[5::6] = tmp[3::4]
            print('==================== interleaver is full! ====================',
                  len(result[0::6]), len(tmp[0::4]), np.sum(result==0))
            self._i = 0
        return result


## ---- physcal layer class -----------------------------------------------------
class PhysicalLayer(object):
    """Physical layer description for MIL-STD-188-110 Appendix C = STANAG 4539"""

    def __init__(self, sps):
        """intialization"""
        self._sps = sps
        self._mode_name = '110C' # default is plain 110C, other supported mode names are '12800bpsBurst', 'HFXL'
        self._frame_counter = -2
        self._constellations = [BPSK, QPSK, PSK8, QAM16, QAM32, QAM64, QAM64p]
        self._preamble = self.get_preamble()
        self._scramble = ScrambleData()
        self._viterbi_decoder = viterbi27(0x6d, 0x4f)
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
        print('-------------------- get_frame --------------------', self._frame_counter)
        success = True
        if self._frame_counter == -2: ## ---- preamble
            self._deintl_depunct = None
            self._mode = {}
            self._preamble_offset = 0
            self._frame_counter += 1
            return [self._preamble,MODE_BPSK,success,False]

        if self._frame_counter == -1: ## --- re-inserted preamble
            self._frame_counter += 1
            success = self.get_preamble_quality(symbols) if self._frame_counter < 4 else self.get_data_frame_quality(symbols)
            return [self.make_reinserted_preamble(self._preamble_offset,success),MODE_QPSK,success,False]

        if self._frame_counter >= 0: ## ---- data frames
            success = False
            self._frame_counter += 1
            if self._frame_counter == 1:
                success = self.decode_reinserted_preamble(symbols)
            elif self._frame_counter == 2 and self.is_HFXL():
                success = self.decode_hfxl_preamble(symbols)
            else:
                success = self.get_data_frame_quality(symbols)
            if self.is_plain_110C() or self.is_12800bpsBurst() or self._frame_counter >= 2:
                return [self.make_data_frame(success),self._constellation_index,success,success]
            if self.is_HFXL() and self._frame_counter == 1:
                return [self.make_hfxl_preamble(success),MODE_QPSK,success,False]

    def get_doppler(self, iq_samples):
        """quality check and doppler estimation for preamble"""
        r = {'success': False, ## -- quality flag
             'doppler': 0}     ## -- doppler estimate (rad/symb)
        if len(iq_samples) != 0:
            sps  = self._sps
            m    = 23*sps
            idx  = np.arange(m)
            idx2  = np.arange(m+23*sps)
            _,zp = self.get_preamble_z()
            n    = len(zp)
            cc   = np.correlate(iq_samples, zp)
            imax = np.argmax(np.abs(cc[0:23*sps]))
            print('imax=', imax, len(iq_samples))
            pks  = np.array([np.correlate(iq_samples[imax+i*m+idx],
                                          zp[i*m+idx])[0]
                             for i in range(n//m)])
            val  = np.array([np.mean(np.abs(np.correlate(iq_samples[imax+i*m+idx2],
                                                         zp[i*m+idx])[11*sps+np.arange(-2*sps,2*sps)]))
                             for i in range((n//m)-1)])
            ## filter out
            (i,) = (np.abs(pks) > 0.5*np.mean(np.abs(pks[-3:]))).nonzero()
            i    = i[0] if i.size > 0 else 0
            pks = pks[i:]
            val = val[i:]
            if pks.size > 1:
                tests = np.abs(pks[:-1])/val
                r['success'] = bool(np.median(tests) > 2.0)
                print('test:', np.abs(pks), tests)
            if r['success']:
                print('doppler apks', np.abs(pks))
                print('doppler ppks', np.angle(pks),
                      np.diff(np.unwrap(np.angle(pks)))/m,
                      np.mean(np.diff(np.unwrap(np.angle(pks)))/m))
                r['doppler'] = common.freq_est(pks)/m
            print(r)
        return r

    def set_mode(self, mode):
        pass

    def get_mode(self):
        return self._mode_description

    def get_preamble_quality(self, symbols):
        print('get_preamble_quality', np.abs(np.mean(symbols[-32:])), symbols[-32:])
        return np.abs(np.mean(symbols[-32:])) > 0.5

    def get_data_frame_quality(self, symbols):
        print('get_data_frame_quality', np.mean(symbols[-31:]))
        return np.abs(np.mean(symbols[-31:])) > 0.5

    def is_plain_110C(self):
        return self._mode_name == '110C'
    def is_12800bpsBurst(self):
        return self._mode_name == '12800bpsBurst'
    def is_HFXL(self):
        return self._mode_name == 'HFXL'

    def decode_reinserted_preamble(self, symbols):
        ## decode D0,D1,D2
        success = True
        z = np.array([np.mean(symbols[-71+i*13:-71+(i+1)*13]) for i in range(3)])
        if np.mean(np.abs(z)) < 0.4:
            return False
        print('decode_reinserted_preamble',
              '\nHH', symbols[0:-71],
              '\nD0', symbols[-71   :-71+13],
              '\nD1', symbols[-71+13:-71+26],
              '\nD2', symbols[-71+26:-71+39],
              '\nTT', symbols[-71+4*13:], z)
        d0d1d2 = map(np.uint8, np.mod(np.round(np.angle(z)/np.pi*2),4))
        self._dibits = dibits = [TO_DIBIT[idx] for idx in d0d1d2]
        mode = {'rate':        tuple([x[0] for x in dibits]),
                'interleaver': tuple([x[1] for x in dibits])}
        if self._mode != {}:
            success = (mode == self._mode)
        if not success:
            return success
        self._mode = mode
        self._rate_info = rate_info = TO_RATE[self._mode['rate']]
        self._intl_info = intl_info = TO_INTERLEAVER[self._mode['interleaver']]

        self._mode_name = '110C'
        if mode['rate']==(1,1,0) and mode['interleaver']==(0,0,1):
            self._mode_name = '12800bpsBurst'
        if rate_info['baud'] == 'HFXL':
            self._mode_name = 'HFXL'

        self._mode_description = '%s rate=%s intl=%s' % (self._mode_name, rate_info['baud'], intl_info['id'])

        print('======== rate,interleaver:', rate_info, intl_info, self._mode_name)
        self._data_scramble_xor = np.zeros(256, dtype=np.uint8)
        self._data_scramble     = np.ones (256, dtype=np.complex64)
        if self.is_12800bpsBurst():
            self._scrp = ScrambleDataP()
            self._constellation_index = MODE_BPSK
        elif self.is_HFXL():
            self._scramble.reset()
            num_bits = 3
            iscr = np.array([self._scramble.next(num_bits) for _ in range(256)],
                            dtype=np.uint8)
            self._data_scramble[:] = common.n_psk(8, iscr)
            self._constellation_index = MODE_8PSK
            pass
        elif self.is_plain_110C():
            self._interleaver_frames = intl_info['frames']
            baud      = rate_info['baud']
            intl_id   = intl_info['id']
            intl_size = INTL_SIZE[baud][intl_id]
            intl_incr = INTL_INCR[baud][intl_id]
            if self._deintl_depunct == None:
                self._deintl_depunct = DeIntl_DePunct(size=intl_size,
                                                      incr=intl_incr)
            self._constellation_index = rate_info['ci']
            print('constellation index', self._constellation_index)
            self._scramble.reset()
            num_bits = max(3, rate_info['bits_per_symbol'])
            iscr = np.array([self._scramble.next(num_bits) for _ in range(256)],
                            dtype=np.uint8)
            print('iscr=', iscr)
            if rate_info['ci'] > MODE_8PSK:
                self._data_scramble_xor[:] = iscr
            else:
                self._data_scramble[:] = common.n_psk(8, iscr)
        else:
            ## TODO: generate an error message
            success = False
        return success

    def decode_hfxl_preamble(self, symbols):
        ## decode D0,D1,D2
        success = True
        z = np.mean(symbols[0:7*13].reshape(7,13),1)
        print('decode_hfxl_preamble: z=', z, np.mean(np.abs(z)))
        if np.mean(np.abs(z)) < 0.4:
            return False
        ds = map(np.uint8, np.mod(np.round(np.angle(z)/np.pi*2),4))
        self._dibits += [TO_DIBIT[idx] for idx in ds]
        l = tuple([x[1] for x in self._dibits[0:4]])
        try:
            self._constellation_index = TO_HFXL_MOD[l]
        except KeyError:
            print('decode_hfxl_preamble: dibits new list', l)
            self._constellation_index = MODE_8PSK

        if self._constellation_index > MODE_8PSK:
            self._data_scramble[:] = 1
        print('decode_hfxl_preamble: ds=', ds, l)
        print('decode_hfxl_preamble: dibits=', self._dibits)
        return success

    def make_reinserted_preamble(self, offset, success):
        """ offset=  0 -> 1st reinsesrted preamble
            offset=-72 -> all following reinserted preambles"""
        a = common.make_scr(REINSERTED_PREAMBLE[offset:], REINSERTED_PREAMBLE[offset:])
        a['symb'][-71:-71+3*13] = 0 ## D0,D1,D2
        print('make_reinserted_preamble', offset, success, len(a['symb']))
        if not success:
            self._frame_counter = -2
        return a

    def make_hfxl_preamble(self, success):
        a = common.make_scr(HFXL_PREAMBLE, HFXL_PREAMBLE)
        a['symb'][0:7*13] = 0
        if not success:
            self._frame_counter = -2
        return a

    def make_data_frame(self, success):
        self._preamble_offset = -72 ## all following reinserted preambles start at index -72
        a = np.zeros(256+31, common.SYMB_SCRAMBLE_DTYPE)
        if self.is_12800bpsBurst():
            a['scramble'][:256] = QAM64p['points'][[self._scrp.next() for _ in range(256)]]
        elif self.is_HFXL():
            a['scramble'][:256] = self._data_scramble
        elif self.is_plain_110C():
            a['scramble'][:256] = self._data_scramble
        else:
            ## TODO: generate an error message
            pass
        a['scramble_xor'][:256] = self._data_scramble_xor
        if self.is_plain_110C() or self.is_12800bpsBurst():
            n = (self._frame_counter-1)%72
            if self._frame_counter == 72:
                self._frame_counter = -1 ## trigger reinserted preamble
            m = n%18
            if m == 0:
                cnt = n//18
                self._mp = (1,1,1,1,1,1,1,0)+self._mode['rate']+self._mode['interleaver']+MP_COUNTER[cnt]+(0,)
                print('new mini-probe signs n=',n,'m=',m, 'cnt=',cnt, self._mp)
            print('make_data_frame', m, self._mp[m])
            a['symb'][256:]     = MINI_PROBE[self._mp[m]]
            a['scramble'][256:] = MINI_PROBE[self._mp[m]]
        elif self.is_HFXL(): ## only plus sign mini-probes are used
            a['symb'][256:]     = MINI_PROBE[0]
            a['scramble'][256:] = MINI_PROBE[0]
        else:
            pass # TODO
        if not success:
            self._frame_counter = -2
        return a

    def decode_soft_dec(self, soft_dec):
        if self.is_12800bpsBurst():
            print('decode_soft_dec', len(soft_dec))
            n = len(soft_dec) // 32
            soft_bits = np.zeros(2*n, dtype=np.float32)
            for i in range(n):
                w = np.sum(soft_dec[32*i:32*(i+1)].reshape(8,4),0)
                b = FROM_WALSH4[np.packbits(w>0)[0]] ## TODO use 2nd half of WALSH bits
                abs_soft_dec = np.mean(np.abs(w))
                print('WALSH', i, w, b, abs_soft_dec)
                soft_bits[2*i]   = abs_soft_dec*(2*(b>>1)-1)
                soft_bits[2*i+1] = abs_soft_dec*(2*(b &1)-1)

            return soft_bits>0,100.0
        elif self.is_HFXL():
            ## TODO
            return np.zeros(0, dtype=np.float32),0.0

        elif self.is_plain_110C():
            r = self._deintl_depunct.load(soft_dec)
            if r.shape[0] == 0:
                return np.zeros(0, dtype=np.float32),0.0
            self._viterbi_decoder.reset()
            decoded_bits = np.roll(self._viterbi_decoder.udpate(r), 7)
            print('bits=', decoded_bits[:100])
            quality = 120.0*self._viterbi_decoder.quality()/(2*len(decoded_bits))
            print('quality={}% ({},{})'.format(quality,
                                               self._viterbi_decoder.quality(),
                                               len(decoded_bits)))
            return decoded_bits,quality
        else:
            return np.zeros(0, dtype=np.float32),0.0

    @staticmethod
    def get_preamble():
        """preamble symbols + scrambler"""
        return common.make_scr(PREAMBLE, PREAMBLE)

    def get_preamble_z(self):
        """preamble symbols for preamble correlation"""
        return 2,np.array([z for z in PREAMBLE for _ in range(self._sps)])

if __name__ == '__main__':
    print(PREAMBLE)
    z = common.n_psk(8,PREAMBLE)
    cc = [np.sum(z[0:23]*np.conj(z[23*i:23*i+23])) for i in range(6)]
    print(np.abs(cc))
    print(np.angle(cc)/np.pi*4)
    print(all(z==PhysicalLayer.get_preamble()['symb']))
    print(len(PhysicalLayer.get_preamble()['symb']))
    s = ScrambleData()
    print([s.next(1) for _ in range(511)])
    print([s.next(1) for _ in range(511)] ==
          [s.next(1) for _ in range(511)])
    #print(QAM64)
    #print(QAM32)
    #print(QAM16)
    #print(PSK8)
    #print(QPSK)
    #print(BPSK)
    #print(MINI_PROBE_PLUS)
    #print(MINI_PROBE_MINUS)
    #print(MINI_PROBE_PLUS*MINI_PROBE_MINUS)
    #for i in range(len(QAM64)):
    #    print(QAM64['points'][i])

    print([s.next(6) for _ in range(256)])

    s = ScrambleDataP()
    assert(np.all(s.get_seq()[0:20]==np.array([0,2,4,3,3,6,4,5,7,6,7,0,5,5,4,3,5,4,3,7], dtype=np.uint8)))
