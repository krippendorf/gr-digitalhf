## -*- python -*-

import numpy as np

CONST_DTYPE=np.dtype([('points',  np.complex64),
                      ('symbols', np.int32)])

SYMB_SCRAMBLE_DTYPE=np.dtype([('symb',     np.complex64),
                              ('scramble', np.complex64)])

def n_psk(n,x):
    """n-ary PSK constellation"""
    return np.complex64(np.exp(2j*np.pi*x/n))

def freq_est(z):
    """Data-Aided Frequency Estimation for Burst Digital Transmission,
        Umberto Mengali and M. Morelli, IEEE TRANSACTIONS ON COMMUNICATIONS,
        VOL. 45, NO. 1, JANUARY 1997"""
    L0 = len(z)
    N  = L0//2
    R  = np.zeros(N+1, dtype=np.complex64)
    for i in range(N+1):
        R[i] = 1.0/(L0-i)*np.sum(z[i:]*np.conj(z[0:L0-i])) ## eq (3)
    m  = np.arange(N+1, dtype=np.float32)
    w  = 3*((L0-m)*(L0-m+1)-N*(L0-N))/(N*(4*N*N - 6*N*L0 + 3*L0*L0-1)) ## eq (9)
    mod_2pi = lambda x : np.mod(x-np.pi, 2*np.pi) - np.pi
    return np.sum(w[1:] * mod_2pi(np.diff(np.angle(R))))   ## eq (8)

class Depuncturer(object):
    def __init__(self, repeat=1, puncture_pattern=['1','1']):
        assert(repeat >= 1)
        self._repeat = repeat
        self._num_patterns   = num_patterns = len(puncture_pattern)
        assert(num_patterns >= 2)
        assert(all([len(puncture_pattern[0]) == len(p) for p in puncture_pattern[1:]]))
        m = np.array([x=='1' for y in puncture_pattern for x in y])
        self._num_unpacked   = len(m)
        self._num_packed     = np.sum(m)
        self._pattern        = m.reshape(num_patterns, self._num_unpacked//num_patterns).transpose().reshape(1, self._num_unpacked)[0]
        self._range_packed   = np.arange(self._num_packed)
        self._range_unpacked = np.arange(self._num_unpacked)

    def process(self, x):
        n  = len(x)
        assert(n%(self._num_packed * self._repeat) == 0)
        ## (1) unpack
        xd = np.zeros(n * self._num_unpacked // self._num_packed, dtype=np.float64)
        i = 0
        j = 0
        while i < len(xd):
            xd[(i + self._range_unpacked)[self._pattern]] += x[j + self._range_packed]
            j += self._num_packed
            i += self._num_unpacked
        assert(j == n)
        assert(i == len(xd))
        if self._repeat == 1:
            return xd

        ## (2) combine repeated data
        xu = np.zeros(len(xd) // self._repeat, dtype=np.float64)
        i = 0
        j = 0
        m = self._num_patterns
        r = np.arange(m)
        while i < len(xu):
            for k in range(self._repeat):
                xu[i + r] += xd[j + r]
                j += m
            i += m
        assert(i == len(xu))
        assert(j == len(xd))
        return xu

if __name__ == '__main__':
    idx=np.arange(3)
    z=np.exp(1j*idx*0.056+1j)
    print(freq_est(z)/0.056)

