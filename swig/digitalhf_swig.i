/* -*- c++ -*- */

#define DIGITALHF_API

%include "gnuradio.i"			// the common stuff

//load generated python docstrings
%include "digitalhf_swig_doc.i"

%{
#include "digitalhf/adaptive_dfe.h"
#include "digitalhf/doppler_correction_cc.h"
#include "digitalhf/oqpsk_demodulator.h"
#include "digitalhf/viterbi27.h"
#include "digitalhf/viterbi29.h"
#include "digitalhf/viterbi39.h"
#include "digitalhf/viterbi48.h"
#include "digitalhf/vector_pll_cc.h"
#include "digitalhf/vector_early_late_cc.h"
%}

%include "digitalhf/adaptive_dfe.h"
GR_SWIG_BLOCK_MAGIC2(digitalhf, adaptive_dfe);

%include "digitalhf/doppler_correction_cc.h"
GR_SWIG_BLOCK_MAGIC2(digitalhf, doppler_correction_cc);

%include "digitalhf/vector_pll_cc.h"
GR_SWIG_BLOCK_MAGIC2(digitalhf, vector_pll_cc);

%include "digitalhf/oqpsk_demodulator.h"
GR_SWIG_BLOCK_MAGIC2(digitalhf, oqpsk_demodulator);

%include "digitalhf/vector_early_late_cc.h"
GR_SWIG_BLOCK_MAGIC2(digitalhf, vector_early_late_cc);

/* FIXME */
%include "digitalhf/viterbi27.h"
GR_SWIG_BLOCK_MAGIC2(digitalhf, viterbi27);
%include "digitalhf/viterbi29.h"
GR_SWIG_BLOCK_MAGIC2(digitalhf, viterbi29);
%include "digitalhf/viterbi39.h"
GR_SWIG_BLOCK_MAGIC2(digitalhf, viterbi39);
%include "digitalhf/viterbi48.h"
GR_SWIG_BLOCK_MAGIC2(digitalhf, viterbi48);
