// -*- c++ -*-

#ifndef _LIB_KALMAN_HSU_HPP_
#define _LIB_KALMAN_HSU_HPP_

#include <vector>

#include "filter_update.hpp"
#include "volk_allocator.hpp"


namespace gr {
namespace digitalhf {

// see
// [1] IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. IT-28, NO. 5, SEPTEMBER 1982 753
//     "Square Root Kalman Filtering for High-Speed Data Received over Fading Dispersive HF Channels
//     FRANK M. HSU
// [2] https://open.library.ubc.ca/collections/ubctheses/831/items/1.0096286

class kalman_hsu : public filter_update {
public:
  kalman_hsu(float q, float e);
  virtual ~kalman_hsu();

  static sptr make(float q, float e);

  virtual void reset();
  virtual gr_complex const* update(gr_complex const*, gr_complex const*);
  virtual void set_parameters(std::map<std::string, float>const &);

  inline float q() const { return _q; }
  inline float e() const { return _e; }

protected:
  void resize(size_t);

  inline unsigned idx(unsigned i, unsigned j) const {
    return i+j*(j-1)/2; // lower-triangular matrix index -> linear index
  }
private:
  typedef std::vector<gr_complex, volk_allocator<gr_complex > > complex_vec_type;
  typedef std::vector<float, volk_allocator<float> > real_vec_type;
  float _q;
  float _e;
  complex_vec_type _g; // n       -- kaman gain
  complex_vec_type _u; // n*(n-1)/2
  real_vec_type    _d; // n
  real_vec_type    _a; // n
  complex_vec_type _f; // n
  complex_vec_type _h; // n
} ;

} // namespace digitalhf
} // namespace gr
#endif // _LIB_KALMAN_HSU_HPP_
