// -*- c++ -*-

#ifndef _LIB_KALMAN_EXP_HPP_
#define _LIB_KALMAN_EXP_HPP_

#include <vector>

#include "filter_update.hpp"
#include "volk_allocator.hpp"

// see
// [1] https://open.library.ubc.ca/collections/ubctheses/831/items/1.0096286

namespace gr {
namespace digitalhf {

class kalman_exp : public filter_update {
public:
  kalman_exp(float r, float lambda);
  virtual ~kalman_exp();

  static sptr make(float r, float lambda);

  virtual void reset();
  virtual gr_complex const* update(gr_complex const*, gr_complex const*);
  virtual void set_parameters(std::map<std::string, float>const &);

protected:
  void resize(size_t);

private:
  typedef std::vector<gr_complex, volk_allocator<gr_complex> > vec_type;
  float    _lambda;
  float    _r;
  vec_type _gain;
  vec_type _p;
  vec_type _temp;
  vec_type _t1;
} ;

} // namespace digitalhf
} // namespace gr
#endif // _LIB_KALMAN_EXP_HPP_
