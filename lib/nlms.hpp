// -*- c++ -*-

#ifndef _LIB_NLMS_HPP_
#define _LIB_NLMS_HPP_

#include <vector>

#include "filter_update.hpp"
#include "volk_alloc.h"


namespace gr {
namespace digitalhf {

class nlms : public filter_update {
public:
  virtual ~nlms();

  static sptr make(float mu, float delta);

  virtual void reset();
  virtual gr_complex const* update(gr_complex const*, gr_complex const*);
  virtual void set_parameters(std::map<std::string, float>const &);

protected:
  void resize(size_t);

private:
  nlms(float mu, float delta);

  typedef volk::vector<gr_complex> complex_vec_type;
  typedef volk::vector<float> real_vec_type;
  float _mu;
  float _delta;
  complex_vec_type _gain;
  complex_vec_type _tmp;
  real_vec_type    _t1;
} ;

} // namespace digitalhf
} // namespace gr
#endif // _LIB_NLMS_HPP_
