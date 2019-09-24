// -*- c++ -*-

#ifndef _LIB_RLS_HPP_
#define _LIB_RLS_HPP_

#include <vector>

#include "filter_update.hpp"
#include "volk_alloc.h"


namespace gr {
namespace digitalhf {

class rls : public filter_update {
public:
  virtual ~rls();

  static sptr make(float delta, float lambda);

  virtual void reset();
  virtual gr_complex const* update(gr_complex const*, gr_complex const*);
  virtual void set_parameters(std::map<std::string, float>const &);

protected:
  void resize(size_t);

private:
  rls(float delta, float lambda);

  typedef volk::vector<gr_complex> vec_type;
  float    _delta;
  float    _lambda;
  vec_type _gain;
  vec_type _inv_corr;
  vec_type _pu;
  vec_type _tmp;
} ;

} // namespace digitalhf
} // namespace gr
#endif // _LIB_RLS_HPP_
