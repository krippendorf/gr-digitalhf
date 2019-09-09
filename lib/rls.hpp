// -*- c++ -*-

#ifndef _LIB_RLS_HPP_
#define _LIB_RLS_HPP_

#include <vector>

#include "filter_update.hpp"
#include "volk_allocator.hpp"


namespace gr {
namespace digitalhf {

class rls : public filter_update {
public:
  rls(float delta, float lambda);
  virtual ~rls();

  static sptr make(float delta, float lambda);

  virtual void reset();
  virtual gr_complex const* update(gr_complex const*, gr_complex const*);

protected:
  void resize(size_t);

private:
  float _delta;
  float _lambda;
  std::vector<gr_complex, volk_allocator<gr_complex> > _gain;
  std::vector<gr_complex, volk_allocator<gr_complex> > _inv_corr;
} ;

} // namespace digitalhf
} // namespace gr
#endif // _LIB_RLS_HPP_
