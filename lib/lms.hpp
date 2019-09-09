// -*- c++ -*-

#ifndef _LIB_LMS_HPP_
#define _LIB_LMS_HPP_

#include <vector>

#include "filter_update.hpp"
#include "volk_allocator.hpp"


namespace gr {
namespace digitalhf {

class lms : public filter_update {
public:
  lms(float mu);
  virtual ~lms();

  static sptr make(float mu);

  virtual void reset();
  virtual gr_complex const* update(gr_complex const*, gr_complex const*);

protected:
  void resize(size_t);

private:
  float _mu;
  std::vector<gr_complex, volk_allocator<gr_complex> > _gain;
} ;

} // namespace digitalhf
} // namespace gr
#endif // _LIB_LMS_HPP_
