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
  virtual void set_parameters(std::map<std::string, float>const &);

protected:
  void resize(size_t);

private:
  typedef std::vector<gr_complex, volk_allocator<gr_complex> > vec_type;
  float    _mu;
  vec_type _gain;
  vec_type _tmp;
} ;

} // namespace digitalhf
} // namespace gr
#endif // _LIB_LMS_HPP_
