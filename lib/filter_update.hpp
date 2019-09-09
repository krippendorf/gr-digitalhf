// -*- C++ -*-

#ifndef _LIB_FILTER_UPDATE_HPP_
#define _LIB_FILTER_UPDATE_HPP_

#include <memory>
#include <string>

#include <boost/noncopyable.hpp>
#include <gnuradio/gr_complex.h>

namespace gr {
namespace digitalhf {

class filter_update : private boost::noncopyable {
public:
  typedef std::unique_ptr<filter_update> sptr;

  virtual void reset() = 0;
  virtual gr_complex const* update(gr_complex const*, gr_complex const*) = 0;

protected:
private:
} ;


} // namespace digitalhf
} // namespace gr
#endif // _LIB_FILTER_UPDATE_HPP_
