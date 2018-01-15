#ifndef _pointers_h
#define _pointers_h

#ifdef _WIN32
#include <memory>
using ::std::unique_ptr;
using ::std::shared_ptr;
using ::std::weak_ptr;
using ::std::make_shared;

#else

#if __cplusplus < 201103L
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
using ::boost::shared_ptr;
using ::boost::weak_ptr;
using ::boost::make_shared;
#define unique_ptr shared_ptr
#define make_unique make_shared
#else
#include <memory>
using ::std::unique_ptr;
using ::std::shared_ptr;
using ::std::weak_ptr;
using ::std::make_shared;

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif

#endif // _WIN32

#endif
