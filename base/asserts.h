#ifndef BASE_ASSERTS_H_
#ifndef CHECK
#define BASE_ASSERTS_H_

#include <assert.h>
#include <cmath>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

//void print_trace (void);

#define CHECK(val)				\
  {						\
    assert(val);				\
  }						

#define CHECK_NEQ(lhs, rhs)			\
  {						\
    assert(lhs != rhs);				\
  }

#define CHECK_EQ(lhs, rhs)			\
  {						\
    assert(lhs == rhs);				\
  }						

#define CHECK_LT(lhs, rhs)			\
  {						\
    assert(lhs < rhs);				\
  }						

#define CHECK_GT(lhs, rhs)			\
  {						\
    assert(lhs > rhs);				\
  }						\

#define CHECK_LE(lhs, rhs)			\
  {						\
    assert(lhs <= rhs);				\
  }

#define CHECK_GE(lhs, rhs)			\
  {						\
    assert(lhs >= rhs);				\
  }						

#define CHECK_NEAR(lhs, rhs, tol)		\
  {                                             \
    assert(abs(lhs - rhs) <= tol);		\
  }

#define CHECK_NOTNULL(val)			\
  {						\
    assert(val != NULL);			\
  }

#define DISALLOW_COPY_AND_ASSIGN(TypeName)      \
  TypeName(TypeName&);                          \
  void operator=(TypeName);

// c++11
//#define DISALLOW_COPY_AND_ASSIGN(TypeName)
//TypeName(TypeName&) = delete;
//void operator=(TypeName) = delete;

#endif
#endif  // BASE_ASSERTS_H_
