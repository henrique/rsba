#ifndef SHELL_UTILS_H
#define SHELL_UTILS_H

#include <stdio.h>

#define shUtilsPrint(...) printf(__VA_ARGS__);
#define shUtilsLog(...) printf(__VA_ARGS__);

#define _DEBUG_FUNCTION_START(...)					\
  {									\
    printf(__VA_ARGS__);						\
    printf("{\n");							\
  }

#define _DEBUG_FUNCTION_STATE(...)					\
  {									\
    printf("\t");							\
    printf(__VA_ARGS__);						\
    printf("\n");							\
  }

#define _DEBUG_FUNCTION_END(...)					\
  {									\
    printf("}");							\
    printf(__VA_ARGS__);						\
    printf("\n\n");							\
  }

#endif
