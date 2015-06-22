set(H_SRCS logHelper.h)
set(C_SRCS logHelper.c)

set(logHelper_SRCS ${C_SRCS} ${H_SRCS})

add_library(logHelper STATIC ${logHelper_SRCS})
