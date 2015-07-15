set(H_SRCS logHelper.h)
set(C_SRCS logHelper.c)

set(logHelper_SRCS ${C_SRCS} ${H_SRCS})

add_library(logHelper STATIC ${logHelper_SRCS})

# In order to POSIX Libraries work using std=C99
set_target_properties(logHelper PROPERTIES COMPILE_DEFINITIONS "_POSIX_C_SOURCE=200112L")
if (MIC_NATIVE)
      set_target_properties(logHelper PROPERTIES COMPILE_DEFINITIONS "LOG=1;MIC_NATIVE=1;_POSIX_C_SOURCE=200112L")
endif(MIC_NATIVE)
