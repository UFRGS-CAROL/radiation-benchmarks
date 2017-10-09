#ifndef CoMD_info_hpp
#define CoMD_info_hpp

#define CoMD_VARIANT "CoMD-cuda"
#define CoMD_HOSTNAME "fernando-apu"
#define CoMD_KERNEL_NAME "'Linux'"
#define CoMD_KERNEL_RELEASE "'4.4.0-96-generic'"
#define CoMD_PROCESSOR "'x86_64'"

#define CoMD_COMPILER "'/usr/bin/gcc'"
#define CoMD_COMPILER_VERSION "'gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609'"
#define CoMD_CFLAGS "'-std=c99 -Wno-unused-result -DMAXATOMS=64  -DNDEBUG -g -O3 -DCOMD_DOUBLE  -I/usr/local/cuda-8.0//include'"
#define CoMD_LDFLAGS "'-lm -lstdc++ -L/usr/local/cuda-8.0//lib64 -lcudart'"

#endif
