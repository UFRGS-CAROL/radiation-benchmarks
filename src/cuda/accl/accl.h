#ifndef SCCL_CU_H
#define SCCL_CU_H
double acclCuda(int *out, int *components, const int *in,
                 uint nFrames, uint nFramsPerStream, const int rows,
                 const int cols, int logs_active);

#endif
