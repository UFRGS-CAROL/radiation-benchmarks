#ifndef TYPE_H_
#define TYPE_H_

#define HALF 16
#define FLOAT 32
#define DOUBLE 64

#include <stdlib.h>
#include <stdio.h>

#if REAL_TYPE == HALF


typedef float real_t;

#define FLT_MAX 1E+37

//#define REAL_INFINITY 0x7C00
#define REAL_INFINITY 0x7F800000

//void transform_float_to_half_array(real_t_device* dst, float* src, size_t n);

//---------------------------------------------------------------------------------------------------

#elif REAL_TYPE == FLOAT

//FLOAT----------------------------------------------------------------------------------------------
// Single precision
typedef float real_t;
typedef real_t real_t_device;

#define FLT_MAX 1E+37

#define REAL_INFINITY 0x7F800000
//---------------------------------------------------------------------------------------------------

#elif REAL_TYPE == DOUBLE

//DOUBLE----------------------------------------------------------------------------------------------
//Double precision
typedef double real_t;
typedef real_t real_t_device;

#define FLT_MAX 1E+307

#define REAL_INFINITY 0x7FF0000000000000

//---------------------------------------------------------------------------------------------------
#endif

typedef struct __device_builtin__ {
	real_t x;
	real_t y;
	real_t z;
} real_t3;

#define REAL_RAND_MAX FLT_MAX

int fread_float_to_real_t(real_t* dst, size_t siz, size_t times, FILE* fp);

#if REAL_TYPE == HALF

#ifdef __cplusplus
extern "C"
#endif
void run_cuda_gemm_half(int TA, int TB, int M, int N, int K, real_t ALPHA, real_t *A_gpu,
		int lda, real_t *B_gpu, int ldb, real_t BETA, real_t *C_gpu, int ldc);
#endif
//#ifdef __NVCC__
//__device__ __forceinline__ real_t_device exp_real(real_t_device x);
//__device__ __forceinline__ real_t_device floor_real(real_t_device x);
//__device__ __forceinline__ real_t_device pow_real(real_t_device x,
//		real_t_device y);
//__device__ __forceinline__ real_t_device sqrt_real(real_t_device x);
//__device__ __forceinline__ real_t_device fabs_real(real_t_device x);
//__device__ __forceinline__ real_t_device log_real(real_t_device x);
//__device__ __forceinline__ real_t_device atomic_add_real(real_t_device *x,
//		real_t_device val);
//__device__ __forceinline__ real_t_device cos_real(real_t_device x);
//__device__ __forceinline__ real_t_device sin_real(real_t_device x);
//#endif

#endif /* TYPE_H_ */
