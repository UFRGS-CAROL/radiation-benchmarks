// -*- c++ -*-

// This code uses algorithm described in:
// "Fitting FFT onto G80 Architecture". Vasily Volkov and Brian Kazian, UC Berkeley CS258 project report. May 2008.
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_fp64: enable
//#pragma OPENCL EXTENSION cl_nv_compiler_options: enable

#ifndef M_PI
# define M_PI 3.14159265358979323846f
#endif

#ifndef M_SQRT1_2
# define M_SQRT1_2      0.70710678118654752440f
#endif


#define exp_1_8   (double2)(  1, -1 )//requires post-multiply by 1/sqrt(2)
#define exp_1_4   (double2)(  0, -1 )
#define exp_3_8   (double2)( -1, -1 )//requires post-multiply by 1/sqrt(2)

#define iexp_1_8   (double2)(  1, 1 )//requires post-multiply by 1/sqrt(2)
#define iexp_1_4   (double2)(  0, 1 )
#define iexp_3_8   (double2)( -1, 1 )//requires post-multiply by 1/sqrt(2)


inline void globalLoads8(double2 *data, __global double2 *in, int stride) {
    for( int i = 0; i < 8; i++ )
        data[i] = in[i*stride];
}

// Stores the data back to main array, this function should be called in 
// the middle of execution when the work is distributted. This way the second
// device can catch up the corrected data computed so far
inline void globalStoresMiddle(double2 *data, __global double2 *out, int stride) {
    for( int i = 0; i < 8; i++ )
        out[i*stride] = data[i];
}

// This function should be called only when the FFT execution is finished by 
// all the devices
inline void globalStores8(double2 *data, __global double2 *out, int stride) {
    int reversed[] = {0,4,2,6,1,5,3,7};

//#pragma unroll
    for( int i = 0; i < 8; i++ )
        out[i*stride] = data[reversed[i]];
}

inline void storex8( double2 *a, __local double *x, int sx ) {
    int reversed[] = {0,4,2,6,1,5,3,7};

//#pragma unroll
    for( int i = 0; i < 8; i++ )
        x[i*sx] = a[reversed[i]].x;
}

inline void storey8( double2 *a, __local double *x, int sx ) {
    int reversed[] = {0,4,2,6,1,5,3,7};

//#pragma unroll
    for( int i = 0; i < 8; i++ )
        x[i*sx] = a[reversed[i]].y;
}


inline void loadx8( double2 *a, __local double *x, int sx ) {
    for( int i = 0; i < 8; i++ )
        a[i].x = x[i*sx];
}

inline void loady8( double2 *a, __local double *x, int sx ) {
    for( int i = 0; i < 8; i++ )
        a[i].y = x[i*sx];
}


#define transpose( a, s, ds, l, dl, sync )                              \
{                                                                       \
    storex8( a, s, ds );  if( (sync)&8 ) barrier(CLK_LOCAL_MEM_FENCE);  \
    loadx8 ( a, l, dl );  if( (sync)&4 ) barrier(CLK_LOCAL_MEM_FENCE);  \
    storey8( a, s, ds );  if( (sync)&2 ) barrier(CLK_LOCAL_MEM_FENCE);  \
    loady8 ( a, l, dl );  if( (sync)&1 ) barrier(CLK_LOCAL_MEM_FENCE);  \
}

inline double2 exp_i( double phi ) {
//#ifdef USE_NATIVE
//    return (double2)( native_cos(phi), native_sin(phi) );
//#else
    return (double2)( cos(phi), sin(phi) );
//#endif
}

inline double2 cmplx_mul( double2 a, double2 b ) {
    return (double2)( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x );
}
inline double2 cm_fl_mul( double2 a, double  b ) {
    return (double2)( b*a.x, b*a.y );
}
inline double2 cmplx_add( double2 a, double2 b ) {
    return (double2)( a.x + b.x, a.y + b.y );
}
inline double2 cmplx_sub( double2 a, double2 b ) {
    return (double2)( a.x - b.x, a.y - b.y );
}


#define twiddle8(a, i, n )                                              \
{                                                                       \
    int reversed8[] = {0,4,2,6,1,5,3,7};                                \
    for( int j = 1; j < 8; j++ ){                                       \
        a[j] = cmplx_mul( a[j],exp_i((-2*M_PI*reversed8[j]/(n))*(i)) ); \
    }                                                                   \
}

#define FFT2(a0, a1)                            \
{                                               \
    double2 c0 = *a0;                           \
    *a0 = cmplx_add(c0,*a1);                    \
    *a1 = cmplx_sub(c0,*a1);                    \
}

#define FFT4(a0, a1, a2, a3)                    \
{                                               \
    FFT2( a0, a2 );                             \
    FFT2( a1, a3 );                             \
    *a3 = cmplx_mul(*a3,exp_1_4);               \
    FFT2( a0, a1 );                             \
    FFT2( a2, a3 );                             \
}

#define FFT8(a)                                                 \
{                                                               \
    FFT2( &a[0], &a[4] );                                       \
    FFT2( &a[1], &a[5] );                                       \
    FFT2( &a[2], &a[6] );                                       \
    FFT2( &a[3], &a[7] );                                       \
                                                                \
    a[5] = cm_fl_mul( cmplx_mul(a[5],exp_1_8) , M_SQRT1_2 );    \
    a[6] =  cmplx_mul( a[6] , exp_1_4);                         \
    a[7] = cm_fl_mul( cmplx_mul(a[7],exp_3_8) , M_SQRT1_2 );    \
                                                                \
    FFT4( &a[0], &a[1], &a[2], &a[3] );                         \
    FFT4( &a[4], &a[5], &a[6], &a[7] );                         \
}

#define itwiddle8( a, i, n )                                            \
{                                                                       \
    int reversed8[] = {0,4,2,6,1,5,3,7};                                \
    for( int j = 1; j < 8; j++ )                                        \
        a[j] = cmplx_mul(a[j] , exp_i((2*M_PI*reversed8[j]/(n))*(i)) ); \
}

#define IFFT2 FFT2

#define IFFT4( a0, a1, a2, a3 )                 \
{                                               \
    IFFT2( a0, a2 );                            \
    IFFT2( a1, a3 );                            \
    *a3 = cmplx_mul(*a3 , iexp_1_4);            \
    IFFT2( a0, a1 );                            \
    IFFT2( a2, a3);                             \
}

#define IFFT8( a )                                              \
{                                                               \
    IFFT2( &a[0], &a[4] );                                      \
    IFFT2( &a[1], &a[5] );                                      \
    IFFT2( &a[2], &a[6] );                                      \
    IFFT2( &a[3], &a[7] );                                      \
                                                                \
    a[5] = cm_fl_mul( cmplx_mul(a[5],iexp_1_8) , M_SQRT1_2 );   \
    a[6] = cmplx_mul( a[6] , iexp_1_4);                         \
    a[7] = cm_fl_mul( cmplx_mul(a[7],iexp_3_8) , M_SQRT1_2 );   \
                                                                \
    IFFT4( &a[0], &a[1], &a[2], &a[3] );                        \
    IFFT4( &a[4], &a[5], &a[6], &a[7] );                        \
}

///////////////////////////////////////////

//distr = 0,   0%  cpu | gpu 100%
//distr = 1,  33%  cpu | gpu  66%
//distr = 2,  66%  cpu | gpu  33%
//distr = 3, 100%  cpu | gpu   0%
__kernel void fft1D_512 (__global double2 *work, int distr, int fromGPU)
{

    int tid = get_local_id(0);
    int blockIdx = get_group_id(0) * 512 + tid;
    int hi = tid>>3;
    int lo = tid&7;
    double2 data[8];
    __local double smem[8*8*9];
    work = work + blockIdx;

    // CPU running
    if(fromGPU == 0) {
        switch(distr) {
            // 0% CPU, 100% GPU
        case 0:
            // do nothing
            break;

            // 33% CPU, 66% GPU
        case 1:
            // CPU execute after GPU, therefore execute last_33%
            globalLoads8(data, work, 64);
            FFT8( data );
            globalStores8(data, work, 64);
            break;
            // 66% CPU, 33% GPU
        case 2:
            // CPU execute after GPU, therefore execute last_66%
            globalLoads8(data, work, 64);
            twiddle8( data, tid, 512 );
            transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);
            FFT8( data );
            twiddle8( data, hi, 64 );
            transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);
            FFT8( data );
            globalStores8(data, work, 64);
            break;
            // 100% CPU, 0% GPU
        case 3:
            globalLoads8(data, work, 64);
            FFT8( data );
            twiddle8( data, tid, 512 );
            transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);
            FFT8( data );
            twiddle8( data, hi, 64 );
            transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);
            FFT8( data );
            globalStores8(data, work, 64);
            break;
        }
    }
    // GPU running
    else if(fromGPU == 1) {
        switch(distr) {
            // 0% CPU, 100% GPU
        case 0:
            globalLoads8(data, work, 64);
            FFT8( data );
            twiddle8( data, tid, 512 );
            transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);
            FFT8( data );
            twiddle8( data, hi, 64 );
            transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);
            FFT8( data );
            globalStores8(data, work, 64);
            break;

            // 33% CPU, 66% GPU
        case 1:
            // CPU execute after GPU, therefore execute first_66%
            globalLoads8(data, work, 64);
            FFT8( data );
            twiddle8( data, tid, 512 );
            transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);
            FFT8( data );
            twiddle8( data, hi, 64 );
            transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);
            globalStoresMiddle(data, work, 64);
            //globalStores8(data, work, 64);
            break;
            // 66% CPU, 33% GPU
        case 2:
            // CPU execute after GPU, therefore execute first_33%
            globalLoads8(data, work, 64);
            FFT8( data );
            globalStoresMiddle(data, work, 64);
            //globalStores8(data, work, 64);
            break;
            // 100% CPU, 0% GPU
        case 3:
            // do nothing
            break;
        }
    }
}
__kernel void ifft1D_512 (__global double2 *work, int distr, int fromGPU)
{
    //work[0].x = 666;
    int i;
    int tid = get_local_id(0);
    int blockIdx = get_group_id(0) * 512 + tid;
    int hi = tid>>3;
    int lo = tid&7;
    double2 data[8];
    __local double smem[8*8*9];

    // starting index of data to/from global memory
    work = work + blockIdx;
    globalLoads8(data, work, 64); // coalesced global reads

    // Inject an artificial error for testing the sensitivity of FFT
    // if( blockIdx == 0 ){ data[6] *= 1.001; }

    IFFT8( data );

    itwiddle8( data, tid, 512 );
    transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);

    IFFT8( data );

    itwiddle8( data, hi, 64 );
    transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);

    IFFT8( data );

    for(i=0; i<8; i++) {
        data[i].x = data[i].x/512.0f;
        data[i].y = data[i].y/512.0f;
    }

    globalStores8(data, work, 64);

}


__kernel void
chk1D_512(__global double2* work, int half_n_cmplx, __global int* fail)
{
    int i, tid = get_local_id(0);
    int blockIdx = get_group_id(0) * 512 + tid;
    double2 a[8], b[8];

    work = work + blockIdx;

    for (i = 0; i < 8; i++) {
        a[i] = work[i*64];
    }

    for (i = 0; i < 8; i++) {
        b[i] = work[half_n_cmplx+i*64];
    }

    for (i = 0; i < 8; i++) {
        if (a[i].x != b[i].x || a[i].y != b[i].y) {
            *fail = 1;
        }
    }
}

__kernel void
GoldChk(__global double2* gold, __global double2* resultCPU, int N, __global int* kerrors, double AVOIDZERO, double ACCEPTDIFF)
{
        int i = get_global_id(0);

	
        if ((fabs(gold[i].x)>AVOIDZERO)&&
        ((fabs((resultCPU[i].x-gold[i].x)/resultCPU[i].x)>ACCEPTDIFF)||
         (fabs((resultCPU[i].x-gold[i].x)/gold[i].x)>ACCEPTDIFF))) {
		    atomic_inc(kerrors);
	    }

        if ((fabs(gold[i].y)>AVOIDZERO)&&
        ((fabs((resultCPU[i].y-gold[i].y)/resultCPU[i].y)>ACCEPTDIFF)||
         (fabs((resultCPU[i].y-gold[i].y)/gold[i].y)>ACCEPTDIFF))) {
		    atomic_inc(kerrors);
	    }

}
