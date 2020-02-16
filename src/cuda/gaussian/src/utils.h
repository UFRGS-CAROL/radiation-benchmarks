/*
 * utils.h
 *
 *  Created on: 15/02/2020
 *      Author: fernando
 */

#ifndef UTILS_H_
#define UTILS_H_

#ifdef RD_WG_SIZE_0_0
#define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define MAXBLOCKSIZE RD_WG_SIZE
#else
#define MAXBLOCKSIZE 512
#endif

//2D defines. Go from specific to general
#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_XY RD_WG_SIZE
#else
#define BLOCK_SIZE_XY 4
#endif

void InitProblemOnce(char *filename);
void InitPerRun();
void ForwardSub();
void BackSub();
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
void PrintDeviceProperties();
void InitMat(float *ary, int nrow, int ncol);
void InitAry(float *ary, int ary_size);
void BackSub();


static int Size;
static float *a, *b, *finalVec;
static float *m;

static FILE *fp;
static unsigned int totalKernelTime = 0;

#endif /* UTILS_H_ */
