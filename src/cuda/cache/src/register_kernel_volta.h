#ifndef REGISTER_KERNEL_VOLTA_H_
#define REGISTER_KERNEL_VOLTA_H_

#include "utils.h"
#include "Parameters.h"

#include "device_functions.h"

__constant__ __device__ static uint32 reg_array[RF_SIZE][2] = {
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
		{ 0xffffffff, 0x00000000 },
};

__global__ void test_register_file_kernel_volta(uint32 *rf1, uint32 *rf2, uint32 *rf3, const uint32 zero_or_one, const uint64 sleep_cycles) {
	const uint32 i =  (blockDim.x * blockIdx.x + threadIdx.x) * RF_SIZE;
	register uint32 r0 = __ldg(&(reg_array[0][zero_or_one]));
	register uint32 r1 = __ldg(&(reg_array[1][zero_or_one]));
	register uint32 r2 = __ldg(&(reg_array[2][zero_or_one]));
	register uint32 r3 = __ldg(&(reg_array[3][zero_or_one]));
	register uint32 r4 = __ldg(&(reg_array[4][zero_or_one]));
	register uint32 r5 = __ldg(&(reg_array[5][zero_or_one]));
	register uint32 r6 = __ldg(&(reg_array[6][zero_or_one]));
	register uint32 r7 = __ldg(&(reg_array[7][zero_or_one]));
	register uint32 r8 = __ldg(&(reg_array[8][zero_or_one]));
	register uint32 r9 = __ldg(&(reg_array[9][zero_or_one]));
	register uint32 r10 = __ldg(&(reg_array[10][zero_or_one]));
	register uint32 r11 = __ldg(&(reg_array[11][zero_or_one]));
	register uint32 r12 = __ldg(&(reg_array[12][zero_or_one]));
	register uint32 r13 = __ldg(&(reg_array[13][zero_or_one]));
	register uint32 r14 = __ldg(&(reg_array[14][zero_or_one]));
	register uint32 r15 = __ldg(&(reg_array[15][zero_or_one]));
	register uint32 r16 = __ldg(&(reg_array[16][zero_or_one]));
	register uint32 r17 = __ldg(&(reg_array[17][zero_or_one]));
	register uint32 r18 = __ldg(&(reg_array[18][zero_or_one]));
	register uint32 r19 = __ldg(&(reg_array[19][zero_or_one]));
	register uint32 r20 = __ldg(&(reg_array[20][zero_or_one]));
	register uint32 r21 = __ldg(&(reg_array[21][zero_or_one]));
	register uint32 r22 = __ldg(&(reg_array[22][zero_or_one]));
	register uint32 r23 = __ldg(&(reg_array[23][zero_or_one]));
	register uint32 r24 = __ldg(&(reg_array[24][zero_or_one]));
	register uint32 r25 = __ldg(&(reg_array[25][zero_or_one]));
	register uint32 r26 = __ldg(&(reg_array[26][zero_or_one]));
	register uint32 r27 = __ldg(&(reg_array[27][zero_or_one]));
	register uint32 r28 = __ldg(&(reg_array[28][zero_or_one]));
	register uint32 r29 = __ldg(&(reg_array[29][zero_or_one]));
	register uint32 r30 = __ldg(&(reg_array[30][zero_or_one]));
	register uint32 r31 = __ldg(&(reg_array[31][zero_or_one]));
	register uint32 r32 = __ldg(&(reg_array[32][zero_or_one]));
	register uint32 r33 = __ldg(&(reg_array[33][zero_or_one]));
	register uint32 r34 = __ldg(&(reg_array[34][zero_or_one]));
	register uint32 r35 = __ldg(&(reg_array[35][zero_or_one]));
	register uint32 r36 = __ldg(&(reg_array[36][zero_or_one]));
	register uint32 r37 = __ldg(&(reg_array[37][zero_or_one]));
	register uint32 r38 = __ldg(&(reg_array[38][zero_or_one]));
	register uint32 r39 = __ldg(&(reg_array[39][zero_or_one]));
	register uint32 r40 = __ldg(&(reg_array[40][zero_or_one]));
	register uint32 r41 = __ldg(&(reg_array[41][zero_or_one]));
	register uint32 r42 = __ldg(&(reg_array[42][zero_or_one]));
	register uint32 r43 = __ldg(&(reg_array[43][zero_or_one]));
	register uint32 r44 = __ldg(&(reg_array[44][zero_or_one]));
	register uint32 r45 = __ldg(&(reg_array[45][zero_or_one]));
	register uint32 r46 = __ldg(&(reg_array[46][zero_or_one]));
	register uint32 r47 = __ldg(&(reg_array[47][zero_or_one]));
	register uint32 r48 = __ldg(&(reg_array[48][zero_or_one]));
	register uint32 r49 = __ldg(&(reg_array[49][zero_or_one]));
	register uint32 r50 = __ldg(&(reg_array[50][zero_or_one]));
	register uint32 r51 = __ldg(&(reg_array[51][zero_or_one]));
	register uint32 r52 = __ldg(&(reg_array[52][zero_or_one]));
	register uint32 r53 = __ldg(&(reg_array[53][zero_or_one]));
	register uint32 r54 = __ldg(&(reg_array[54][zero_or_one]));
	register uint32 r55 = __ldg(&(reg_array[55][zero_or_one]));
	register uint32 r56 = __ldg(&(reg_array[56][zero_or_one]));
	register uint32 r57 = __ldg(&(reg_array[57][zero_or_one]));
	register uint32 r58 = __ldg(&(reg_array[58][zero_or_one]));
	register uint32 r59 = __ldg(&(reg_array[59][zero_or_one]));
	register uint32 r60 = __ldg(&(reg_array[60][zero_or_one]));
	register uint32 r61 = __ldg(&(reg_array[61][zero_or_one]));
	register uint32 r62 = __ldg(&(reg_array[62][zero_or_one]));
	register uint32 r63 = __ldg(&(reg_array[63][zero_or_one]));
	register uint32 r64 = __ldg(&(reg_array[64][zero_or_one]));
	register uint32 r65 = __ldg(&(reg_array[65][zero_or_one]));
	register uint32 r66 = __ldg(&(reg_array[66][zero_or_one]));
	register uint32 r67 = __ldg(&(reg_array[67][zero_or_one]));
	register uint32 r68 = __ldg(&(reg_array[68][zero_or_one]));
	register uint32 r69 = __ldg(&(reg_array[69][zero_or_one]));
	register uint32 r70 = __ldg(&(reg_array[70][zero_or_one]));
	register uint32 r71 = __ldg(&(reg_array[71][zero_or_one]));
	register uint32 r72 = __ldg(&(reg_array[72][zero_or_one]));
	register uint32 r73 = __ldg(&(reg_array[73][zero_or_one]));
	register uint32 r74 = __ldg(&(reg_array[74][zero_or_one]));
	register uint32 r75 = __ldg(&(reg_array[75][zero_or_one]));
	register uint32 r76 = __ldg(&(reg_array[76][zero_or_one]));
	register uint32 r77 = __ldg(&(reg_array[77][zero_or_one]));
	register uint32 r78 = __ldg(&(reg_array[78][zero_or_one]));
	register uint32 r79 = __ldg(&(reg_array[79][zero_or_one]));
	register uint32 r80 = __ldg(&(reg_array[80][zero_or_one]));
	register uint32 r81 = __ldg(&(reg_array[81][zero_or_one]));
	register uint32 r82 = __ldg(&(reg_array[82][zero_or_one]));
	register uint32 r83 = __ldg(&(reg_array[83][zero_or_one]));
	register uint32 r84 = __ldg(&(reg_array[84][zero_or_one]));
	register uint32 r85 = __ldg(&(reg_array[85][zero_or_one]));
	register uint32 r86 = __ldg(&(reg_array[86][zero_or_one]));
	register uint32 r87 = __ldg(&(reg_array[87][zero_or_one]));
	register uint32 r88 = __ldg(&(reg_array[88][zero_or_one]));
	register uint32 r89 = __ldg(&(reg_array[89][zero_or_one]));
	register uint32 r90 = __ldg(&(reg_array[90][zero_or_one]));
	register uint32 r91 = __ldg(&(reg_array[91][zero_or_one]));
	register uint32 r92 = __ldg(&(reg_array[92][zero_or_one]));
	register uint32 r93 = __ldg(&(reg_array[93][zero_or_one]));
	register uint32 r94 = __ldg(&(reg_array[94][zero_or_one]));
	register uint32 r95 = __ldg(&(reg_array[95][zero_or_one]));
	register uint32 r96 = __ldg(&(reg_array[96][zero_or_one]));
	register uint32 r97 = __ldg(&(reg_array[97][zero_or_one]));
	register uint32 r98 = __ldg(&(reg_array[98][zero_or_one]));
	register uint32 r99 = __ldg(&(reg_array[99][zero_or_one]));
	register uint32 r100 = __ldg(&(reg_array[100][zero_or_one]));
	register uint32 r101 = __ldg(&(reg_array[101][zero_or_one]));
	register uint32 r102 = __ldg(&(reg_array[102][zero_or_one]));
	register uint32 r103 = __ldg(&(reg_array[103][zero_or_one]));
	register uint32 r104 = __ldg(&(reg_array[104][zero_or_one]));
	register uint32 r105 = __ldg(&(reg_array[105][zero_or_one]));
	register uint32 r106 = __ldg(&(reg_array[106][zero_or_one]));
	register uint32 r107 = __ldg(&(reg_array[107][zero_or_one]));
	register uint32 r108 = __ldg(&(reg_array[108][zero_or_one]));
	register uint32 r109 = __ldg(&(reg_array[109][zero_or_one]));
	register uint32 r110 = __ldg(&(reg_array[110][zero_or_one]));
	register uint32 r111 = __ldg(&(reg_array[111][zero_or_one]));
	register uint32 r112 = __ldg(&(reg_array[112][zero_or_one]));
	register uint32 r113 = __ldg(&(reg_array[113][zero_or_one]));
	register uint32 r114 = __ldg(&(reg_array[114][zero_or_one]));
	register uint32 r115 = __ldg(&(reg_array[115][zero_or_one]));
	register uint32 r116 = __ldg(&(reg_array[116][zero_or_one]));
	register uint32 r117 = __ldg(&(reg_array[117][zero_or_one]));
	register uint32 r118 = __ldg(&(reg_array[118][zero_or_one]));
	register uint32 r119 = __ldg(&(reg_array[119][zero_or_one]));
	register uint32 r120 = __ldg(&(reg_array[120][zero_or_one]));
	register uint32 r121 = __ldg(&(reg_array[121][zero_or_one]));
	register uint32 r122 = __ldg(&(reg_array[122][zero_or_one]));
	register uint32 r123 = __ldg(&(reg_array[123][zero_or_one]));
	register uint32 r124 = __ldg(&(reg_array[124][zero_or_one]));
	register uint32 r125 = __ldg(&(reg_array[125][zero_or_one]));
	register uint32 r126 = __ldg(&(reg_array[126][zero_or_one]));
	register uint32 r127 = __ldg(&(reg_array[127][zero_or_one]));
	register uint32 r128 = __ldg(&(reg_array[128][zero_or_one]));
	register uint32 r129 = __ldg(&(reg_array[129][zero_or_one]));
	register uint32 r130 = __ldg(&(reg_array[130][zero_or_one]));
	register uint32 r131 = __ldg(&(reg_array[131][zero_or_one]));
	register uint32 r132 = __ldg(&(reg_array[132][zero_or_one]));
	register uint32 r133 = __ldg(&(reg_array[133][zero_or_one]));
	register uint32 r134 = __ldg(&(reg_array[134][zero_or_one]));
	register uint32 r135 = __ldg(&(reg_array[135][zero_or_one]));
	register uint32 r136 = __ldg(&(reg_array[136][zero_or_one]));
	register uint32 r137 = __ldg(&(reg_array[137][zero_or_one]));
	register uint32 r138 = __ldg(&(reg_array[138][zero_or_one]));
	register uint32 r139 = __ldg(&(reg_array[139][zero_or_one]));
	register uint32 r140 = __ldg(&(reg_array[140][zero_or_one]));
	register uint32 r141 = __ldg(&(reg_array[141][zero_or_one]));
	register uint32 r142 = __ldg(&(reg_array[142][zero_or_one]));
	register uint32 r143 = __ldg(&(reg_array[143][zero_or_one]));
	register uint32 r144 = __ldg(&(reg_array[144][zero_or_one]));
	register uint32 r145 = __ldg(&(reg_array[145][zero_or_one]));
	register uint32 r146 = __ldg(&(reg_array[146][zero_or_one]));
	register uint32 r147 = __ldg(&(reg_array[147][zero_or_one]));
	register uint32 r148 = __ldg(&(reg_array[148][zero_or_one]));
	register uint32 r149 = __ldg(&(reg_array[149][zero_or_one]));
	register uint32 r150 = __ldg(&(reg_array[150][zero_or_one]));
	register uint32 r151 = __ldg(&(reg_array[151][zero_or_one]));
	register uint32 r152 = __ldg(&(reg_array[152][zero_or_one]));
	register uint32 r153 = __ldg(&(reg_array[153][zero_or_one]));
	register uint32 r154 = __ldg(&(reg_array[154][zero_or_one]));
	register uint32 r155 = __ldg(&(reg_array[155][zero_or_one]));
	register uint32 r156 = __ldg(&(reg_array[156][zero_or_one]));
	register uint32 r157 = __ldg(&(reg_array[157][zero_or_one]));
	register uint32 r158 = __ldg(&(reg_array[158][zero_or_one]));
	register uint32 r159 = __ldg(&(reg_array[159][zero_or_one]));
	register uint32 r160 = __ldg(&(reg_array[160][zero_or_one]));
	register uint32 r161 = __ldg(&(reg_array[161][zero_or_one]));
	register uint32 r162 = __ldg(&(reg_array[162][zero_or_one]));
	register uint32 r163 = __ldg(&(reg_array[163][zero_or_one]));
	register uint32 r164 = __ldg(&(reg_array[164][zero_or_one]));
	register uint32 r165 = __ldg(&(reg_array[165][zero_or_one]));
	register uint32 r166 = __ldg(&(reg_array[166][zero_or_one]));
	register uint32 r167 = __ldg(&(reg_array[167][zero_or_one]));
	register uint32 r168 = __ldg(&(reg_array[168][zero_or_one]));
	register uint32 r169 = __ldg(&(reg_array[169][zero_or_one]));
	register uint32 r170 = __ldg(&(reg_array[170][zero_or_one]));
	register uint32 r171 = __ldg(&(reg_array[171][zero_or_one]));
	register uint32 r172 = __ldg(&(reg_array[172][zero_or_one]));
	register uint32 r173 = __ldg(&(reg_array[173][zero_or_one]));
	register uint32 r174 = __ldg(&(reg_array[174][zero_or_one]));
	register uint32 r175 = __ldg(&(reg_array[175][zero_or_one]));
	register uint32 r176 = __ldg(&(reg_array[176][zero_or_one]));
	register uint32 r177 = __ldg(&(reg_array[177][zero_or_one]));
	register uint32 r178 = __ldg(&(reg_array[178][zero_or_one]));
	register uint32 r179 = __ldg(&(reg_array[179][zero_or_one]));
	register uint32 r180 = __ldg(&(reg_array[180][zero_or_one]));
	register uint32 r181 = __ldg(&(reg_array[181][zero_or_one]));
	register uint32 r182 = __ldg(&(reg_array[182][zero_or_one]));
	register uint32 r183 = __ldg(&(reg_array[183][zero_or_one]));
	register uint32 r184 = __ldg(&(reg_array[184][zero_or_one]));
	register uint32 r185 = __ldg(&(reg_array[185][zero_or_one]));
	register uint32 r186 = __ldg(&(reg_array[186][zero_or_one]));
	register uint32 r187 = __ldg(&(reg_array[187][zero_or_one]));
	register uint32 r188 = __ldg(&(reg_array[188][zero_or_one]));
	register uint32 r189 = __ldg(&(reg_array[189][zero_or_one]));
	register uint32 r190 = __ldg(&(reg_array[190][zero_or_one]));
	register uint32 r191 = __ldg(&(reg_array[191][zero_or_one]));
	register uint32 r192 = __ldg(&(reg_array[192][zero_or_one]));
	register uint32 r193 = __ldg(&(reg_array[193][zero_or_one]));
	register uint32 r194 = __ldg(&(reg_array[194][zero_or_one]));
	register uint32 r195 = __ldg(&(reg_array[195][zero_or_one]));
	register uint32 r196 = __ldg(&(reg_array[196][zero_or_one]));
	register uint32 r197 = __ldg(&(reg_array[197][zero_or_one]));
	register uint32 r198 = __ldg(&(reg_array[198][zero_or_one]));
	register uint32 r199 = __ldg(&(reg_array[199][zero_or_one]));
	register uint32 r200 = __ldg(&(reg_array[200][zero_or_one]));
	register uint32 r201 = __ldg(&(reg_array[201][zero_or_one]));
	register uint32 r202 = __ldg(&(reg_array[202][zero_or_one]));
	register uint32 r203 = __ldg(&(reg_array[203][zero_or_one]));
	register uint32 r204 = __ldg(&(reg_array[204][zero_or_one]));
	register uint32 r205 = __ldg(&(reg_array[205][zero_or_one]));
	register uint32 r206 = __ldg(&(reg_array[206][zero_or_one]));
	register uint32 r207 = __ldg(&(reg_array[207][zero_or_one]));
	register uint32 r208 = __ldg(&(reg_array[208][zero_or_one]));
	register uint32 r209 = __ldg(&(reg_array[209][zero_or_one]));
	register uint32 r210 = __ldg(&(reg_array[210][zero_or_one]));
	register uint32 r211 = __ldg(&(reg_array[211][zero_or_one]));
	register uint32 r212 = __ldg(&(reg_array[212][zero_or_one]));
	register uint32 r213 = __ldg(&(reg_array[213][zero_or_one]));
	register uint32 r214 = __ldg(&(reg_array[214][zero_or_one]));
	register uint32 r215 = __ldg(&(reg_array[215][zero_or_one]));
	register uint32 r216 = __ldg(&(reg_array[216][zero_or_one]));
	register uint32 r217 = __ldg(&(reg_array[217][zero_or_one]));
	register uint32 r218 = __ldg(&(reg_array[218][zero_or_one]));
	register uint32 r219 = __ldg(&(reg_array[219][zero_or_one]));
	register uint32 r220 = __ldg(&(reg_array[220][zero_or_one]));
	register uint32 r221 = __ldg(&(reg_array[221][zero_or_one]));
	register uint32 r222 = __ldg(&(reg_array[222][zero_or_one]));
	register uint32 r223 = __ldg(&(reg_array[223][zero_or_one]));
	register uint32 r224 = __ldg(&(reg_array[224][zero_or_one]));
	register uint32 r225 = __ldg(&(reg_array[225][zero_or_one]));
	register uint32 r226 = __ldg(&(reg_array[226][zero_or_one]));
	register uint32 r227 = __ldg(&(reg_array[227][zero_or_one]));
	register uint32 r228 = __ldg(&(reg_array[228][zero_or_one]));
	register uint32 r229 = __ldg(&(reg_array[229][zero_or_one]));
	register uint32 r230 = __ldg(&(reg_array[230][zero_or_one]));
	register uint32 r231 = __ldg(&(reg_array[231][zero_or_one]));
	register uint32 r232 = __ldg(&(reg_array[232][zero_or_one]));
	register uint32 r233 = __ldg(&(reg_array[233][zero_or_one]));
	register uint32 r234 = __ldg(&(reg_array[234][zero_or_one]));
	register uint32 r235 = __ldg(&(reg_array[235][zero_or_one]));
	register uint32 r236 = __ldg(&(reg_array[236][zero_or_one]));
	register uint32 r237 = __ldg(&(reg_array[237][zero_or_one]));
	register uint32 r238 = __ldg(&(reg_array[238][zero_or_one]));
	register uint32 r239 = __ldg(&(reg_array[239][zero_or_one]));
	register uint32 r240 = __ldg(&(reg_array[240][zero_or_one]));
	register uint32 r241 = __ldg(&(reg_array[241][zero_or_one]));
	register uint32 r242 = __ldg(&(reg_array[242][zero_or_one]));
	register uint32 r243 = __ldg(&(reg_array[243][zero_or_one]));
	register uint32 r244 = __ldg(&(reg_array[244][zero_or_one]));
	register uint32 r245 = __ldg(&(reg_array[245][zero_or_one]));
	register uint32 r246 = __ldg(&(reg_array[246][zero_or_one]));
	register uint32 r247 = __ldg(&(reg_array[247][zero_or_one]));
	register uint32 r248 = __ldg(&(reg_array[248][zero_or_one]));
	register uint32 r249 = __ldg(&(reg_array[249][zero_or_one]));
	register uint32 r250 = __ldg(&(reg_array[250][zero_or_one]));
	register uint32 r251 = __ldg(&(reg_array[251][zero_or_one]));
	register uint32 r252 = __ldg(&(reg_array[252][zero_or_one]));
	register uint32 r253 = __ldg(&(reg_array[253][zero_or_one]));
	register uint32 r254 = __ldg(&(reg_array[254][zero_or_one]));
	register uint32 r255 = __ldg(&(reg_array[255][zero_or_one]));

	sleep_cuda(sleep_cycles);

	rf1[i + 0] = r0;
	rf2[i + 0] = r0;
	rf3[i + 0] = r0;

	rf1[i + 1] = r1;
	rf2[i + 1] = r1;
	rf3[i + 1] = r1;

	rf1[i + 2] = r2;
	rf2[i + 2] = r2;
	rf3[i + 2] = r2;

	rf1[i + 3] = r3;
	rf2[i + 3] = r3;
	rf3[i + 3] = r3;

	rf1[i + 4] = r4;
	rf2[i + 4] = r4;
	rf3[i + 4] = r4;

	rf1[i + 5] = r5;
	rf2[i + 5] = r5;
	rf3[i + 5] = r5;

	rf1[i + 6] = r6;
	rf2[i + 6] = r6;
	rf3[i + 6] = r6;

	rf1[i + 7] = r7;
	rf2[i + 7] = r7;
	rf3[i + 7] = r7;

	rf1[i + 8] = r8;
	rf2[i + 8] = r8;
	rf3[i + 8] = r8;

	rf1[i + 9] = r9;
	rf2[i + 9] = r9;
	rf3[i + 9] = r9;

	rf1[i + 10] = r10;
	rf2[i + 10] = r10;
	rf3[i + 10] = r10;

	rf1[i + 11] = r11;
	rf2[i + 11] = r11;
	rf3[i + 11] = r11;

	rf1[i + 12] = r12;
	rf2[i + 12] = r12;
	rf3[i + 12] = r12;

	rf1[i + 13] = r13;
	rf2[i + 13] = r13;
	rf3[i + 13] = r13;

	rf1[i + 14] = r14;
	rf2[i + 14] = r14;
	rf3[i + 14] = r14;

	rf1[i + 15] = r15;
	rf2[i + 15] = r15;
	rf3[i + 15] = r15;

	rf1[i + 16] = r16;
	rf2[i + 16] = r16;
	rf3[i + 16] = r16;

	rf1[i + 17] = r17;
	rf2[i + 17] = r17;
	rf3[i + 17] = r17;

	rf1[i + 18] = r18;
	rf2[i + 18] = r18;
	rf3[i + 18] = r18;

	rf1[i + 19] = r19;
	rf2[i + 19] = r19;
	rf3[i + 19] = r19;

	rf1[i + 20] = r20;
	rf2[i + 20] = r20;
	rf3[i + 20] = r20;

	rf1[i + 21] = r21;
	rf2[i + 21] = r21;
	rf3[i + 21] = r21;

	rf1[i + 22] = r22;
	rf2[i + 22] = r22;
	rf3[i + 22] = r22;

	rf1[i + 23] = r23;
	rf2[i + 23] = r23;
	rf3[i + 23] = r23;

	rf1[i + 24] = r24;
	rf2[i + 24] = r24;
	rf3[i + 24] = r24;

	rf1[i + 25] = r25;
	rf2[i + 25] = r25;
	rf3[i + 25] = r25;

	rf1[i + 26] = r26;
	rf2[i + 26] = r26;
	rf3[i + 26] = r26;

	rf1[i + 27] = r27;
	rf2[i + 27] = r27;
	rf3[i + 27] = r27;

	rf1[i + 28] = r28;
	rf2[i + 28] = r28;
	rf3[i + 28] = r28;

	rf1[i + 29] = r29;
	rf2[i + 29] = r29;
	rf3[i + 29] = r29;

	rf1[i + 30] = r30;
	rf2[i + 30] = r30;
	rf3[i + 30] = r30;

	rf1[i + 31] = r31;
	rf2[i + 31] = r31;
	rf3[i + 31] = r31;

	rf1[i + 32] = r32;
	rf2[i + 32] = r32;
	rf3[i + 32] = r32;

	rf1[i + 33] = r33;
	rf2[i + 33] = r33;
	rf3[i + 33] = r33;

	rf1[i + 34] = r34;
	rf2[i + 34] = r34;
	rf3[i + 34] = r34;

	rf1[i + 35] = r35;
	rf2[i + 35] = r35;
	rf3[i + 35] = r35;

	rf1[i + 36] = r36;
	rf2[i + 36] = r36;
	rf3[i + 36] = r36;

	rf1[i + 37] = r37;
	rf2[i + 37] = r37;
	rf3[i + 37] = r37;

	rf1[i + 38] = r38;
	rf2[i + 38] = r38;
	rf3[i + 38] = r38;

	rf1[i + 39] = r39;
	rf2[i + 39] = r39;
	rf3[i + 39] = r39;

	rf1[i + 40] = r40;
	rf2[i + 40] = r40;
	rf3[i + 40] = r40;

	rf1[i + 41] = r41;
	rf2[i + 41] = r41;
	rf3[i + 41] = r41;

	rf1[i + 42] = r42;
	rf2[i + 42] = r42;
	rf3[i + 42] = r42;

	rf1[i + 43] = r43;
	rf2[i + 43] = r43;
	rf3[i + 43] = r43;

	rf1[i + 44] = r44;
	rf2[i + 44] = r44;
	rf3[i + 44] = r44;

	rf1[i + 45] = r45;
	rf2[i + 45] = r45;
	rf3[i + 45] = r45;

	rf1[i + 46] = r46;
	rf2[i + 46] = r46;
	rf3[i + 46] = r46;

	rf1[i + 47] = r47;
	rf2[i + 47] = r47;
	rf3[i + 47] = r47;

	rf1[i + 48] = r48;
	rf2[i + 48] = r48;
	rf3[i + 48] = r48;

	rf1[i + 49] = r49;
	rf2[i + 49] = r49;
	rf3[i + 49] = r49;

	rf1[i + 50] = r50;
	rf2[i + 50] = r50;
	rf3[i + 50] = r50;

	rf1[i + 51] = r51;
	rf2[i + 51] = r51;
	rf3[i + 51] = r51;

	rf1[i + 52] = r52;
	rf2[i + 52] = r52;
	rf3[i + 52] = r52;

	rf1[i + 53] = r53;
	rf2[i + 53] = r53;
	rf3[i + 53] = r53;

	rf1[i + 54] = r54;
	rf2[i + 54] = r54;
	rf3[i + 54] = r54;

	rf1[i + 55] = r55;
	rf2[i + 55] = r55;
	rf3[i + 55] = r55;

	rf1[i + 56] = r56;
	rf2[i + 56] = r56;
	rf3[i + 56] = r56;

	rf1[i + 57] = r57;
	rf2[i + 57] = r57;
	rf3[i + 57] = r57;

	rf1[i + 58] = r58;
	rf2[i + 58] = r58;
	rf3[i + 58] = r58;

	rf1[i + 59] = r59;
	rf2[i + 59] = r59;
	rf3[i + 59] = r59;

	rf1[i + 60] = r60;
	rf2[i + 60] = r60;
	rf3[i + 60] = r60;

	rf1[i + 61] = r61;
	rf2[i + 61] = r61;
	rf3[i + 61] = r61;

	rf1[i + 62] = r62;
	rf2[i + 62] = r62;
	rf3[i + 62] = r62;

	rf1[i + 63] = r63;
	rf2[i + 63] = r63;
	rf3[i + 63] = r63;

	rf1[i + 64] = r64;
	rf2[i + 64] = r64;
	rf3[i + 64] = r64;

	rf1[i + 65] = r65;
	rf2[i + 65] = r65;
	rf3[i + 65] = r65;

	rf1[i + 66] = r66;
	rf2[i + 66] = r66;
	rf3[i + 66] = r66;

	rf1[i + 67] = r67;
	rf2[i + 67] = r67;
	rf3[i + 67] = r67;

	rf1[i + 68] = r68;
	rf2[i + 68] = r68;
	rf3[i + 68] = r68;

	rf1[i + 69] = r69;
	rf2[i + 69] = r69;
	rf3[i + 69] = r69;

	rf1[i + 70] = r70;
	rf2[i + 70] = r70;
	rf3[i + 70] = r70;

	rf1[i + 71] = r71;
	rf2[i + 71] = r71;
	rf3[i + 71] = r71;

	rf1[i + 72] = r72;
	rf2[i + 72] = r72;
	rf3[i + 72] = r72;

	rf1[i + 73] = r73;
	rf2[i + 73] = r73;
	rf3[i + 73] = r73;

	rf1[i + 74] = r74;
	rf2[i + 74] = r74;
	rf3[i + 74] = r74;

	rf1[i + 75] = r75;
	rf2[i + 75] = r75;
	rf3[i + 75] = r75;

	rf1[i + 76] = r76;
	rf2[i + 76] = r76;
	rf3[i + 76] = r76;

	rf1[i + 77] = r77;
	rf2[i + 77] = r77;
	rf3[i + 77] = r77;

	rf1[i + 78] = r78;
	rf2[i + 78] = r78;
	rf3[i + 78] = r78;

	rf1[i + 79] = r79;
	rf2[i + 79] = r79;
	rf3[i + 79] = r79;

	rf1[i + 80] = r80;
	rf2[i + 80] = r80;
	rf3[i + 80] = r80;

	rf1[i + 81] = r81;
	rf2[i + 81] = r81;
	rf3[i + 81] = r81;

	rf1[i + 82] = r82;
	rf2[i + 82] = r82;
	rf3[i + 82] = r82;

	rf1[i + 83] = r83;
	rf2[i + 83] = r83;
	rf3[i + 83] = r83;

	rf1[i + 84] = r84;
	rf2[i + 84] = r84;
	rf3[i + 84] = r84;

	rf1[i + 85] = r85;
	rf2[i + 85] = r85;
	rf3[i + 85] = r85;

	rf1[i + 86] = r86;
	rf2[i + 86] = r86;
	rf3[i + 86] = r86;

	rf1[i + 87] = r87;
	rf2[i + 87] = r87;
	rf3[i + 87] = r87;

	rf1[i + 88] = r88;
	rf2[i + 88] = r88;
	rf3[i + 88] = r88;

	rf1[i + 89] = r89;
	rf2[i + 89] = r89;
	rf3[i + 89] = r89;

	rf1[i + 90] = r90;
	rf2[i + 90] = r90;
	rf3[i + 90] = r90;

	rf1[i + 91] = r91;
	rf2[i + 91] = r91;
	rf3[i + 91] = r91;

	rf1[i + 92] = r92;
	rf2[i + 92] = r92;
	rf3[i + 92] = r92;

	rf1[i + 93] = r93;
	rf2[i + 93] = r93;
	rf3[i + 93] = r93;

	rf1[i + 94] = r94;
	rf2[i + 94] = r94;
	rf3[i + 94] = r94;

	rf1[i + 95] = r95;
	rf2[i + 95] = r95;
	rf3[i + 95] = r95;

	rf1[i + 96] = r96;
	rf2[i + 96] = r96;
	rf3[i + 96] = r96;

	rf1[i + 97] = r97;
	rf2[i + 97] = r97;
	rf3[i + 97] = r97;

	rf1[i + 98] = r98;
	rf2[i + 98] = r98;
	rf3[i + 98] = r98;

	rf1[i + 99] = r99;
	rf2[i + 99] = r99;
	rf3[i + 99] = r99;

	rf1[i + 100] = r100;
	rf2[i + 100] = r100;
	rf3[i + 100] = r100;

	rf1[i + 101] = r101;
	rf2[i + 101] = r101;
	rf3[i + 101] = r101;

	rf1[i + 102] = r102;
	rf2[i + 102] = r102;
	rf3[i + 102] = r102;

	rf1[i + 103] = r103;
	rf2[i + 103] = r103;
	rf3[i + 103] = r103;

	rf1[i + 104] = r104;
	rf2[i + 104] = r104;
	rf3[i + 104] = r104;

	rf1[i + 105] = r105;
	rf2[i + 105] = r105;
	rf3[i + 105] = r105;

	rf1[i + 106] = r106;
	rf2[i + 106] = r106;
	rf3[i + 106] = r106;

	rf1[i + 107] = r107;
	rf2[i + 107] = r107;
	rf3[i + 107] = r107;

	rf1[i + 108] = r108;
	rf2[i + 108] = r108;
	rf3[i + 108] = r108;

	rf1[i + 109] = r109;
	rf2[i + 109] = r109;
	rf3[i + 109] = r109;

	rf1[i + 110] = r110;
	rf2[i + 110] = r110;
	rf3[i + 110] = r110;

	rf1[i + 111] = r111;
	rf2[i + 111] = r111;
	rf3[i + 111] = r111;

	rf1[i + 112] = r112;
	rf2[i + 112] = r112;
	rf3[i + 112] = r112;

	rf1[i + 113] = r113;
	rf2[i + 113] = r113;
	rf3[i + 113] = r113;

	rf1[i + 114] = r114;
	rf2[i + 114] = r114;
	rf3[i + 114] = r114;

	rf1[i + 115] = r115;
	rf2[i + 115] = r115;
	rf3[i + 115] = r115;

	rf1[i + 116] = r116;
	rf2[i + 116] = r116;
	rf3[i + 116] = r116;

	rf1[i + 117] = r117;
	rf2[i + 117] = r117;
	rf3[i + 117] = r117;

	rf1[i + 118] = r118;
	rf2[i + 118] = r118;
	rf3[i + 118] = r118;

	rf1[i + 119] = r119;
	rf2[i + 119] = r119;
	rf3[i + 119] = r119;

	rf1[i + 120] = r120;
	rf2[i + 120] = r120;
	rf3[i + 120] = r120;

	rf1[i + 121] = r121;
	rf2[i + 121] = r121;
	rf3[i + 121] = r121;

	rf1[i + 122] = r122;
	rf2[i + 122] = r122;
	rf3[i + 122] = r122;

	rf1[i + 123] = r123;
	rf2[i + 123] = r123;
	rf3[i + 123] = r123;

	rf1[i + 124] = r124;
	rf2[i + 124] = r124;
	rf3[i + 124] = r124;

	rf1[i + 125] = r125;
	rf2[i + 125] = r125;
	rf3[i + 125] = r125;

	rf1[i + 126] = r126;
	rf2[i + 126] = r126;
	rf3[i + 126] = r126;

	rf1[i + 127] = r127;
	rf2[i + 127] = r127;
	rf3[i + 127] = r127;

	rf1[i + 128] = r128;
	rf2[i + 128] = r128;
	rf3[i + 128] = r128;

	rf1[i + 129] = r129;
	rf2[i + 129] = r129;
	rf3[i + 129] = r129;

	rf1[i + 130] = r130;
	rf2[i + 130] = r130;
	rf3[i + 130] = r130;

	rf1[i + 131] = r131;
	rf2[i + 131] = r131;
	rf3[i + 131] = r131;

	rf1[i + 132] = r132;
	rf2[i + 132] = r132;
	rf3[i + 132] = r132;

	rf1[i + 133] = r133;
	rf2[i + 133] = r133;
	rf3[i + 133] = r133;

	rf1[i + 134] = r134;
	rf2[i + 134] = r134;
	rf3[i + 134] = r134;

	rf1[i + 135] = r135;
	rf2[i + 135] = r135;
	rf3[i + 135] = r135;

	rf1[i + 136] = r136;
	rf2[i + 136] = r136;
	rf3[i + 136] = r136;

	rf1[i + 137] = r137;
	rf2[i + 137] = r137;
	rf3[i + 137] = r137;

	rf1[i + 138] = r138;
	rf2[i + 138] = r138;
	rf3[i + 138] = r138;

	rf1[i + 139] = r139;
	rf2[i + 139] = r139;
	rf3[i + 139] = r139;

	rf1[i + 140] = r140;
	rf2[i + 140] = r140;
	rf3[i + 140] = r140;

	rf1[i + 141] = r141;
	rf2[i + 141] = r141;
	rf3[i + 141] = r141;

	rf1[i + 142] = r142;
	rf2[i + 142] = r142;
	rf3[i + 142] = r142;

	rf1[i + 143] = r143;
	rf2[i + 143] = r143;
	rf3[i + 143] = r143;

	rf1[i + 144] = r144;
	rf2[i + 144] = r144;
	rf3[i + 144] = r144;

	rf1[i + 145] = r145;
	rf2[i + 145] = r145;
	rf3[i + 145] = r145;

	rf1[i + 146] = r146;
	rf2[i + 146] = r146;
	rf3[i + 146] = r146;

	rf1[i + 147] = r147;
	rf2[i + 147] = r147;
	rf3[i + 147] = r147;

	rf1[i + 148] = r148;
	rf2[i + 148] = r148;
	rf3[i + 148] = r148;

	rf1[i + 149] = r149;
	rf2[i + 149] = r149;
	rf3[i + 149] = r149;

	rf1[i + 150] = r150;
	rf2[i + 150] = r150;
	rf3[i + 150] = r150;

	rf1[i + 151] = r151;
	rf2[i + 151] = r151;
	rf3[i + 151] = r151;

	rf1[i + 152] = r152;
	rf2[i + 152] = r152;
	rf3[i + 152] = r152;

	rf1[i + 153] = r153;
	rf2[i + 153] = r153;
	rf3[i + 153] = r153;

	rf1[i + 154] = r154;
	rf2[i + 154] = r154;
	rf3[i + 154] = r154;

	rf1[i + 155] = r155;
	rf2[i + 155] = r155;
	rf3[i + 155] = r155;

	rf1[i + 156] = r156;
	rf2[i + 156] = r156;
	rf3[i + 156] = r156;

	rf1[i + 157] = r157;
	rf2[i + 157] = r157;
	rf3[i + 157] = r157;

	rf1[i + 158] = r158;
	rf2[i + 158] = r158;
	rf3[i + 158] = r158;

	rf1[i + 159] = r159;
	rf2[i + 159] = r159;
	rf3[i + 159] = r159;

	rf1[i + 160] = r160;
	rf2[i + 160] = r160;
	rf3[i + 160] = r160;

	rf1[i + 161] = r161;
	rf2[i + 161] = r161;
	rf3[i + 161] = r161;

	rf1[i + 162] = r162;
	rf2[i + 162] = r162;
	rf3[i + 162] = r162;

	rf1[i + 163] = r163;
	rf2[i + 163] = r163;
	rf3[i + 163] = r163;

	rf1[i + 164] = r164;
	rf2[i + 164] = r164;
	rf3[i + 164] = r164;

	rf1[i + 165] = r165;
	rf2[i + 165] = r165;
	rf3[i + 165] = r165;

	rf1[i + 166] = r166;
	rf2[i + 166] = r166;
	rf3[i + 166] = r166;

	rf1[i + 167] = r167;
	rf2[i + 167] = r167;
	rf3[i + 167] = r167;

	rf1[i + 168] = r168;
	rf2[i + 168] = r168;
	rf3[i + 168] = r168;

	rf1[i + 169] = r169;
	rf2[i + 169] = r169;
	rf3[i + 169] = r169;

	rf1[i + 170] = r170;
	rf2[i + 170] = r170;
	rf3[i + 170] = r170;

	rf1[i + 171] = r171;
	rf2[i + 171] = r171;
	rf3[i + 171] = r171;

	rf1[i + 172] = r172;
	rf2[i + 172] = r172;
	rf3[i + 172] = r172;

	rf1[i + 173] = r173;
	rf2[i + 173] = r173;
	rf3[i + 173] = r173;

	rf1[i + 174] = r174;
	rf2[i + 174] = r174;
	rf3[i + 174] = r174;

	rf1[i + 175] = r175;
	rf2[i + 175] = r175;
	rf3[i + 175] = r175;

	rf1[i + 176] = r176;
	rf2[i + 176] = r176;
	rf3[i + 176] = r176;

	rf1[i + 177] = r177;
	rf2[i + 177] = r177;
	rf3[i + 177] = r177;

	rf1[i + 178] = r178;
	rf2[i + 178] = r178;
	rf3[i + 178] = r178;

	rf1[i + 179] = r179;
	rf2[i + 179] = r179;
	rf3[i + 179] = r179;

	rf1[i + 180] = r180;
	rf2[i + 180] = r180;
	rf3[i + 180] = r180;

	rf1[i + 181] = r181;
	rf2[i + 181] = r181;
	rf3[i + 181] = r181;

	rf1[i + 182] = r182;
	rf2[i + 182] = r182;
	rf3[i + 182] = r182;

	rf1[i + 183] = r183;
	rf2[i + 183] = r183;
	rf3[i + 183] = r183;

	rf1[i + 184] = r184;
	rf2[i + 184] = r184;
	rf3[i + 184] = r184;

	rf1[i + 185] = r185;
	rf2[i + 185] = r185;
	rf3[i + 185] = r185;

	rf1[i + 186] = r186;
	rf2[i + 186] = r186;
	rf3[i + 186] = r186;

	rf1[i + 187] = r187;
	rf2[i + 187] = r187;
	rf3[i + 187] = r187;

	rf1[i + 188] = r188;
	rf2[i + 188] = r188;
	rf3[i + 188] = r188;

	rf1[i + 189] = r189;
	rf2[i + 189] = r189;
	rf3[i + 189] = r189;

	rf1[i + 190] = r190;
	rf2[i + 190] = r190;
	rf3[i + 190] = r190;

	rf1[i + 191] = r191;
	rf2[i + 191] = r191;
	rf3[i + 191] = r191;

	rf1[i + 192] = r192;
	rf2[i + 192] = r192;
	rf3[i + 192] = r192;

	rf1[i + 193] = r193;
	rf2[i + 193] = r193;
	rf3[i + 193] = r193;

	rf1[i + 194] = r194;
	rf2[i + 194] = r194;
	rf3[i + 194] = r194;

	rf1[i + 195] = r195;
	rf2[i + 195] = r195;
	rf3[i + 195] = r195;

	rf1[i + 196] = r196;
	rf2[i + 196] = r196;
	rf3[i + 196] = r196;

	rf1[i + 197] = r197;
	rf2[i + 197] = r197;
	rf3[i + 197] = r197;

	rf1[i + 198] = r198;
	rf2[i + 198] = r198;
	rf3[i + 198] = r198;

	rf1[i + 199] = r199;
	rf2[i + 199] = r199;
	rf3[i + 199] = r199;

	rf1[i + 200] = r200;
	rf2[i + 200] = r200;
	rf3[i + 200] = r200;

	rf1[i + 201] = r201;
	rf2[i + 201] = r201;
	rf3[i + 201] = r201;

	rf1[i + 202] = r202;
	rf2[i + 202] = r202;
	rf3[i + 202] = r202;

	rf1[i + 203] = r203;
	rf2[i + 203] = r203;
	rf3[i + 203] = r203;

	rf1[i + 204] = r204;
	rf2[i + 204] = r204;
	rf3[i + 204] = r204;

	rf1[i + 205] = r205;
	rf2[i + 205] = r205;
	rf3[i + 205] = r205;

	rf1[i + 206] = r206;
	rf2[i + 206] = r206;
	rf3[i + 206] = r206;

	rf1[i + 207] = r207;
	rf2[i + 207] = r207;
	rf3[i + 207] = r207;

	rf1[i + 208] = r208;
	rf2[i + 208] = r208;
	rf3[i + 208] = r208;

	rf1[i + 209] = r209;
	rf2[i + 209] = r209;
	rf3[i + 209] = r209;

	rf1[i + 210] = r210;
	rf2[i + 210] = r210;
	rf3[i + 210] = r210;

	rf1[i + 211] = r211;
	rf2[i + 211] = r211;
	rf3[i + 211] = r211;

	rf1[i + 212] = r212;
	rf2[i + 212] = r212;
	rf3[i + 212] = r212;

	rf1[i + 213] = r213;
	rf2[i + 213] = r213;
	rf3[i + 213] = r213;

	rf1[i + 214] = r214;
	rf2[i + 214] = r214;
	rf3[i + 214] = r214;

	rf1[i + 215] = r215;
	rf2[i + 215] = r215;
	rf3[i + 215] = r215;

	rf1[i + 216] = r216;
	rf2[i + 216] = r216;
	rf3[i + 216] = r216;

	rf1[i + 217] = r217;
	rf2[i + 217] = r217;
	rf3[i + 217] = r217;

	rf1[i + 218] = r218;
	rf2[i + 218] = r218;
	rf3[i + 218] = r218;

	rf1[i + 219] = r219;
	rf2[i + 219] = r219;
	rf3[i + 219] = r219;

	rf1[i + 220] = r220;
	rf2[i + 220] = r220;
	rf3[i + 220] = r220;

	rf1[i + 221] = r221;
	rf2[i + 221] = r221;
	rf3[i + 221] = r221;

	rf1[i + 222] = r222;
	rf2[i + 222] = r222;
	rf3[i + 222] = r222;

	rf1[i + 223] = r223;
	rf2[i + 223] = r223;
	rf3[i + 223] = r223;

	rf1[i + 224] = r224;
	rf2[i + 224] = r224;
	rf3[i + 224] = r224;

	rf1[i + 225] = r225;
	rf2[i + 225] = r225;
	rf3[i + 225] = r225;

	rf1[i + 226] = r226;
	rf2[i + 226] = r226;
	rf3[i + 226] = r226;

	rf1[i + 227] = r227;
	rf2[i + 227] = r227;
	rf3[i + 227] = r227;

	rf1[i + 228] = r228;
	rf2[i + 228] = r228;
	rf3[i + 228] = r228;

	rf1[i + 229] = r229;
	rf2[i + 229] = r229;
	rf3[i + 229] = r229;

	rf1[i + 230] = r230;
	rf2[i + 230] = r230;
	rf3[i + 230] = r230;

	rf1[i + 231] = r231;
	rf2[i + 231] = r231;
	rf3[i + 231] = r231;

	rf1[i + 232] = r232;
	rf2[i + 232] = r232;
	rf3[i + 232] = r232;

	rf1[i + 233] = r233;
	rf2[i + 233] = r233;
	rf3[i + 233] = r233;

	rf1[i + 234] = r234;
	rf2[i + 234] = r234;
	rf3[i + 234] = r234;

	rf1[i + 235] = r235;
	rf2[i + 235] = r235;
	rf3[i + 235] = r235;

	rf1[i + 236] = r236;
	rf2[i + 236] = r236;
	rf3[i + 236] = r236;

	rf1[i + 237] = r237;
	rf2[i + 237] = r237;
	rf3[i + 237] = r237;

	rf1[i + 238] = r238;
	rf2[i + 238] = r238;
	rf3[i + 238] = r238;

	rf1[i + 239] = r239;
	rf2[i + 239] = r239;
	rf3[i + 239] = r239;

	rf1[i + 240] = r240;
	rf2[i + 240] = r240;
	rf3[i + 240] = r240;

	rf1[i + 241] = r241;
	rf2[i + 241] = r241;
	rf3[i + 241] = r241;

	rf1[i + 242] = r242;
	rf2[i + 242] = r242;
	rf3[i + 242] = r242;

	rf1[i + 243] = r243;
	rf2[i + 243] = r243;
	rf3[i + 243] = r243;

	rf1[i + 244] = r244;
	rf2[i + 244] = r244;
	rf3[i + 244] = r244;

	rf1[i + 245] = r245;
	rf2[i + 245] = r245;
	rf3[i + 245] = r245;

	rf1[i + 246] = r246;
	rf2[i + 246] = r246;
	rf3[i + 246] = r246;

	rf1[i + 247] = r247;
	rf2[i + 247] = r247;
	rf3[i + 247] = r247;

	rf1[i + 248] = r248;
	rf2[i + 248] = r248;
	rf3[i + 248] = r248;

	rf1[i + 249] = r249;
	rf2[i + 249] = r249;
	rf3[i + 249] = r249;

	rf1[i + 250] = r250;
	rf2[i + 250] = r250;
	rf3[i + 250] = r250;

	rf1[i + 251] = r251;
	rf2[i + 251] = r251;
	rf3[i + 251] = r251;

	rf1[i + 252] = r252;
	rf2[i + 252] = r252;
	rf3[i + 252] = r252;

	rf1[i + 253] = r253;
	rf2[i + 253] = r253;
	rf3[i + 253] = r253;

	rf1[i + 254] = r254;
	rf2[i + 254] = r254;
	rf3[i + 254] = r254;

	rf1[i + 255] = r255;
	rf2[i + 255] = r255;
	rf3[i + 255] = r255;

}

#endif /* REGISTER_KERNEL_VOLTA_H_ */
