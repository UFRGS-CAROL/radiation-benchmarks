/*
 * l1_move_function.h
 *
 *  Created on: 20/09/2019
 *      Author: fernando
 */

#ifndef L1_MOVE_FUNCTION_H_
#define L1_MOVE_FUNCTION_H_

__device__ __forceinline__ void mov_cache_data(volatile uint64* dst, volatile uint64* src) {
	dst[0] = src[0];
	dst[1] = src[1];
	dst[2] = src[2];
	dst[3] = src[3];
	dst[4] = src[4];
	dst[5] = src[5];
	dst[6] = src[6];
	dst[7] = src[7];
	dst[8] = src[8];
	dst[9] = src[9];
	dst[10] = src[10];
	dst[11] = src[11];
	dst[12] = src[12];
	dst[13] = src[13];
	dst[14] = src[14];
	dst[15] = src[15];
	dst[16] = src[16];
	dst[17] = src[17];
	dst[18] = src[18];
	dst[19] = src[19];
	dst[20] = src[20];
	dst[21] = src[21];
	dst[22] = src[22];
	dst[23] = src[23];
	dst[24] = src[24];
	dst[25] = src[25];
	dst[26] = src[26];
	dst[27] = src[27];
	dst[28] = src[28];
	dst[29] = src[29];
	dst[30] = src[30];
	dst[31] = src[31];
	dst[32] = src[32];
	dst[33] = src[33];
	dst[34] = src[34];
	dst[35] = src[35];
	dst[36] = src[36];
	dst[37] = src[37];
	dst[38] = src[38];
	dst[39] = src[39];
	dst[40] = src[40];
	dst[41] = src[41];
	dst[42] = src[42];
	dst[43] = src[43];
	dst[44] = src[44];
	dst[45] = src[45];
	dst[46] = src[46];
	dst[47] = src[47];
	dst[48] = src[48];
	dst[49] = src[49];
	dst[50] = src[50];
	dst[51] = src[51];
	dst[52] = src[52];
	dst[53] = src[53];
	dst[54] = src[54];
	dst[55] = src[55];
	dst[56] = src[56];
	dst[57] = src[57];
	dst[58] = src[58];
	dst[59] = src[59];
	dst[60] = src[60];
	dst[61] = src[61];
	dst[62] = src[62];
	dst[63] = src[63];
	dst[64] = src[64];
	dst[65] = src[65];
	dst[66] = src[66];
	dst[67] = src[67];
	dst[68] = src[68];
	dst[69] = src[69];
	dst[70] = src[70];
	dst[71] = src[71];
	dst[72] = src[72];
	dst[73] = src[73];
	dst[74] = src[74];
	dst[75] = src[75];
	dst[76] = src[76];
	dst[77] = src[77];
	dst[78] = src[78];
	dst[79] = src[79];
	dst[80] = src[80];
	dst[81] = src[81];
	dst[82] = src[82];
	dst[83] = src[83];
	dst[84] = src[84];
	dst[85] = src[85];
	dst[86] = src[86];
	dst[87] = src[87];
	dst[88] = src[88];
	dst[89] = src[89];
	dst[90] = src[90];
	dst[91] = src[91];
	dst[92] = src[92];
	dst[93] = src[93];
	dst[94] = src[94];
	dst[95] = src[95];
	dst[96] = src[96];
	dst[97] = src[97];
	dst[98] = src[98];
	dst[99] = src[99];
	dst[100] = src[100];
	dst[101] = src[101];
	dst[102] = src[102];
	dst[103] = src[103];
	dst[104] = src[104];
	dst[105] = src[105];
	dst[106] = src[106];
	dst[107] = src[107];
	dst[108] = src[108];
	dst[109] = src[109];
	dst[110] = src[110];
	dst[111] = src[111];
	dst[112] = src[112];
	dst[113] = src[113];
	dst[114] = src[114];
	dst[115] = src[115];
	dst[116] = src[116];
	dst[117] = src[117];
	dst[118] = src[118];
	dst[119] = src[119];
	dst[120] = src[120];
	dst[121] = src[121];
	dst[122] = src[122];
	dst[123] = src[123];
	dst[124] = src[124];
	dst[125] = src[125];
	dst[126] = src[126];
	dst[127] = src[127];
}


#endif /* L1_MOVE_FUNCTION_H_ */
