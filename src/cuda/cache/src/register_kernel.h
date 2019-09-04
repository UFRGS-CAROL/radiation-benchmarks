#ifndef REGISTER_KERNEL_H_
#define REGISTER_KERNEL_H_

#include "utils.h"

__global__ void test_register_file_kernel_or(uint32 *rf1, uint32 *rf2,
		uint32 *rf3, uint32 *mem1, uint32 *mem2, uint32 *mem3, uint32 reg_data,
		const uint64 sleep_cycles) {
	const uint32 i = blockDim.x * blockIdx.x + threadIdx.x;
	register uint32 r0 = reg_data | mem1[0] | mem2[0] | mem3[0];
	register uint32 r1 = reg_data | mem1[1] | mem2[1] | mem3[1];
	register uint32 r2 = reg_data | mem1[2] | mem2[2] | mem3[2];
	register uint32 r3 = reg_data | mem1[3] | mem2[3] | mem3[3];
	register uint32 r4 = reg_data | mem1[4] | mem2[4] | mem3[4];
	register uint32 r5 = reg_data | mem1[5] | mem2[5] | mem3[5];
	register uint32 r6 = reg_data | mem1[6] | mem2[6] | mem3[6];
	register uint32 r7 = reg_data | mem1[7] | mem2[7] | mem3[7];
	register uint32 r8 = reg_data | mem1[8] | mem2[8] | mem3[8];
	register uint32 r9 = reg_data | mem1[9] | mem2[9] | mem3[9];
	register uint32 r10 = reg_data | mem1[10] | mem2[10] | mem3[10];
	register uint32 r11 = reg_data | mem1[11] | mem2[11] | mem3[11];
	register uint32 r12 = reg_data | mem1[12] | mem2[12] | mem3[12];
	register uint32 r13 = reg_data | mem1[13] | mem2[13] | mem3[13];
	register uint32 r14 = reg_data | mem1[14] | mem2[14] | mem3[14];
	register uint32 r15 = reg_data | mem1[15] | mem2[15] | mem3[15];
	register uint32 r16 = reg_data | mem1[16] | mem2[16] | mem3[16];
	register uint32 r17 = reg_data | mem1[17] | mem2[17] | mem3[17];
	register uint32 r18 = reg_data | mem1[18] | mem2[18] | mem3[18];
	register uint32 r19 = reg_data | mem1[19] | mem2[19] | mem3[19];
	register uint32 r20 = reg_data | mem1[20] | mem2[20] | mem3[20];
	register uint32 r21 = reg_data | mem1[21] | mem2[21] | mem3[21];
	register uint32 r22 = reg_data | mem1[22] | mem2[22] | mem3[22];
	register uint32 r23 = reg_data | mem1[23] | mem2[23] | mem3[23];
	register uint32 r24 = reg_data | mem1[24] | mem2[24] | mem3[24];
	register uint32 r25 = reg_data | mem1[25] | mem2[25] | mem3[25];
	register uint32 r26 = reg_data | mem1[26] | mem2[26] | mem3[26];
	register uint32 r27 = reg_data | mem1[27] | mem2[27] | mem3[27];
	register uint32 r28 = reg_data | mem1[28] | mem2[28] | mem3[28];
	register uint32 r29 = reg_data | mem1[29] | mem2[29] | mem3[29];
	register uint32 r30 = reg_data | mem1[30] | mem2[30] | mem3[30];
	register uint32 r31 = reg_data | mem1[31] | mem2[31] | mem3[31];
	register uint32 r32 = reg_data | mem1[32] | mem2[32] | mem3[32];
	register uint32 r33 = reg_data | mem1[33] | mem2[33] | mem3[33];
	register uint32 r34 = reg_data | mem1[34] | mem2[34] | mem3[34];
	register uint32 r35 = reg_data | mem1[35] | mem2[35] | mem3[35];
	register uint32 r36 = reg_data | mem1[36] | mem2[36] | mem3[36];
	register uint32 r37 = reg_data | mem1[37] | mem2[37] | mem3[37];
	register uint32 r38 = reg_data | mem1[38] | mem2[38] | mem3[38];
	register uint32 r39 = reg_data | mem1[39] | mem2[39] | mem3[39];
	register uint32 r40 = reg_data | mem1[40] | mem2[40] | mem3[40];
	register uint32 r41 = reg_data | mem1[41] | mem2[41] | mem3[41];
	register uint32 r42 = reg_data | mem1[42] | mem2[42] | mem3[42];
	register uint32 r43 = reg_data | mem1[43] | mem2[43] | mem3[43];
	register uint32 r44 = reg_data | mem1[44] | mem2[44] | mem3[44];
	register uint32 r45 = reg_data | mem1[45] | mem2[45] | mem3[45];
	register uint32 r46 = reg_data | mem1[46] | mem2[46] | mem3[46];
	register uint32 r47 = reg_data | mem1[47] | mem2[47] | mem3[47];
	register uint32 r48 = reg_data | mem1[48] | mem2[48] | mem3[48];
	register uint32 r49 = reg_data | mem1[49] | mem2[49] | mem3[49];
	register uint32 r50 = reg_data | mem1[50] | mem2[50] | mem3[50];
	register uint32 r51 = reg_data | mem1[51] | mem2[51] | mem3[51];
	register uint32 r52 = reg_data | mem1[52] | mem2[52] | mem3[52];
	register uint32 r53 = reg_data | mem1[53] | mem2[53] | mem3[53];
	register uint32 r54 = reg_data | mem1[54] | mem2[54] | mem3[54];
	register uint32 r55 = reg_data | mem1[55] | mem2[55] | mem3[55];
	register uint32 r56 = reg_data | mem1[56] | mem2[56] | mem3[56];
	register uint32 r57 = reg_data | mem1[57] | mem2[57] | mem3[57];
	register uint32 r58 = reg_data | mem1[58] | mem2[58] | mem3[58];
	register uint32 r59 = reg_data | mem1[59] | mem2[59] | mem3[59];
	register uint32 r60 = reg_data | mem1[60] | mem2[60] | mem3[60];
	register uint32 r61 = reg_data | mem1[61] | mem2[61] | mem3[61];
	register uint32 r62 = reg_data | mem1[62] | mem2[62] | mem3[62];
	register uint32 r63 = reg_data | mem1[63] | mem2[63] | mem3[63];
	register uint32 r64 = reg_data | mem1[64] | mem2[64] | mem3[64];
	register uint32 r65 = reg_data | mem1[65] | mem2[65] | mem3[65];
	register uint32 r66 = reg_data | mem1[66] | mem2[66] | mem3[66];
	register uint32 r67 = reg_data | mem1[67] | mem2[67] | mem3[67];
	register uint32 r68 = reg_data | mem1[68] | mem2[68] | mem3[68];
	register uint32 r69 = reg_data | mem1[69] | mem2[69] | mem3[69];
	register uint32 r70 = reg_data | mem1[70] | mem2[70] | mem3[70];
	register uint32 r71 = reg_data | mem1[71] | mem2[71] | mem3[71];
	register uint32 r72 = reg_data | mem1[72] | mem2[72] | mem3[72];
	register uint32 r73 = reg_data | mem1[73] | mem2[73] | mem3[73];
	register uint32 r74 = reg_data | mem1[74] | mem2[74] | mem3[74];
	register uint32 r75 = reg_data | mem1[75] | mem2[75] | mem3[75];
	register uint32 r76 = reg_data | mem1[76] | mem2[76] | mem3[76];
	register uint32 r77 = reg_data | mem1[77] | mem2[77] | mem3[77];
	register uint32 r78 = reg_data | mem1[78] | mem2[78] | mem3[78];
	register uint32 r79 = reg_data | mem1[79] | mem2[79] | mem3[79];
	register uint32 r80 = reg_data | mem1[80] | mem2[80] | mem3[80];
	register uint32 r81 = reg_data | mem1[81] | mem2[81] | mem3[81];
	register uint32 r82 = reg_data | mem1[82] | mem2[82] | mem3[82];
	register uint32 r83 = reg_data | mem1[83] | mem2[83] | mem3[83];
	register uint32 r84 = reg_data | mem1[84] | mem2[84] | mem3[84];
	register uint32 r85 = reg_data | mem1[85] | mem2[85] | mem3[85];
	register uint32 r86 = reg_data | mem1[86] | mem2[86] | mem3[86];
	register uint32 r87 = reg_data | mem1[87] | mem2[87] | mem3[87];
	register uint32 r88 = reg_data | mem1[88] | mem2[88] | mem3[88];
	register uint32 r89 = reg_data | mem1[89] | mem2[89] | mem3[89];
	register uint32 r90 = reg_data | mem1[90] | mem2[90] | mem3[90];
	register uint32 r91 = reg_data | mem1[91] | mem2[91] | mem3[91];
	register uint32 r92 = reg_data | mem1[92] | mem2[92] | mem3[92];
	register uint32 r93 = reg_data | mem1[93] | mem2[93] | mem3[93];
	register uint32 r94 = reg_data | mem1[94] | mem2[94] | mem3[94];
	register uint32 r95 = reg_data | mem1[95] | mem2[95] | mem3[95];
	register uint32 r96 = reg_data | mem1[96] | mem2[96] | mem3[96];
	register uint32 r97 = reg_data | mem1[97] | mem2[97] | mem3[97];
	register uint32 r98 = reg_data | mem1[98] | mem2[98] | mem3[98];
	register uint32 r99 = reg_data | mem1[99] | mem2[99] | mem3[99];
	register uint32 r100 = reg_data | mem1[100] | mem2[100] | mem3[100];
	register uint32 r101 = reg_data | mem1[101] | mem2[101] | mem3[101];
	register uint32 r102 = reg_data | mem1[102] | mem2[102] | mem3[102];
	register uint32 r103 = reg_data | mem1[103] | mem2[103] | mem3[103];
	register uint32 r104 = reg_data | mem1[104] | mem2[104] | mem3[104];
	register uint32 r105 = reg_data | mem1[105] | mem2[105] | mem3[105];
	register uint32 r106 = reg_data | mem1[106] | mem2[106] | mem3[106];
	register uint32 r107 = reg_data | mem1[107] | mem2[107] | mem3[107];
	register uint32 r108 = reg_data | mem1[108] | mem2[108] | mem3[108];
	register uint32 r109 = reg_data | mem1[109] | mem2[109] | mem3[109];
	register uint32 r110 = reg_data | mem1[110] | mem2[110] | mem3[110];
	register uint32 r111 = reg_data | mem1[111] | mem2[111] | mem3[111];
	register uint32 r112 = reg_data | mem1[112] | mem2[112] | mem3[112];
	register uint32 r113 = reg_data | mem1[113] | mem2[113] | mem3[113];
	register uint32 r114 = reg_data | mem1[114] | mem2[114] | mem3[114];
	register uint32 r115 = reg_data | mem1[115] | mem2[115] | mem3[115];
	register uint32 r116 = reg_data | mem1[116] | mem2[116] | mem3[116];
	register uint32 r117 = reg_data | mem1[117] | mem2[117] | mem3[117];
	register uint32 r118 = reg_data | mem1[118] | mem2[118] | mem3[118];
	register uint32 r119 = reg_data | mem1[119] | mem2[119] | mem3[119];
	register uint32 r120 = reg_data | mem1[120] | mem2[120] | mem3[120];
	register uint32 r121 = reg_data | mem1[121] | mem2[121] | mem3[121];
	register uint32 r122 = reg_data | mem1[122] | mem2[122] | mem3[122];
	register uint32 r123 = reg_data | mem1[123] | mem2[123] | mem3[123];
	register uint32 r124 = reg_data | mem1[124] | mem2[124] | mem3[124];
	register uint32 r125 = reg_data | mem1[125] | mem2[125] | mem3[125];
	register uint32 r126 = reg_data | mem1[126] | mem2[126] | mem3[126];
	register uint32 r127 = reg_data | mem1[127] | mem2[127] | mem3[127];
	register uint32 r128 = reg_data | mem1[128] | mem2[128] | mem3[128];
	register uint32 r129 = reg_data | mem1[129] | mem2[129] | mem3[129];
	register uint32 r130 = reg_data | mem1[130] | mem2[130] | mem3[130];
	register uint32 r131 = reg_data | mem1[131] | mem2[131] | mem3[131];
	register uint32 r132 = reg_data | mem1[132] | mem2[132] | mem3[132];
	register uint32 r133 = reg_data | mem1[133] | mem2[133] | mem3[133];
	register uint32 r134 = reg_data | mem1[134] | mem2[134] | mem3[134];
	register uint32 r135 = reg_data | mem1[135] | mem2[135] | mem3[135];
	register uint32 r136 = reg_data | mem1[136] | mem2[136] | mem3[136];
	register uint32 r137 = reg_data | mem1[137] | mem2[137] | mem3[137];
	register uint32 r138 = reg_data | mem1[138] | mem2[138] | mem3[138];
	register uint32 r139 = reg_data | mem1[139] | mem2[139] | mem3[139];
	register uint32 r140 = reg_data | mem1[140] | mem2[140] | mem3[140];
	register uint32 r141 = reg_data | mem1[141] | mem2[141] | mem3[141];
	register uint32 r142 = reg_data | mem1[142] | mem2[142] | mem3[142];
	register uint32 r143 = reg_data | mem1[143] | mem2[143] | mem3[143];
	register uint32 r144 = reg_data | mem1[144] | mem2[144] | mem3[144];
	register uint32 r145 = reg_data | mem1[145] | mem2[145] | mem3[145];
	register uint32 r146 = reg_data | mem1[146] | mem2[146] | mem3[146];
	register uint32 r147 = reg_data | mem1[147] | mem2[147] | mem3[147];
	register uint32 r148 = reg_data | mem1[148] | mem2[148] | mem3[148];
	register uint32 r149 = reg_data | mem1[149] | mem2[149] | mem3[149];
	register uint32 r150 = reg_data | mem1[150] | mem2[150] | mem3[150];
	register uint32 r151 = reg_data | mem1[151] | mem2[151] | mem3[151];
	register uint32 r152 = reg_data | mem1[152] | mem2[152] | mem3[152];
	register uint32 r153 = reg_data | mem1[153] | mem2[153] | mem3[153];
	register uint32 r154 = reg_data | mem1[154] | mem2[154] | mem3[154];
	register uint32 r155 = reg_data | mem1[155] | mem2[155] | mem3[155];
	register uint32 r156 = reg_data | mem1[156] | mem2[156] | mem3[156];
	register uint32 r157 = reg_data | mem1[157] | mem2[157] | mem3[157];
	register uint32 r158 = reg_data | mem1[158] | mem2[158] | mem3[158];
	register uint32 r159 = reg_data | mem1[159] | mem2[159] | mem3[159];
	register uint32 r160 = reg_data | mem1[160] | mem2[160] | mem3[160];
	register uint32 r161 = reg_data | mem1[161] | mem2[161] | mem3[161];
	register uint32 r162 = reg_data | mem1[162] | mem2[162] | mem3[162];
	register uint32 r163 = reg_data | mem1[163] | mem2[163] | mem3[163];
	register uint32 r164 = reg_data | mem1[164] | mem2[164] | mem3[164];
	register uint32 r165 = reg_data | mem1[165] | mem2[165] | mem3[165];
	register uint32 r166 = reg_data | mem1[166] | mem2[166] | mem3[166];
	register uint32 r167 = reg_data | mem1[167] | mem2[167] | mem3[167];
	register uint32 r168 = reg_data | mem1[168] | mem2[168] | mem3[168];
	register uint32 r169 = reg_data | mem1[169] | mem2[169] | mem3[169];
	register uint32 r170 = reg_data | mem1[170] | mem2[170] | mem3[170];
	register uint32 r171 = reg_data | mem1[171] | mem2[171] | mem3[171];
	register uint32 r172 = reg_data | mem1[172] | mem2[172] | mem3[172];
	register uint32 r173 = reg_data | mem1[173] | mem2[173] | mem3[173];
	register uint32 r174 = reg_data | mem1[174] | mem2[174] | mem3[174];
	register uint32 r175 = reg_data | mem1[175] | mem2[175] | mem3[175];
	register uint32 r176 = reg_data | mem1[176] | mem2[176] | mem3[176];
	register uint32 r177 = reg_data | mem1[177] | mem2[177] | mem3[177];
	register uint32 r178 = reg_data | mem1[178] | mem2[178] | mem3[178];
	register uint32 r179 = reg_data | mem1[179] | mem2[179] | mem3[179];
	register uint32 r180 = reg_data | mem1[180] | mem2[180] | mem3[180];
	register uint32 r181 = reg_data | mem1[181] | mem2[181] | mem3[181];
	register uint32 r182 = reg_data | mem1[182] | mem2[182] | mem3[182];
	register uint32 r183 = reg_data | mem1[183] | mem2[183] | mem3[183];
	register uint32 r184 = reg_data | mem1[184] | mem2[184] | mem3[184];
	register uint32 r185 = reg_data | mem1[185] | mem2[185] | mem3[185];

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

	rf1[i + 186] = reg_data;
	rf2[i + 186] = reg_data;
	rf3[i + 186] = reg_data;

	rf1[i + 187] = reg_data;
	rf2[i + 187] = reg_data;
	rf3[i + 187] = reg_data;

	rf1[i + 188] = reg_data;
	rf2[i + 188] = reg_data;
	rf3[i + 188] = reg_data;

	rf1[i + 189] = reg_data;
	rf2[i + 189] = reg_data;
	rf3[i + 189] = reg_data;

	rf1[i + 190] = reg_data;
	rf2[i + 190] = reg_data;
	rf3[i + 190] = reg_data;

	rf1[i + 191] = reg_data;
	rf2[i + 191] = reg_data;
	rf3[i + 191] = reg_data;

	rf1[i + 192] = reg_data;
	rf2[i + 192] = reg_data;
	rf3[i + 192] = reg_data;

	rf1[i + 193] = reg_data;
	rf2[i + 193] = reg_data;
	rf3[i + 193] = reg_data;

	rf1[i + 194] = reg_data;
	rf2[i + 194] = reg_data;
	rf3[i + 194] = reg_data;

	rf1[i + 195] = reg_data;
	rf2[i + 195] = reg_data;
	rf3[i + 195] = reg_data;

	rf1[i + 196] = reg_data;
	rf2[i + 196] = reg_data;
	rf3[i + 196] = reg_data;

	rf1[i + 197] = reg_data;
	rf2[i + 197] = reg_data;
	rf3[i + 197] = reg_data;

	rf1[i + 198] = reg_data;
	rf2[i + 198] = reg_data;
	rf3[i + 198] = reg_data;

	rf1[i + 199] = reg_data;
	rf2[i + 199] = reg_data;
	rf3[i + 199] = reg_data;

	rf1[i + 200] = reg_data;
	rf2[i + 200] = reg_data;
	rf3[i + 200] = reg_data;

	rf1[i + 201] = reg_data;
	rf2[i + 201] = reg_data;
	rf3[i + 201] = reg_data;

	rf1[i + 202] = reg_data;
	rf2[i + 202] = reg_data;
	rf3[i + 202] = reg_data;

	rf1[i + 203] = reg_data;
	rf2[i + 203] = reg_data;
	rf3[i + 203] = reg_data;

	rf1[i + 204] = reg_data;
	rf2[i + 204] = reg_data;
	rf3[i + 204] = reg_data;

	rf1[i + 205] = reg_data;
	rf2[i + 205] = reg_data;
	rf3[i + 205] = reg_data;

	rf1[i + 206] = reg_data;
	rf2[i + 206] = reg_data;
	rf3[i + 206] = reg_data;

	rf1[i + 207] = reg_data;
	rf2[i + 207] = reg_data;
	rf3[i + 207] = reg_data;

	rf1[i + 208] = reg_data;
	rf2[i + 208] = reg_data;
	rf3[i + 208] = reg_data;

	rf1[i + 209] = reg_data;
	rf2[i + 209] = reg_data;
	rf3[i + 209] = reg_data;

	rf1[i + 210] = reg_data;
	rf2[i + 210] = reg_data;
	rf3[i + 210] = reg_data;

	rf1[i + 211] = reg_data;
	rf2[i + 211] = reg_data;
	rf3[i + 211] = reg_data;

	rf1[i + 212] = reg_data;
	rf2[i + 212] = reg_data;
	rf3[i + 212] = reg_data;

	rf1[i + 213] = reg_data;
	rf2[i + 213] = reg_data;
	rf3[i + 213] = reg_data;

	rf1[i + 214] = reg_data;
	rf2[i + 214] = reg_data;
	rf3[i + 214] = reg_data;

	rf1[i + 215] = reg_data;
	rf2[i + 215] = reg_data;
	rf3[i + 215] = reg_data;

	rf1[i + 216] = reg_data;
	rf2[i + 216] = reg_data;
	rf3[i + 216] = reg_data;

	rf1[i + 217] = reg_data;
	rf2[i + 217] = reg_data;
	rf3[i + 217] = reg_data;

	rf1[i + 218] = reg_data;
	rf2[i + 218] = reg_data;
	rf3[i + 218] = reg_data;

	rf1[i + 219] = reg_data;
	rf2[i + 219] = reg_data;
	rf3[i + 219] = reg_data;

	rf1[i + 220] = reg_data;
	rf2[i + 220] = reg_data;
	rf3[i + 220] = reg_data;

	rf1[i + 221] = reg_data;
	rf2[i + 221] = reg_data;
	rf3[i + 221] = reg_data;

	rf1[i + 222] = reg_data;
	rf2[i + 222] = reg_data;
	rf3[i + 222] = reg_data;

	rf1[i + 223] = reg_data;
	rf2[i + 223] = reg_data;
	rf3[i + 223] = reg_data;

	rf1[i + 224] = reg_data;
	rf2[i + 224] = reg_data;
	rf3[i + 224] = reg_data;

	rf1[i + 225] = reg_data;
	rf2[i + 225] = reg_data;
	rf3[i + 225] = reg_data;

	rf1[i + 226] = reg_data;
	rf2[i + 226] = reg_data;
	rf3[i + 226] = reg_data;

	rf1[i + 227] = reg_data;
	rf2[i + 227] = reg_data;
	rf3[i + 227] = reg_data;

	rf1[i + 228] = reg_data;
	rf2[i + 228] = reg_data;
	rf3[i + 228] = reg_data;

	rf1[i + 229] = reg_data;
	rf2[i + 229] = reg_data;
	rf3[i + 229] = reg_data;

	rf1[i + 230] = reg_data;
	rf2[i + 230] = reg_data;
	rf3[i + 230] = reg_data;

	rf1[i + 231] = reg_data;
	rf2[i + 231] = reg_data;
	rf3[i + 231] = reg_data;

	rf1[i + 232] = reg_data;
	rf2[i + 232] = reg_data;
	rf3[i + 232] = reg_data;

	rf1[i + 233] = reg_data;
	rf2[i + 233] = reg_data;
	rf3[i + 233] = reg_data;

	rf1[i + 234] = reg_data;
	rf2[i + 234] = reg_data;
	rf3[i + 234] = reg_data;

	rf1[i + 235] = reg_data;
	rf2[i + 235] = reg_data;
	rf3[i + 235] = reg_data;

	rf1[i + 236] = reg_data;
	rf2[i + 236] = reg_data;
	rf3[i + 236] = reg_data;

	rf1[i + 237] = reg_data;
	rf2[i + 237] = reg_data;
	rf3[i + 237] = reg_data;

	rf1[i + 238] = reg_data;
	rf2[i + 238] = reg_data;
	rf3[i + 238] = reg_data;

	rf1[i + 239] = reg_data;
	rf2[i + 239] = reg_data;
	rf3[i + 239] = reg_data;

	rf1[i + 240] = reg_data;
	rf2[i + 240] = reg_data;
	rf3[i + 240] = reg_data;

	rf1[i + 241] = reg_data;
	rf2[i + 241] = reg_data;
	rf3[i + 241] = reg_data;

	rf1[i + 242] = reg_data;
	rf2[i + 242] = reg_data;
	rf3[i + 242] = reg_data;

	rf1[i + 243] = reg_data;
	rf2[i + 243] = reg_data;
	rf3[i + 243] = reg_data;

	rf1[i + 244] = reg_data;
	rf2[i + 244] = reg_data;
	rf3[i + 244] = reg_data;

	rf1[i + 245] = reg_data;
	rf2[i + 245] = reg_data;
	rf3[i + 245] = reg_data;

	rf1[i + 246] = reg_data;
	rf2[i + 246] = reg_data;
	rf3[i + 246] = reg_data;

	rf1[i + 247] = reg_data;
	rf2[i + 247] = reg_data;
	rf3[i + 247] = reg_data;

	rf1[i + 248] = reg_data;
	rf2[i + 248] = reg_data;
	rf3[i + 248] = reg_data;

	rf1[i + 249] = reg_data;
	rf2[i + 249] = reg_data;
	rf3[i + 249] = reg_data;

	rf1[i + 250] = reg_data;
	rf2[i + 250] = reg_data;
	rf3[i + 250] = reg_data;

	rf1[i + 251] = reg_data;
	rf2[i + 251] = reg_data;
	rf3[i + 251] = reg_data;

	rf1[i + 252] = reg_data;
	rf2[i + 252] = reg_data;
	rf3[i + 252] = reg_data;

	rf1[i + 253] = reg_data;
	rf2[i + 253] = reg_data;
	rf3[i + 253] = reg_data;

	rf1[i + 254] = reg_data;
	rf2[i + 254] = reg_data;
	rf3[i + 254] = reg_data;

	rf1[i + 255] = reg_data;
	rf2[i + 255] = reg_data;
	rf3[i + 255] = reg_data;

}

__global__ void test_register_file_kernel_and(uint32 *rf1, uint32 *rf2,
		uint32 *rf3, uint32 *mem1, uint32 *mem2, uint32 *mem3, uint32 reg_data,
		const uint64 sleep_cycles) {
	const uint32 i = blockIdx.x * blockIdx.y + threadIdx.x;
	register uint32 r0 = reg_data & mem1[0] & mem2[0] & mem3[0];
	register uint32 r1 = reg_data & mem1[1] & mem2[1] & mem3[1];
	register uint32 r2 = reg_data & mem1[2] & mem2[2] & mem3[2];
	register uint32 r3 = reg_data & mem1[3] & mem2[3] & mem3[3];
	register uint32 r4 = reg_data & mem1[4] & mem2[4] & mem3[4];
	register uint32 r5 = reg_data & mem1[5] & mem2[5] & mem3[5];
	register uint32 r6 = reg_data & mem1[6] & mem2[6] & mem3[6];
	register uint32 r7 = reg_data & mem1[7] & mem2[7] & mem3[7];
	register uint32 r8 = reg_data & mem1[8] & mem2[8] & mem3[8];
	register uint32 r9 = reg_data & mem1[9] & mem2[9] & mem3[9];
	register uint32 r10 = reg_data & mem1[10] & mem2[10] & mem3[10];
	register uint32 r11 = reg_data & mem1[11] & mem2[11] & mem3[11];
	register uint32 r12 = reg_data & mem1[12] & mem2[12] & mem3[12];
	register uint32 r13 = reg_data & mem1[13] & mem2[13] & mem3[13];
	register uint32 r14 = reg_data & mem1[14] & mem2[14] & mem3[14];
	register uint32 r15 = reg_data & mem1[15] & mem2[15] & mem3[15];
	register uint32 r16 = reg_data & mem1[16] & mem2[16] & mem3[16];
	register uint32 r17 = reg_data & mem1[17] & mem2[17] & mem3[17];
	register uint32 r18 = reg_data & mem1[18] & mem2[18] & mem3[18];
	register uint32 r19 = reg_data & mem1[19] & mem2[19] & mem3[19];
	register uint32 r20 = reg_data & mem1[20] & mem2[20] & mem3[20];
	register uint32 r21 = reg_data & mem1[21] & mem2[21] & mem3[21];
	register uint32 r22 = reg_data & mem1[22] & mem2[22] & mem3[22];
	register uint32 r23 = reg_data & mem1[23] & mem2[23] & mem3[23];
	register uint32 r24 = reg_data & mem1[24] & mem2[24] & mem3[24];
	register uint32 r25 = reg_data & mem1[25] & mem2[25] & mem3[25];
	register uint32 r26 = reg_data & mem1[26] & mem2[26] & mem3[26];
	register uint32 r27 = reg_data & mem1[27] & mem2[27] & mem3[27];
	register uint32 r28 = reg_data & mem1[28] & mem2[28] & mem3[28];
	register uint32 r29 = reg_data & mem1[29] & mem2[29] & mem3[29];
	register uint32 r30 = reg_data & mem1[30] & mem2[30] & mem3[30];
	register uint32 r31 = reg_data & mem1[31] & mem2[31] & mem3[31];
	register uint32 r32 = reg_data & mem1[32] & mem2[32] & mem3[32];
	register uint32 r33 = reg_data & mem1[33] & mem2[33] & mem3[33];
	register uint32 r34 = reg_data & mem1[34] & mem2[34] & mem3[34];
	register uint32 r35 = reg_data & mem1[35] & mem2[35] & mem3[35];
	register uint32 r36 = reg_data & mem1[36] & mem2[36] & mem3[36];
	register uint32 r37 = reg_data & mem1[37] & mem2[37] & mem3[37];
	register uint32 r38 = reg_data & mem1[38] & mem2[38] & mem3[38];
	register uint32 r39 = reg_data & mem1[39] & mem2[39] & mem3[39];
	register uint32 r40 = reg_data & mem1[40] & mem2[40] & mem3[40];
	register uint32 r41 = reg_data & mem1[41] & mem2[41] & mem3[41];
	register uint32 r42 = reg_data & mem1[42] & mem2[42] & mem3[42];
	register uint32 r43 = reg_data & mem1[43] & mem2[43] & mem3[43];
	register uint32 r44 = reg_data & mem1[44] & mem2[44] & mem3[44];
	register uint32 r45 = reg_data & mem1[45] & mem2[45] & mem3[45];
	register uint32 r46 = reg_data & mem1[46] & mem2[46] & mem3[46];
	register uint32 r47 = reg_data & mem1[47] & mem2[47] & mem3[47];
	register uint32 r48 = reg_data & mem1[48] & mem2[48] & mem3[48];
	register uint32 r49 = reg_data & mem1[49] & mem2[49] & mem3[49];
	register uint32 r50 = reg_data & mem1[50] & mem2[50] & mem3[50];
	register uint32 r51 = reg_data & mem1[51] & mem2[51] & mem3[51];
	register uint32 r52 = reg_data & mem1[52] & mem2[52] & mem3[52];
	register uint32 r53 = reg_data & mem1[53] & mem2[53] & mem3[53];
	register uint32 r54 = reg_data & mem1[54] & mem2[54] & mem3[54];
	register uint32 r55 = reg_data & mem1[55] & mem2[55] & mem3[55];
	register uint32 r56 = reg_data & mem1[56] & mem2[56] & mem3[56];
	register uint32 r57 = reg_data & mem1[57] & mem2[57] & mem3[57];
	register uint32 r58 = reg_data & mem1[58] & mem2[58] & mem3[58];
	register uint32 r59 = reg_data & mem1[59] & mem2[59] & mem3[59];
	register uint32 r60 = reg_data & mem1[60] & mem2[60] & mem3[60];
	register uint32 r61 = reg_data & mem1[61] & mem2[61] & mem3[61];
	register uint32 r62 = reg_data & mem1[62] & mem2[62] & mem3[62];
	register uint32 r63 = reg_data & mem1[63] & mem2[63] & mem3[63];
	register uint32 r64 = reg_data & mem1[64] & mem2[64] & mem3[64];
	register uint32 r65 = reg_data & mem1[65] & mem2[65] & mem3[65];
	register uint32 r66 = reg_data & mem1[66] & mem2[66] & mem3[66];
	register uint32 r67 = reg_data & mem1[67] & mem2[67] & mem3[67];
	register uint32 r68 = reg_data & mem1[68] & mem2[68] & mem3[68];
	register uint32 r69 = reg_data & mem1[69] & mem2[69] & mem3[69];
	register uint32 r70 = reg_data & mem1[70] & mem2[70] & mem3[70];
	register uint32 r71 = reg_data & mem1[71] & mem2[71] & mem3[71];
	register uint32 r72 = reg_data & mem1[72] & mem2[72] & mem3[72];
	register uint32 r73 = reg_data & mem1[73] & mem2[73] & mem3[73];
	register uint32 r74 = reg_data & mem1[74] & mem2[74] & mem3[74];
	register uint32 r75 = reg_data & mem1[75] & mem2[75] & mem3[75];
	register uint32 r76 = reg_data & mem1[76] & mem2[76] & mem3[76];
	register uint32 r77 = reg_data & mem1[77] & mem2[77] & mem3[77];
	register uint32 r78 = reg_data & mem1[78] & mem2[78] & mem3[78];
	register uint32 r79 = reg_data & mem1[79] & mem2[79] & mem3[79];
	register uint32 r80 = reg_data & mem1[80] & mem2[80] & mem3[80];
	register uint32 r81 = reg_data & mem1[81] & mem2[81] & mem3[81];
	register uint32 r82 = reg_data & mem1[82] & mem2[82] & mem3[82];
	register uint32 r83 = reg_data & mem1[83] & mem2[83] & mem3[83];
	register uint32 r84 = reg_data & mem1[84] & mem2[84] & mem3[84];
	register uint32 r85 = reg_data & mem1[85] & mem2[85] & mem3[85];
	register uint32 r86 = reg_data & mem1[86] & mem2[86] & mem3[86];
	register uint32 r87 = reg_data & mem1[87] & mem2[87] & mem3[87];
	register uint32 r88 = reg_data & mem1[88] & mem2[88] & mem3[88];
	register uint32 r89 = reg_data & mem1[89] & mem2[89] & mem3[89];
	register uint32 r90 = reg_data & mem1[90] & mem2[90] & mem3[90];
	register uint32 r91 = reg_data & mem1[91] & mem2[91] & mem3[91];
	register uint32 r92 = reg_data & mem1[92] & mem2[92] & mem3[92];
	register uint32 r93 = reg_data & mem1[93] & mem2[93] & mem3[93];
	register uint32 r94 = reg_data & mem1[94] & mem2[94] & mem3[94];
	register uint32 r95 = reg_data & mem1[95] & mem2[95] & mem3[95];
	register uint32 r96 = reg_data & mem1[96] & mem2[96] & mem3[96];
	register uint32 r97 = reg_data & mem1[97] & mem2[97] & mem3[97];
	register uint32 r98 = reg_data & mem1[98] & mem2[98] & mem3[98];
	register uint32 r99 = reg_data & mem1[99] & mem2[99] & mem3[99];
	register uint32 r100 = reg_data & mem1[100] & mem2[100] & mem3[100];
	register uint32 r101 = reg_data & mem1[101] & mem2[101] & mem3[101];
	register uint32 r102 = reg_data & mem1[102] & mem2[102] & mem3[102];
	register uint32 r103 = reg_data & mem1[103] & mem2[103] & mem3[103];
	register uint32 r104 = reg_data & mem1[104] & mem2[104] & mem3[104];
	register uint32 r105 = reg_data & mem1[105] & mem2[105] & mem3[105];
	register uint32 r106 = reg_data & mem1[106] & mem2[106] & mem3[106];
	register uint32 r107 = reg_data & mem1[107] & mem2[107] & mem3[107];
	register uint32 r108 = reg_data & mem1[108] & mem2[108] & mem3[108];
	register uint32 r109 = reg_data & mem1[109] & mem2[109] & mem3[109];
	register uint32 r110 = reg_data & mem1[110] & mem2[110] & mem3[110];
	register uint32 r111 = reg_data & mem1[111] & mem2[111] & mem3[111];
	register uint32 r112 = reg_data & mem1[112] & mem2[112] & mem3[112];
	register uint32 r113 = reg_data & mem1[113] & mem2[113] & mem3[113];
	register uint32 r114 = reg_data & mem1[114] & mem2[114] & mem3[114];
	register uint32 r115 = reg_data & mem1[115] & mem2[115] & mem3[115];
	register uint32 r116 = reg_data & mem1[116] & mem2[116] & mem3[116];
	register uint32 r117 = reg_data & mem1[117] & mem2[117] & mem3[117];
	register uint32 r118 = reg_data & mem1[118] & mem2[118] & mem3[118];
	register uint32 r119 = reg_data & mem1[119] & mem2[119] & mem3[119];
	register uint32 r120 = reg_data & mem1[120] & mem2[120] & mem3[120];
	register uint32 r121 = reg_data & mem1[121] & mem2[121] & mem3[121];
	register uint32 r122 = reg_data & mem1[122] & mem2[122] & mem3[122];
	register uint32 r123 = reg_data & mem1[123] & mem2[123] & mem3[123];
	register uint32 r124 = reg_data & mem1[124] & mem2[124] & mem3[124];
	register uint32 r125 = reg_data & mem1[125] & mem2[125] & mem3[125];
	register uint32 r126 = reg_data & mem1[126] & mem2[126] & mem3[126];
	register uint32 r127 = reg_data & mem1[127] & mem2[127] & mem3[127];
	register uint32 r128 = reg_data & mem1[128] & mem2[128] & mem3[128];
	register uint32 r129 = reg_data & mem1[129] & mem2[129] & mem3[129];
	register uint32 r130 = reg_data & mem1[130] & mem2[130] & mem3[130];
	register uint32 r131 = reg_data & mem1[131] & mem2[131] & mem3[131];
	register uint32 r132 = reg_data & mem1[132] & mem2[132] & mem3[132];
	register uint32 r133 = reg_data & mem1[133] & mem2[133] & mem3[133];
	register uint32 r134 = reg_data & mem1[134] & mem2[134] & mem3[134];
	register uint32 r135 = reg_data & mem1[135] & mem2[135] & mem3[135];
	register uint32 r136 = reg_data & mem1[136] & mem2[136] & mem3[136];
	register uint32 r137 = reg_data & mem1[137] & mem2[137] & mem3[137];
	register uint32 r138 = reg_data & mem1[138] & mem2[138] & mem3[138];
	register uint32 r139 = reg_data & mem1[139] & mem2[139] & mem3[139];
	register uint32 r140 = reg_data & mem1[140] & mem2[140] & mem3[140];
	register uint32 r141 = reg_data & mem1[141] & mem2[141] & mem3[141];
	register uint32 r142 = reg_data & mem1[142] & mem2[142] & mem3[142];
	register uint32 r143 = reg_data & mem1[143] & mem2[143] & mem3[143];
	register uint32 r144 = reg_data & mem1[144] & mem2[144] & mem3[144];
	register uint32 r145 = reg_data & mem1[145] & mem2[145] & mem3[145];
	register uint32 r146 = reg_data & mem1[146] & mem2[146] & mem3[146];
	register uint32 r147 = reg_data & mem1[147] & mem2[147] & mem3[147];
	register uint32 r148 = reg_data & mem1[148] & mem2[148] & mem3[148];
	register uint32 r149 = reg_data & mem1[149] & mem2[149] & mem3[149];
	register uint32 r150 = reg_data & mem1[150] & mem2[150] & mem3[150];
	register uint32 r151 = reg_data & mem1[151] & mem2[151] & mem3[151];
	register uint32 r152 = reg_data & mem1[152] & mem2[152] & mem3[152];
	register uint32 r153 = reg_data & mem1[153] & mem2[153] & mem3[153];
	register uint32 r154 = reg_data & mem1[154] & mem2[154] & mem3[154];
	register uint32 r155 = reg_data & mem1[155] & mem2[155] & mem3[155];
	register uint32 r156 = reg_data & mem1[156] & mem2[156] & mem3[156];
	register uint32 r157 = reg_data & mem1[157] & mem2[157] & mem3[157];
	register uint32 r158 = reg_data & mem1[158] & mem2[158] & mem3[158];
	register uint32 r159 = reg_data & mem1[159] & mem2[159] & mem3[159];
	register uint32 r160 = reg_data & mem1[160] & mem2[160] & mem3[160];
	register uint32 r161 = reg_data & mem1[161] & mem2[161] & mem3[161];
	register uint32 r162 = reg_data & mem1[162] & mem2[162] & mem3[162];
	register uint32 r163 = reg_data & mem1[163] & mem2[163] & mem3[163];
	register uint32 r164 = reg_data & mem1[164] & mem2[164] & mem3[164];
	register uint32 r165 = reg_data & mem1[165] & mem2[165] & mem3[165];
	register uint32 r166 = reg_data & mem1[166] & mem2[166] & mem3[166];
	register uint32 r167 = reg_data & mem1[167] & mem2[167] & mem3[167];
	register uint32 r168 = reg_data & mem1[168] & mem2[168] & mem3[168];
	register uint32 r169 = reg_data & mem1[169] & mem2[169] & mem3[169];
	register uint32 r170 = reg_data & mem1[170] & mem2[170] & mem3[170];
	register uint32 r171 = reg_data & mem1[171] & mem2[171] & mem3[171];
	register uint32 r172 = reg_data & mem1[172] & mem2[172] & mem3[172];
	register uint32 r173 = reg_data & mem1[173] & mem2[173] & mem3[173];
	register uint32 r174 = reg_data & mem1[174] & mem2[174] & mem3[174];
	register uint32 r175 = reg_data & mem1[175] & mem2[175] & mem3[175];
	register uint32 r176 = reg_data & mem1[176] & mem2[176] & mem3[176];
	register uint32 r177 = reg_data & mem1[177] & mem2[177] & mem3[177];
	register uint32 r178 = reg_data & mem1[178] & mem2[178] & mem3[178];
	register uint32 r179 = reg_data & mem1[179] & mem2[179] & mem3[179];
	register uint32 r180 = reg_data & mem1[180] & mem2[180] & mem3[180];
	register uint32 r181 = reg_data & mem1[181] & mem2[181] & mem3[181];
	register uint32 r182 = reg_data & mem1[182] & mem2[182] & mem3[182];
	register uint32 r183 = reg_data & mem1[183] & mem2[183] & mem3[183];
	register uint32 r184 = reg_data & mem1[184] & mem2[184] & mem3[184];
	register uint32 r185 = reg_data & mem1[185] & mem2[185] & mem3[185];

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

	rf1[i + 186] = reg_data;
	rf2[i + 186] = reg_data;
	rf3[i + 186] = reg_data;

	rf1[i + 187] = reg_data;
	rf2[i + 187] = reg_data;
	rf3[i + 187] = reg_data;

	rf1[i + 188] = reg_data;
	rf2[i + 188] = reg_data;
	rf3[i + 188] = reg_data;

	rf1[i + 189] = reg_data;
	rf2[i + 189] = reg_data;
	rf3[i + 189] = reg_data;

	rf1[i + 190] = reg_data;
	rf2[i + 190] = reg_data;
	rf3[i + 190] = reg_data;

	rf1[i + 191] = reg_data;
	rf2[i + 191] = reg_data;
	rf3[i + 191] = reg_data;

	rf1[i + 192] = reg_data;
	rf2[i + 192] = reg_data;
	rf3[i + 192] = reg_data;

	rf1[i + 193] = reg_data;
	rf2[i + 193] = reg_data;
	rf3[i + 193] = reg_data;

	rf1[i + 194] = reg_data;
	rf2[i + 194] = reg_data;
	rf3[i + 194] = reg_data;

	rf1[i + 195] = reg_data;
	rf2[i + 195] = reg_data;
	rf3[i + 195] = reg_data;

	rf1[i + 196] = reg_data;
	rf2[i + 196] = reg_data;
	rf3[i + 196] = reg_data;

	rf1[i + 197] = reg_data;
	rf2[i + 197] = reg_data;
	rf3[i + 197] = reg_data;

	rf1[i + 198] = reg_data;
	rf2[i + 198] = reg_data;
	rf3[i + 198] = reg_data;

	rf1[i + 199] = reg_data;
	rf2[i + 199] = reg_data;
	rf3[i + 199] = reg_data;

	rf1[i + 200] = reg_data;
	rf2[i + 200] = reg_data;
	rf3[i + 200] = reg_data;

	rf1[i + 201] = reg_data;
	rf2[i + 201] = reg_data;
	rf3[i + 201] = reg_data;

	rf1[i + 202] = reg_data;
	rf2[i + 202] = reg_data;
	rf3[i + 202] = reg_data;

	rf1[i + 203] = reg_data;
	rf2[i + 203] = reg_data;
	rf3[i + 203] = reg_data;

	rf1[i + 204] = reg_data;
	rf2[i + 204] = reg_data;
	rf3[i + 204] = reg_data;

	rf1[i + 205] = reg_data;
	rf2[i + 205] = reg_data;
	rf3[i + 205] = reg_data;

	rf1[i + 206] = reg_data;
	rf2[i + 206] = reg_data;
	rf3[i + 206] = reg_data;

	rf1[i + 207] = reg_data;
	rf2[i + 207] = reg_data;
	rf3[i + 207] = reg_data;

	rf1[i + 208] = reg_data;
	rf2[i + 208] = reg_data;
	rf3[i + 208] = reg_data;

	rf1[i + 209] = reg_data;
	rf2[i + 209] = reg_data;
	rf3[i + 209] = reg_data;

	rf1[i + 210] = reg_data;
	rf2[i + 210] = reg_data;
	rf3[i + 210] = reg_data;

	rf1[i + 211] = reg_data;
	rf2[i + 211] = reg_data;
	rf3[i + 211] = reg_data;

	rf1[i + 212] = reg_data;
	rf2[i + 212] = reg_data;
	rf3[i + 212] = reg_data;

	rf1[i + 213] = reg_data;
	rf2[i + 213] = reg_data;
	rf3[i + 213] = reg_data;

	rf1[i + 214] = reg_data;
	rf2[i + 214] = reg_data;
	rf3[i + 214] = reg_data;

	rf1[i + 215] = reg_data;
	rf2[i + 215] = reg_data;
	rf3[i + 215] = reg_data;

	rf1[i + 216] = reg_data;
	rf2[i + 216] = reg_data;
	rf3[i + 216] = reg_data;

	rf1[i + 217] = reg_data;
	rf2[i + 217] = reg_data;
	rf3[i + 217] = reg_data;

	rf1[i + 218] = reg_data;
	rf2[i + 218] = reg_data;
	rf3[i + 218] = reg_data;

	rf1[i + 219] = reg_data;
	rf2[i + 219] = reg_data;
	rf3[i + 219] = reg_data;

	rf1[i + 220] = reg_data;
	rf2[i + 220] = reg_data;
	rf3[i + 220] = reg_data;

	rf1[i + 221] = reg_data;
	rf2[i + 221] = reg_data;
	rf3[i + 221] = reg_data;

	rf1[i + 222] = reg_data;
	rf2[i + 222] = reg_data;
	rf3[i + 222] = reg_data;

	rf1[i + 223] = reg_data;
	rf2[i + 223] = reg_data;
	rf3[i + 223] = reg_data;

	rf1[i + 224] = reg_data;
	rf2[i + 224] = reg_data;
	rf3[i + 224] = reg_data;

	rf1[i + 225] = reg_data;
	rf2[i + 225] = reg_data;
	rf3[i + 225] = reg_data;

	rf1[i + 226] = reg_data;
	rf2[i + 226] = reg_data;
	rf3[i + 226] = reg_data;

	rf1[i + 227] = reg_data;
	rf2[i + 227] = reg_data;
	rf3[i + 227] = reg_data;

	rf1[i + 228] = reg_data;
	rf2[i + 228] = reg_data;
	rf3[i + 228] = reg_data;

	rf1[i + 229] = reg_data;
	rf2[i + 229] = reg_data;
	rf3[i + 229] = reg_data;

	rf1[i + 230] = reg_data;
	rf2[i + 230] = reg_data;
	rf3[i + 230] = reg_data;

	rf1[i + 231] = reg_data;
	rf2[i + 231] = reg_data;
	rf3[i + 231] = reg_data;

	rf1[i + 232] = reg_data;
	rf2[i + 232] = reg_data;
	rf3[i + 232] = reg_data;

	rf1[i + 233] = reg_data;
	rf2[i + 233] = reg_data;
	rf3[i + 233] = reg_data;

	rf1[i + 234] = reg_data;
	rf2[i + 234] = reg_data;
	rf3[i + 234] = reg_data;

	rf1[i + 235] = reg_data;
	rf2[i + 235] = reg_data;
	rf3[i + 235] = reg_data;

	rf1[i + 236] = reg_data;
	rf2[i + 236] = reg_data;
	rf3[i + 236] = reg_data;

	rf1[i + 237] = reg_data;
	rf2[i + 237] = reg_data;
	rf3[i + 237] = reg_data;

	rf1[i + 238] = reg_data;
	rf2[i + 238] = reg_data;
	rf3[i + 238] = reg_data;

	rf1[i + 239] = reg_data;
	rf2[i + 239] = reg_data;
	rf3[i + 239] = reg_data;

	rf1[i + 240] = reg_data;
	rf2[i + 240] = reg_data;
	rf3[i + 240] = reg_data;

	rf1[i + 241] = reg_data;
	rf2[i + 241] = reg_data;
	rf3[i + 241] = reg_data;

	rf1[i + 242] = reg_data;
	rf2[i + 242] = reg_data;
	rf3[i + 242] = reg_data;

	rf1[i + 243] = reg_data;
	rf2[i + 243] = reg_data;
	rf3[i + 243] = reg_data;

	rf1[i + 244] = reg_data;
	rf2[i + 244] = reg_data;
	rf3[i + 244] = reg_data;

	rf1[i + 245] = reg_data;
	rf2[i + 245] = reg_data;
	rf3[i + 245] = reg_data;

	rf1[i + 246] = reg_data;
	rf2[i + 246] = reg_data;
	rf3[i + 246] = reg_data;

	rf1[i + 247] = reg_data;
	rf2[i + 247] = reg_data;
	rf3[i + 247] = reg_data;

	rf1[i + 248] = reg_data;
	rf2[i + 248] = reg_data;
	rf3[i + 248] = reg_data;

	rf1[i + 249] = reg_data;
	rf2[i + 249] = reg_data;
	rf3[i + 249] = reg_data;

	rf1[i + 250] = reg_data;
	rf2[i + 250] = reg_data;
	rf3[i + 250] = reg_data;

	rf1[i + 251] = reg_data;
	rf2[i + 251] = reg_data;
	rf3[i + 251] = reg_data;

	rf1[i + 252] = reg_data;
	rf2[i + 252] = reg_data;
	rf3[i + 252] = reg_data;

	rf1[i + 253] = reg_data;
	rf2[i + 253] = reg_data;
	rf3[i + 253] = reg_data;

	rf1[i + 254] = reg_data;
	rf2[i + 254] = reg_data;
	rf3[i + 254] = reg_data;

	rf1[i + 255] = reg_data;
	rf2[i + 255] = reg_data;
	rf3[i + 255] = reg_data;

}

#endif /* REGISTER_KERNEL_H_ */
