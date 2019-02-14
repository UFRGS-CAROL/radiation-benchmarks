#ifndef REGISTER_KERNEL_H_
#define REGISTER_KERNEL_H_

__device__ uint64 register_file_errors1;

__device__ uint64 register_file_errors2;

__device__ uint64 register_file_errors3;


__device__ uint32 trip_mem1[256];

__device__ uint32 trip_mem2[256];

__device__ uint32 trip_mem3[256];


__global__ void test_register_file_kernel(uint32 *output_rf1, uint32 *output_rf2, uint32 *output_rf3, uint32 reg_data, const uint64 sleep_cycles) {
	const uint32 i = blockIdx.x * blockIdx.y + threadIdx.x;
	uint32 reg_errors = 0;
	register uint32 r0 = reg_data | trip_mem1[0] | trip_mem2[0] | trip_mem3[0];
	register uint32 r1 = reg_data | trip_mem1[1] | trip_mem2[1] | trip_mem3[1];
	register uint32 r2 = reg_data | trip_mem1[2] | trip_mem2[2] | trip_mem3[2];
	register uint32 r3 = reg_data | trip_mem1[3] | trip_mem2[3] | trip_mem3[3];
	register uint32 r4 = reg_data | trip_mem1[4] | trip_mem2[4] | trip_mem3[4];
	register uint32 r5 = reg_data | trip_mem1[5] | trip_mem2[5] | trip_mem3[5];
	register uint32 r6 = reg_data | trip_mem1[6] | trip_mem2[6] | trip_mem3[6];
	register uint32 r7 = reg_data | trip_mem1[7] | trip_mem2[7] | trip_mem3[7];
	register uint32 r8 = reg_data | trip_mem1[8] | trip_mem2[8] | trip_mem3[8];
	register uint32 r9 = reg_data | trip_mem1[9] | trip_mem2[9] | trip_mem3[9];
	register uint32 r10 = reg_data | trip_mem1[10] | trip_mem2[10] | trip_mem3[10];
	register uint32 r11 = reg_data | trip_mem1[11] | trip_mem2[11] | trip_mem3[11];
	register uint32 r12 = reg_data | trip_mem1[12] | trip_mem2[12] | trip_mem3[12];
	register uint32 r13 = reg_data | trip_mem1[13] | trip_mem2[13] | trip_mem3[13];
	register uint32 r14 = reg_data | trip_mem1[14] | trip_mem2[14] | trip_mem3[14];
	register uint32 r15 = reg_data | trip_mem1[15] | trip_mem2[15] | trip_mem3[15];
	register uint32 r16 = reg_data | trip_mem1[16] | trip_mem2[16] | trip_mem3[16];
	register uint32 r17 = reg_data | trip_mem1[17] | trip_mem2[17] | trip_mem3[17];
	register uint32 r18 = reg_data | trip_mem1[18] | trip_mem2[18] | trip_mem3[18];
	register uint32 r19 = reg_data | trip_mem1[19] | trip_mem2[19] | trip_mem3[19];
	register uint32 r20 = reg_data | trip_mem1[20] | trip_mem2[20] | trip_mem3[20];
	register uint32 r21 = reg_data | trip_mem1[21] | trip_mem2[21] | trip_mem3[21];
	register uint32 r22 = reg_data | trip_mem1[22] | trip_mem2[22] | trip_mem3[22];
	register uint32 r23 = reg_data | trip_mem1[23] | trip_mem2[23] | trip_mem3[23];
	register uint32 r24 = reg_data | trip_mem1[24] | trip_mem2[24] | trip_mem3[24];
	register uint32 r25 = reg_data | trip_mem1[25] | trip_mem2[25] | trip_mem3[25];
	register uint32 r26 = reg_data | trip_mem1[26] | trip_mem2[26] | trip_mem3[26];
	register uint32 r27 = reg_data | trip_mem1[27] | trip_mem2[27] | trip_mem3[27];
	register uint32 r28 = reg_data | trip_mem1[28] | trip_mem2[28] | trip_mem3[28];
	register uint32 r29 = reg_data | trip_mem1[29] | trip_mem2[29] | trip_mem3[29];
	register uint32 r30 = reg_data | trip_mem1[30] | trip_mem2[30] | trip_mem3[30];
	register uint32 r31 = reg_data | trip_mem1[31] | trip_mem2[31] | trip_mem3[31];
	register uint32 r32 = reg_data | trip_mem1[32] | trip_mem2[32] | trip_mem3[32];
	register uint32 r33 = reg_data | trip_mem1[33] | trip_mem2[33] | trip_mem3[33];
	register uint32 r34 = reg_data | trip_mem1[34] | trip_mem2[34] | trip_mem3[34];
	register uint32 r35 = reg_data | trip_mem1[35] | trip_mem2[35] | trip_mem3[35];
	register uint32 r36 = reg_data | trip_mem1[36] | trip_mem2[36] | trip_mem3[36];
	register uint32 r37 = reg_data | trip_mem1[37] | trip_mem2[37] | trip_mem3[37];
	register uint32 r38 = reg_data | trip_mem1[38] | trip_mem2[38] | trip_mem3[38];
	register uint32 r39 = reg_data | trip_mem1[39] | trip_mem2[39] | trip_mem3[39];
	register uint32 r40 = reg_data | trip_mem1[40] | trip_mem2[40] | trip_mem3[40];
	register uint32 r41 = reg_data | trip_mem1[41] | trip_mem2[41] | trip_mem3[41];
	register uint32 r42 = reg_data | trip_mem1[42] | trip_mem2[42] | trip_mem3[42];
	register uint32 r43 = reg_data | trip_mem1[43] | trip_mem2[43] | trip_mem3[43];
	register uint32 r44 = reg_data | trip_mem1[44] | trip_mem2[44] | trip_mem3[44];
	register uint32 r45 = reg_data | trip_mem1[45] | trip_mem2[45] | trip_mem3[45];
	register uint32 r46 = reg_data | trip_mem1[46] | trip_mem2[46] | trip_mem3[46];
	register uint32 r47 = reg_data | trip_mem1[47] | trip_mem2[47] | trip_mem3[47];
	register uint32 r48 = reg_data | trip_mem1[48] | trip_mem2[48] | trip_mem3[48];
	register uint32 r49 = reg_data | trip_mem1[49] | trip_mem2[49] | trip_mem3[49];
	register uint32 r50 = reg_data | trip_mem1[50] | trip_mem2[50] | trip_mem3[50];
	register uint32 r51 = reg_data | trip_mem1[51] | trip_mem2[51] | trip_mem3[51];
	register uint32 r52 = reg_data | trip_mem1[52] | trip_mem2[52] | trip_mem3[52];
	register uint32 r53 = reg_data | trip_mem1[53] | trip_mem2[53] | trip_mem3[53];
	register uint32 r54 = reg_data | trip_mem1[54] | trip_mem2[54] | trip_mem3[54];
	register uint32 r55 = reg_data | trip_mem1[55] | trip_mem2[55] | trip_mem3[55];
	register uint32 r56 = reg_data | trip_mem1[56] | trip_mem2[56] | trip_mem3[56];
	register uint32 r57 = reg_data | trip_mem1[57] | trip_mem2[57] | trip_mem3[57];
	register uint32 r58 = reg_data | trip_mem1[58] | trip_mem2[58] | trip_mem3[58];
	register uint32 r59 = reg_data | trip_mem1[59] | trip_mem2[59] | trip_mem3[59];
	register uint32 r60 = reg_data | trip_mem1[60] | trip_mem2[60] | trip_mem3[60];
	register uint32 r61 = reg_data | trip_mem1[61] | trip_mem2[61] | trip_mem3[61];
	register uint32 r62 = reg_data | trip_mem1[62] | trip_mem2[62] | trip_mem3[62];
	register uint32 r63 = reg_data | trip_mem1[63] | trip_mem2[63] | trip_mem3[63];
	register uint32 r64 = reg_data | trip_mem1[64] | trip_mem2[64] | trip_mem3[64];
	register uint32 r65 = reg_data | trip_mem1[65] | trip_mem2[65] | trip_mem3[65];
	register uint32 r66 = reg_data | trip_mem1[66] | trip_mem2[66] | trip_mem3[66];
	register uint32 r67 = reg_data | trip_mem1[67] | trip_mem2[67] | trip_mem3[67];
	register uint32 r68 = reg_data | trip_mem1[68] | trip_mem2[68] | trip_mem3[68];
	register uint32 r69 = reg_data | trip_mem1[69] | trip_mem2[69] | trip_mem3[69];
	register uint32 r70 = reg_data | trip_mem1[70] | trip_mem2[70] | trip_mem3[70];
	register uint32 r71 = reg_data | trip_mem1[71] | trip_mem2[71] | trip_mem3[71];
	register uint32 r72 = reg_data | trip_mem1[72] | trip_mem2[72] | trip_mem3[72];
	register uint32 r73 = reg_data | trip_mem1[73] | trip_mem2[73] | trip_mem3[73];
	register uint32 r74 = reg_data | trip_mem1[74] | trip_mem2[74] | trip_mem3[74];
	register uint32 r75 = reg_data | trip_mem1[75] | trip_mem2[75] | trip_mem3[75];
	register uint32 r76 = reg_data | trip_mem1[76] | trip_mem2[76] | trip_mem3[76];
	register uint32 r77 = reg_data | trip_mem1[77] | trip_mem2[77] | trip_mem3[77];
	register uint32 r78 = reg_data | trip_mem1[78] | trip_mem2[78] | trip_mem3[78];
	register uint32 r79 = reg_data | trip_mem1[79] | trip_mem2[79] | trip_mem3[79];
	register uint32 r80 = reg_data | trip_mem1[80] | trip_mem2[80] | trip_mem3[80];
	register uint32 r81 = reg_data | trip_mem1[81] | trip_mem2[81] | trip_mem3[81];
	register uint32 r82 = reg_data | trip_mem1[82] | trip_mem2[82] | trip_mem3[82];
	register uint32 r83 = reg_data | trip_mem1[83] | trip_mem2[83] | trip_mem3[83];
	register uint32 r84 = reg_data | trip_mem1[84] | trip_mem2[84] | trip_mem3[84];
	register uint32 r85 = reg_data | trip_mem1[85] | trip_mem2[85] | trip_mem3[85];
	register uint32 r86 = reg_data | trip_mem1[86] | trip_mem2[86] | trip_mem3[86];
	register uint32 r87 = reg_data | trip_mem1[87] | trip_mem2[87] | trip_mem3[87];
	register uint32 r88 = reg_data | trip_mem1[88] | trip_mem2[88] | trip_mem3[88];
	register uint32 r89 = reg_data | trip_mem1[89] | trip_mem2[89] | trip_mem3[89];
	register uint32 r90 = reg_data | trip_mem1[90] | trip_mem2[90] | trip_mem3[90];
	register uint32 r91 = reg_data | trip_mem1[91] | trip_mem2[91] | trip_mem3[91];
	register uint32 r92 = reg_data | trip_mem1[92] | trip_mem2[92] | trip_mem3[92];
	register uint32 r93 = reg_data | trip_mem1[93] | trip_mem2[93] | trip_mem3[93];
	register uint32 r94 = reg_data | trip_mem1[94] | trip_mem2[94] | trip_mem3[94];
	register uint32 r95 = reg_data | trip_mem1[95] | trip_mem2[95] | trip_mem3[95];
	register uint32 r96 = reg_data | trip_mem1[96] | trip_mem2[96] | trip_mem3[96];
	register uint32 r97 = reg_data | trip_mem1[97] | trip_mem2[97] | trip_mem3[97];
	register uint32 r98 = reg_data | trip_mem1[98] | trip_mem2[98] | trip_mem3[98];
	register uint32 r99 = reg_data | trip_mem1[99] | trip_mem2[99] | trip_mem3[99];
	register uint32 r100 = reg_data | trip_mem1[100] | trip_mem2[100] | trip_mem3[100];
	register uint32 r101 = reg_data | trip_mem1[101] | trip_mem2[101] | trip_mem3[101];
	register uint32 r102 = reg_data | trip_mem1[102] | trip_mem2[102] | trip_mem3[102];
	register uint32 r103 = reg_data | trip_mem1[103] | trip_mem2[103] | trip_mem3[103];
	register uint32 r104 = reg_data | trip_mem1[104] | trip_mem2[104] | trip_mem3[104];
	register uint32 r105 = reg_data | trip_mem1[105] | trip_mem2[105] | trip_mem3[105];
	register uint32 r106 = reg_data | trip_mem1[106] | trip_mem2[106] | trip_mem3[106];
	register uint32 r107 = reg_data | trip_mem1[107] | trip_mem2[107] | trip_mem3[107];
	register uint32 r108 = reg_data | trip_mem1[108] | trip_mem2[108] | trip_mem3[108];
	register uint32 r109 = reg_data | trip_mem1[109] | trip_mem2[109] | trip_mem3[109];
	register uint32 r110 = reg_data | trip_mem1[110] | trip_mem2[110] | trip_mem3[110];
	register uint32 r111 = reg_data | trip_mem1[111] | trip_mem2[111] | trip_mem3[111];
	register uint32 r112 = reg_data | trip_mem1[112] | trip_mem2[112] | trip_mem3[112];
	register uint32 r113 = reg_data | trip_mem1[113] | trip_mem2[113] | trip_mem3[113];
	register uint32 r114 = reg_data | trip_mem1[114] | trip_mem2[114] | trip_mem3[114];
	register uint32 r115 = reg_data | trip_mem1[115] | trip_mem2[115] | trip_mem3[115];
	register uint32 r116 = reg_data | trip_mem1[116] | trip_mem2[116] | trip_mem3[116];
	register uint32 r117 = reg_data | trip_mem1[117] | trip_mem2[117] | trip_mem3[117];
	register uint32 r118 = reg_data | trip_mem1[118] | trip_mem2[118] | trip_mem3[118];
	register uint32 r119 = reg_data | trip_mem1[119] | trip_mem2[119] | trip_mem3[119];
	register uint32 r120 = reg_data | trip_mem1[120] | trip_mem2[120] | trip_mem3[120];
	register uint32 r121 = reg_data | trip_mem1[121] | trip_mem2[121] | trip_mem3[121];
	register uint32 r122 = reg_data | trip_mem1[122] | trip_mem2[122] | trip_mem3[122];
	register uint32 r123 = reg_data | trip_mem1[123] | trip_mem2[123] | trip_mem3[123];
	register uint32 r124 = reg_data | trip_mem1[124] | trip_mem2[124] | trip_mem3[124];
	register uint32 r125 = reg_data | trip_mem1[125] | trip_mem2[125] | trip_mem3[125];
	register uint32 r126 = reg_data | trip_mem1[126] | trip_mem2[126] | trip_mem3[126];
	register uint32 r127 = reg_data | trip_mem1[127] | trip_mem2[127] | trip_mem3[127];
	register uint32 r128 = reg_data | trip_mem1[128] | trip_mem2[128] | trip_mem3[128];
	register uint32 r129 = reg_data | trip_mem1[129] | trip_mem2[129] | trip_mem3[129];
	register uint32 r130 = reg_data | trip_mem1[130] | trip_mem2[130] | trip_mem3[130];
	register uint32 r131 = reg_data | trip_mem1[131] | trip_mem2[131] | trip_mem3[131];
	register uint32 r132 = reg_data | trip_mem1[132] | trip_mem2[132] | trip_mem3[132];
	register uint32 r133 = reg_data | trip_mem1[133] | trip_mem2[133] | trip_mem3[133];
	register uint32 r134 = reg_data | trip_mem1[134] | trip_mem2[134] | trip_mem3[134];
	register uint32 r135 = reg_data | trip_mem1[135] | trip_mem2[135] | trip_mem3[135];
	register uint32 r136 = reg_data | trip_mem1[136] | trip_mem2[136] | trip_mem3[136];
	register uint32 r137 = reg_data | trip_mem1[137] | trip_mem2[137] | trip_mem3[137];
	register uint32 r138 = reg_data | trip_mem1[138] | trip_mem2[138] | trip_mem3[138];
	register uint32 r139 = reg_data | trip_mem1[139] | trip_mem2[139] | trip_mem3[139];
	register uint32 r140 = reg_data | trip_mem1[140] | trip_mem2[140] | trip_mem3[140];
	register uint32 r141 = reg_data | trip_mem1[141] | trip_mem2[141] | trip_mem3[141];
	register uint32 r142 = reg_data | trip_mem1[142] | trip_mem2[142] | trip_mem3[142];
	register uint32 r143 = reg_data | trip_mem1[143] | trip_mem2[143] | trip_mem3[143];
	register uint32 r144 = reg_data | trip_mem1[144] | trip_mem2[144] | trip_mem3[144];
	register uint32 r145 = reg_data | trip_mem1[145] | trip_mem2[145] | trip_mem3[145];
	register uint32 r146 = reg_data | trip_mem1[146] | trip_mem2[146] | trip_mem3[146];
	register uint32 r147 = reg_data | trip_mem1[147] | trip_mem2[147] | trip_mem3[147];
	register uint32 r148 = reg_data | trip_mem1[148] | trip_mem2[148] | trip_mem3[148];
	register uint32 r149 = reg_data | trip_mem1[149] | trip_mem2[149] | trip_mem3[149];
	register uint32 r150 = reg_data | trip_mem1[150] | trip_mem2[150] | trip_mem3[150];
	register uint32 r151 = reg_data | trip_mem1[151] | trip_mem2[151] | trip_mem3[151];
	register uint32 r152 = reg_data | trip_mem1[152] | trip_mem2[152] | trip_mem3[152];
	register uint32 r153 = reg_data | trip_mem1[153] | trip_mem2[153] | trip_mem3[153];
	register uint32 r154 = reg_data | trip_mem1[154] | trip_mem2[154] | trip_mem3[154];
	register uint32 r155 = reg_data | trip_mem1[155] | trip_mem2[155] | trip_mem3[155];
	register uint32 r156 = reg_data | trip_mem1[156] | trip_mem2[156] | trip_mem3[156];
	register uint32 r157 = reg_data | trip_mem1[157] | trip_mem2[157] | trip_mem3[157];
	register uint32 r158 = reg_data | trip_mem1[158] | trip_mem2[158] | trip_mem3[158];
	register uint32 r159 = reg_data | trip_mem1[159] | trip_mem2[159] | trip_mem3[159];
	register uint32 r160 = reg_data | trip_mem1[160] | trip_mem2[160] | trip_mem3[160];
	register uint32 r161 = reg_data | trip_mem1[161] | trip_mem2[161] | trip_mem3[161];
	register uint32 r162 = reg_data | trip_mem1[162] | trip_mem2[162] | trip_mem3[162];
	register uint32 r163 = reg_data | trip_mem1[163] | trip_mem2[163] | trip_mem3[163];
	register uint32 r164 = reg_data | trip_mem1[164] | trip_mem2[164] | trip_mem3[164];
	register uint32 r165 = reg_data | trip_mem1[165] | trip_mem2[165] | trip_mem3[165];
	register uint32 r166 = reg_data | trip_mem1[166] | trip_mem2[166] | trip_mem3[166];
	register uint32 r167 = reg_data | trip_mem1[167] | trip_mem2[167] | trip_mem3[167];
	register uint32 r168 = reg_data | trip_mem1[168] | trip_mem2[168] | trip_mem3[168];
	register uint32 r169 = reg_data | trip_mem1[169] | trip_mem2[169] | trip_mem3[169];
	register uint32 r170 = reg_data | trip_mem1[170] | trip_mem2[170] | trip_mem3[170];
	register uint32 r171 = reg_data | trip_mem1[171] | trip_mem2[171] | trip_mem3[171];
	register uint32 r172 = reg_data | trip_mem1[172] | trip_mem2[172] | trip_mem3[172];
	register uint32 r173 = reg_data | trip_mem1[173] | trip_mem2[173] | trip_mem3[173];
	register uint32 r174 = reg_data | trip_mem1[174] | trip_mem2[174] | trip_mem3[174];
	register uint32 r175 = reg_data | trip_mem1[175] | trip_mem2[175] | trip_mem3[175];
	register uint32 r176 = reg_data | trip_mem1[176] | trip_mem2[176] | trip_mem3[176];
	register uint32 r177 = reg_data | trip_mem1[177] | trip_mem2[177] | trip_mem3[177];
	register uint32 r178 = reg_data | trip_mem1[178] | trip_mem2[178] | trip_mem3[178];
	register uint32 r179 = reg_data | trip_mem1[179] | trip_mem2[179] | trip_mem3[179];
	register uint32 r180 = reg_data | trip_mem1[180] | trip_mem2[180] | trip_mem3[180];
	register uint32 r181 = reg_data | trip_mem1[181] | trip_mem2[181] | trip_mem3[181];
	register uint32 r182 = reg_data | trip_mem1[182] | trip_mem2[182] | trip_mem3[182];
	register uint32 r183 = reg_data | trip_mem1[183] | trip_mem2[183] | trip_mem3[183];
	register uint32 r184 = reg_data | trip_mem1[184] | trip_mem2[184] | trip_mem3[184];
	register uint32 r185 = reg_data | trip_mem1[185] | trip_mem2[185] | trip_mem3[185];

	sleep_cuda(sleep_cycles);

	output_rf1[i + 0] = r0;
	output_rf2[i + 0] = r0;
	output_rf3[i + 0] = r0;
	if (r0 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 1] = r1;
	output_rf2[i + 1] = r1;
	output_rf3[i + 1] = r1;
	if (r1 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 2] = r2;
	output_rf2[i + 2] = r2;
	output_rf3[i + 2] = r2;
	if (r2 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 3] = r3;
	output_rf2[i + 3] = r3;
	output_rf3[i + 3] = r3;
	if (r3 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 4] = r4;
	output_rf2[i + 4] = r4;
	output_rf3[i + 4] = r4;
	if (r4 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 5] = r5;
	output_rf2[i + 5] = r5;
	output_rf3[i + 5] = r5;
	if (r5 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 6] = r6;
	output_rf2[i + 6] = r6;
	output_rf3[i + 6] = r6;
	if (r6 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 7] = r7;
	output_rf2[i + 7] = r7;
	output_rf3[i + 7] = r7;
	if (r7 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 8] = r8;
	output_rf2[i + 8] = r8;
	output_rf3[i + 8] = r8;
	if (r8 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 9] = r9;
	output_rf2[i + 9] = r9;
	output_rf3[i + 9] = r9;
	if (r9 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 10] = r10;
	output_rf2[i + 10] = r10;
	output_rf3[i + 10] = r10;
	if (r10 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 11] = r11;
	output_rf2[i + 11] = r11;
	output_rf3[i + 11] = r11;
	if (r11 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 12] = r12;
	output_rf2[i + 12] = r12;
	output_rf3[i + 12] = r12;
	if (r12 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 13] = r13;
	output_rf2[i + 13] = r13;
	output_rf3[i + 13] = r13;
	if (r13 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 14] = r14;
	output_rf2[i + 14] = r14;
	output_rf3[i + 14] = r14;
	if (r14 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 15] = r15;
	output_rf2[i + 15] = r15;
	output_rf3[i + 15] = r15;
	if (r15 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 16] = r16;
	output_rf2[i + 16] = r16;
	output_rf3[i + 16] = r16;
	if (r16 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 17] = r17;
	output_rf2[i + 17] = r17;
	output_rf3[i + 17] = r17;
	if (r17 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 18] = r18;
	output_rf2[i + 18] = r18;
	output_rf3[i + 18] = r18;
	if (r18 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 19] = r19;
	output_rf2[i + 19] = r19;
	output_rf3[i + 19] = r19;
	if (r19 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 20] = r20;
	output_rf2[i + 20] = r20;
	output_rf3[i + 20] = r20;
	if (r20 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 21] = r21;
	output_rf2[i + 21] = r21;
	output_rf3[i + 21] = r21;
	if (r21 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 22] = r22;
	output_rf2[i + 22] = r22;
	output_rf3[i + 22] = r22;
	if (r22 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 23] = r23;
	output_rf2[i + 23] = r23;
	output_rf3[i + 23] = r23;
	if (r23 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 24] = r24;
	output_rf2[i + 24] = r24;
	output_rf3[i + 24] = r24;
	if (r24 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 25] = r25;
	output_rf2[i + 25] = r25;
	output_rf3[i + 25] = r25;
	if (r25 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 26] = r26;
	output_rf2[i + 26] = r26;
	output_rf3[i + 26] = r26;
	if (r26 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 27] = r27;
	output_rf2[i + 27] = r27;
	output_rf3[i + 27] = r27;
	if (r27 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 28] = r28;
	output_rf2[i + 28] = r28;
	output_rf3[i + 28] = r28;
	if (r28 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 29] = r29;
	output_rf2[i + 29] = r29;
	output_rf3[i + 29] = r29;
	if (r29 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 30] = r30;
	output_rf2[i + 30] = r30;
	output_rf3[i + 30] = r30;
	if (r30 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 31] = r31;
	output_rf2[i + 31] = r31;
	output_rf3[i + 31] = r31;
	if (r31 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 32] = r32;
	output_rf2[i + 32] = r32;
	output_rf3[i + 32] = r32;
	if (r32 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 33] = r33;
	output_rf2[i + 33] = r33;
	output_rf3[i + 33] = r33;
	if (r33 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 34] = r34;
	output_rf2[i + 34] = r34;
	output_rf3[i + 34] = r34;
	if (r34 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 35] = r35;
	output_rf2[i + 35] = r35;
	output_rf3[i + 35] = r35;
	if (r35 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 36] = r36;
	output_rf2[i + 36] = r36;
	output_rf3[i + 36] = r36;
	if (r36 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 37] = r37;
	output_rf2[i + 37] = r37;
	output_rf3[i + 37] = r37;
	if (r37 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 38] = r38;
	output_rf2[i + 38] = r38;
	output_rf3[i + 38] = r38;
	if (r38 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 39] = r39;
	output_rf2[i + 39] = r39;
	output_rf3[i + 39] = r39;
	if (r39 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 40] = r40;
	output_rf2[i + 40] = r40;
	output_rf3[i + 40] = r40;
	if (r40 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 41] = r41;
	output_rf2[i + 41] = r41;
	output_rf3[i + 41] = r41;
	if (r41 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 42] = r42;
	output_rf2[i + 42] = r42;
	output_rf3[i + 42] = r42;
	if (r42 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 43] = r43;
	output_rf2[i + 43] = r43;
	output_rf3[i + 43] = r43;
	if (r43 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 44] = r44;
	output_rf2[i + 44] = r44;
	output_rf3[i + 44] = r44;
	if (r44 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 45] = r45;
	output_rf2[i + 45] = r45;
	output_rf3[i + 45] = r45;
	if (r45 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 46] = r46;
	output_rf2[i + 46] = r46;
	output_rf3[i + 46] = r46;
	if (r46 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 47] = r47;
	output_rf2[i + 47] = r47;
	output_rf3[i + 47] = r47;
	if (r47 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 48] = r48;
	output_rf2[i + 48] = r48;
	output_rf3[i + 48] = r48;
	if (r48 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 49] = r49;
	output_rf2[i + 49] = r49;
	output_rf3[i + 49] = r49;
	if (r49 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 50] = r50;
	output_rf2[i + 50] = r50;
	output_rf3[i + 50] = r50;
	if (r50 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 51] = r51;
	output_rf2[i + 51] = r51;
	output_rf3[i + 51] = r51;
	if (r51 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 52] = r52;
	output_rf2[i + 52] = r52;
	output_rf3[i + 52] = r52;
	if (r52 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 53] = r53;
	output_rf2[i + 53] = r53;
	output_rf3[i + 53] = r53;
	if (r53 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 54] = r54;
	output_rf2[i + 54] = r54;
	output_rf3[i + 54] = r54;
	if (r54 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 55] = r55;
	output_rf2[i + 55] = r55;
	output_rf3[i + 55] = r55;
	if (r55 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 56] = r56;
	output_rf2[i + 56] = r56;
	output_rf3[i + 56] = r56;
	if (r56 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 57] = r57;
	output_rf2[i + 57] = r57;
	output_rf3[i + 57] = r57;
	if (r57 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 58] = r58;
	output_rf2[i + 58] = r58;
	output_rf3[i + 58] = r58;
	if (r58 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 59] = r59;
	output_rf2[i + 59] = r59;
	output_rf3[i + 59] = r59;
	if (r59 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 60] = r60;
	output_rf2[i + 60] = r60;
	output_rf3[i + 60] = r60;
	if (r60 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 61] = r61;
	output_rf2[i + 61] = r61;
	output_rf3[i + 61] = r61;
	if (r61 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 62] = r62;
	output_rf2[i + 62] = r62;
	output_rf3[i + 62] = r62;
	if (r62 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 63] = r63;
	output_rf2[i + 63] = r63;
	output_rf3[i + 63] = r63;
	if (r63 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 64] = r64;
	output_rf2[i + 64] = r64;
	output_rf3[i + 64] = r64;
	if (r64 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 65] = r65;
	output_rf2[i + 65] = r65;
	output_rf3[i + 65] = r65;
	if (r65 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 66] = r66;
	output_rf2[i + 66] = r66;
	output_rf3[i + 66] = r66;
	if (r66 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 67] = r67;
	output_rf2[i + 67] = r67;
	output_rf3[i + 67] = r67;
	if (r67 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 68] = r68;
	output_rf2[i + 68] = r68;
	output_rf3[i + 68] = r68;
	if (r68 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 69] = r69;
	output_rf2[i + 69] = r69;
	output_rf3[i + 69] = r69;
	if (r69 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 70] = r70;
	output_rf2[i + 70] = r70;
	output_rf3[i + 70] = r70;
	if (r70 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 71] = r71;
	output_rf2[i + 71] = r71;
	output_rf3[i + 71] = r71;
	if (r71 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 72] = r72;
	output_rf2[i + 72] = r72;
	output_rf3[i + 72] = r72;
	if (r72 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 73] = r73;
	output_rf2[i + 73] = r73;
	output_rf3[i + 73] = r73;
	if (r73 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 74] = r74;
	output_rf2[i + 74] = r74;
	output_rf3[i + 74] = r74;
	if (r74 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 75] = r75;
	output_rf2[i + 75] = r75;
	output_rf3[i + 75] = r75;
	if (r75 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 76] = r76;
	output_rf2[i + 76] = r76;
	output_rf3[i + 76] = r76;
	if (r76 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 77] = r77;
	output_rf2[i + 77] = r77;
	output_rf3[i + 77] = r77;
	if (r77 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 78] = r78;
	output_rf2[i + 78] = r78;
	output_rf3[i + 78] = r78;
	if (r78 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 79] = r79;
	output_rf2[i + 79] = r79;
	output_rf3[i + 79] = r79;
	if (r79 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 80] = r80;
	output_rf2[i + 80] = r80;
	output_rf3[i + 80] = r80;
	if (r80 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 81] = r81;
	output_rf2[i + 81] = r81;
	output_rf3[i + 81] = r81;
	if (r81 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 82] = r82;
	output_rf2[i + 82] = r82;
	output_rf3[i + 82] = r82;
	if (r82 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 83] = r83;
	output_rf2[i + 83] = r83;
	output_rf3[i + 83] = r83;
	if (r83 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 84] = r84;
	output_rf2[i + 84] = r84;
	output_rf3[i + 84] = r84;
	if (r84 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 85] = r85;
	output_rf2[i + 85] = r85;
	output_rf3[i + 85] = r85;
	if (r85 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 86] = r86;
	output_rf2[i + 86] = r86;
	output_rf3[i + 86] = r86;
	if (r86 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 87] = r87;
	output_rf2[i + 87] = r87;
	output_rf3[i + 87] = r87;
	if (r87 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 88] = r88;
	output_rf2[i + 88] = r88;
	output_rf3[i + 88] = r88;
	if (r88 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 89] = r89;
	output_rf2[i + 89] = r89;
	output_rf3[i + 89] = r89;
	if (r89 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 90] = r90;
	output_rf2[i + 90] = r90;
	output_rf3[i + 90] = r90;
	if (r90 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 91] = r91;
	output_rf2[i + 91] = r91;
	output_rf3[i + 91] = r91;
	if (r91 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 92] = r92;
	output_rf2[i + 92] = r92;
	output_rf3[i + 92] = r92;
	if (r92 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 93] = r93;
	output_rf2[i + 93] = r93;
	output_rf3[i + 93] = r93;
	if (r93 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 94] = r94;
	output_rf2[i + 94] = r94;
	output_rf3[i + 94] = r94;
	if (r94 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 95] = r95;
	output_rf2[i + 95] = r95;
	output_rf3[i + 95] = r95;
	if (r95 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 96] = r96;
	output_rf2[i + 96] = r96;
	output_rf3[i + 96] = r96;
	if (r96 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 97] = r97;
	output_rf2[i + 97] = r97;
	output_rf3[i + 97] = r97;
	if (r97 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 98] = r98;
	output_rf2[i + 98] = r98;
	output_rf3[i + 98] = r98;
	if (r98 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 99] = r99;
	output_rf2[i + 99] = r99;
	output_rf3[i + 99] = r99;
	if (r99 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 100] = r100;
	output_rf2[i + 100] = r100;
	output_rf3[i + 100] = r100;
	if (r100 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 101] = r101;
	output_rf2[i + 101] = r101;
	output_rf3[i + 101] = r101;
	if (r101 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 102] = r102;
	output_rf2[i + 102] = r102;
	output_rf3[i + 102] = r102;
	if (r102 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 103] = r103;
	output_rf2[i + 103] = r103;
	output_rf3[i + 103] = r103;
	if (r103 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 104] = r104;
	output_rf2[i + 104] = r104;
	output_rf3[i + 104] = r104;
	if (r104 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 105] = r105;
	output_rf2[i + 105] = r105;
	output_rf3[i + 105] = r105;
	if (r105 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 106] = r106;
	output_rf2[i + 106] = r106;
	output_rf3[i + 106] = r106;
	if (r106 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 107] = r107;
	output_rf2[i + 107] = r107;
	output_rf3[i + 107] = r107;
	if (r107 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 108] = r108;
	output_rf2[i + 108] = r108;
	output_rf3[i + 108] = r108;
	if (r108 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 109] = r109;
	output_rf2[i + 109] = r109;
	output_rf3[i + 109] = r109;
	if (r109 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 110] = r110;
	output_rf2[i + 110] = r110;
	output_rf3[i + 110] = r110;
	if (r110 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 111] = r111;
	output_rf2[i + 111] = r111;
	output_rf3[i + 111] = r111;
	if (r111 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 112] = r112;
	output_rf2[i + 112] = r112;
	output_rf3[i + 112] = r112;
	if (r112 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 113] = r113;
	output_rf2[i + 113] = r113;
	output_rf3[i + 113] = r113;
	if (r113 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 114] = r114;
	output_rf2[i + 114] = r114;
	output_rf3[i + 114] = r114;
	if (r114 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 115] = r115;
	output_rf2[i + 115] = r115;
	output_rf3[i + 115] = r115;
	if (r115 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 116] = r116;
	output_rf2[i + 116] = r116;
	output_rf3[i + 116] = r116;
	if (r116 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 117] = r117;
	output_rf2[i + 117] = r117;
	output_rf3[i + 117] = r117;
	if (r117 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 118] = r118;
	output_rf2[i + 118] = r118;
	output_rf3[i + 118] = r118;
	if (r118 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 119] = r119;
	output_rf2[i + 119] = r119;
	output_rf3[i + 119] = r119;
	if (r119 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 120] = r120;
	output_rf2[i + 120] = r120;
	output_rf3[i + 120] = r120;
	if (r120 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 121] = r121;
	output_rf2[i + 121] = r121;
	output_rf3[i + 121] = r121;
	if (r121 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 122] = r122;
	output_rf2[i + 122] = r122;
	output_rf3[i + 122] = r122;
	if (r122 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 123] = r123;
	output_rf2[i + 123] = r123;
	output_rf3[i + 123] = r123;
	if (r123 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 124] = r124;
	output_rf2[i + 124] = r124;
	output_rf3[i + 124] = r124;
	if (r124 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 125] = r125;
	output_rf2[i + 125] = r125;
	output_rf3[i + 125] = r125;
	if (r125 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 126] = r126;
	output_rf2[i + 126] = r126;
	output_rf3[i + 126] = r126;
	if (r126 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 127] = r127;
	output_rf2[i + 127] = r127;
	output_rf3[i + 127] = r127;
	if (r127 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 128] = r128;
	output_rf2[i + 128] = r128;
	output_rf3[i + 128] = r128;
	if (r128 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 129] = r129;
	output_rf2[i + 129] = r129;
	output_rf3[i + 129] = r129;
	if (r129 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 130] = r130;
	output_rf2[i + 130] = r130;
	output_rf3[i + 130] = r130;
	if (r130 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 131] = r131;
	output_rf2[i + 131] = r131;
	output_rf3[i + 131] = r131;
	if (r131 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 132] = r132;
	output_rf2[i + 132] = r132;
	output_rf3[i + 132] = r132;
	if (r132 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 133] = r133;
	output_rf2[i + 133] = r133;
	output_rf3[i + 133] = r133;
	if (r133 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 134] = r134;
	output_rf2[i + 134] = r134;
	output_rf3[i + 134] = r134;
	if (r134 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 135] = r135;
	output_rf2[i + 135] = r135;
	output_rf3[i + 135] = r135;
	if (r135 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 136] = r136;
	output_rf2[i + 136] = r136;
	output_rf3[i + 136] = r136;
	if (r136 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 137] = r137;
	output_rf2[i + 137] = r137;
	output_rf3[i + 137] = r137;
	if (r137 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 138] = r138;
	output_rf2[i + 138] = r138;
	output_rf3[i + 138] = r138;
	if (r138 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 139] = r139;
	output_rf2[i + 139] = r139;
	output_rf3[i + 139] = r139;
	if (r139 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 140] = r140;
	output_rf2[i + 140] = r140;
	output_rf3[i + 140] = r140;
	if (r140 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 141] = r141;
	output_rf2[i + 141] = r141;
	output_rf3[i + 141] = r141;
	if (r141 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 142] = r142;
	output_rf2[i + 142] = r142;
	output_rf3[i + 142] = r142;
	if (r142 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 143] = r143;
	output_rf2[i + 143] = r143;
	output_rf3[i + 143] = r143;
	if (r143 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 144] = r144;
	output_rf2[i + 144] = r144;
	output_rf3[i + 144] = r144;
	if (r144 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 145] = r145;
	output_rf2[i + 145] = r145;
	output_rf3[i + 145] = r145;
	if (r145 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 146] = r146;
	output_rf2[i + 146] = r146;
	output_rf3[i + 146] = r146;
	if (r146 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 147] = r147;
	output_rf2[i + 147] = r147;
	output_rf3[i + 147] = r147;
	if (r147 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 148] = r148;
	output_rf2[i + 148] = r148;
	output_rf3[i + 148] = r148;
	if (r148 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 149] = r149;
	output_rf2[i + 149] = r149;
	output_rf3[i + 149] = r149;
	if (r149 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 150] = r150;
	output_rf2[i + 150] = r150;
	output_rf3[i + 150] = r150;
	if (r150 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 151] = r151;
	output_rf2[i + 151] = r151;
	output_rf3[i + 151] = r151;
	if (r151 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 152] = r152;
	output_rf2[i + 152] = r152;
	output_rf3[i + 152] = r152;
	if (r152 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 153] = r153;
	output_rf2[i + 153] = r153;
	output_rf3[i + 153] = r153;
	if (r153 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 154] = r154;
	output_rf2[i + 154] = r154;
	output_rf3[i + 154] = r154;
	if (r154 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 155] = r155;
	output_rf2[i + 155] = r155;
	output_rf3[i + 155] = r155;
	if (r155 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 156] = r156;
	output_rf2[i + 156] = r156;
	output_rf3[i + 156] = r156;
	if (r156 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 157] = r157;
	output_rf2[i + 157] = r157;
	output_rf3[i + 157] = r157;
	if (r157 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 158] = r158;
	output_rf2[i + 158] = r158;
	output_rf3[i + 158] = r158;
	if (r158 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 159] = r159;
	output_rf2[i + 159] = r159;
	output_rf3[i + 159] = r159;
	if (r159 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 160] = r160;
	output_rf2[i + 160] = r160;
	output_rf3[i + 160] = r160;
	if (r160 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 161] = r161;
	output_rf2[i + 161] = r161;
	output_rf3[i + 161] = r161;
	if (r161 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 162] = r162;
	output_rf2[i + 162] = r162;
	output_rf3[i + 162] = r162;
	if (r162 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 163] = r163;
	output_rf2[i + 163] = r163;
	output_rf3[i + 163] = r163;
	if (r163 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 164] = r164;
	output_rf2[i + 164] = r164;
	output_rf3[i + 164] = r164;
	if (r164 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 165] = r165;
	output_rf2[i + 165] = r165;
	output_rf3[i + 165] = r165;
	if (r165 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 166] = r166;
	output_rf2[i + 166] = r166;
	output_rf3[i + 166] = r166;
	if (r166 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 167] = r167;
	output_rf2[i + 167] = r167;
	output_rf3[i + 167] = r167;
	if (r167 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 168] = r168;
	output_rf2[i + 168] = r168;
	output_rf3[i + 168] = r168;
	if (r168 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 169] = r169;
	output_rf2[i + 169] = r169;
	output_rf3[i + 169] = r169;
	if (r169 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 170] = r170;
	output_rf2[i + 170] = r170;
	output_rf3[i + 170] = r170;
	if (r170 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 171] = r171;
	output_rf2[i + 171] = r171;
	output_rf3[i + 171] = r171;
	if (r171 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 172] = r172;
	output_rf2[i + 172] = r172;
	output_rf3[i + 172] = r172;
	if (r172 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 173] = r173;
	output_rf2[i + 173] = r173;
	output_rf3[i + 173] = r173;
	if (r173 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 174] = r174;
	output_rf2[i + 174] = r174;
	output_rf3[i + 174] = r174;
	if (r174 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 175] = r175;
	output_rf2[i + 175] = r175;
	output_rf3[i + 175] = r175;
	if (r175 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 176] = r176;
	output_rf2[i + 176] = r176;
	output_rf3[i + 176] = r176;
	if (r176 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 177] = r177;
	output_rf2[i + 177] = r177;
	output_rf3[i + 177] = r177;
	if (r177 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 178] = r178;
	output_rf2[i + 178] = r178;
	output_rf3[i + 178] = r178;
	if (r178 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 179] = r179;
	output_rf2[i + 179] = r179;
	output_rf3[i + 179] = r179;
	if (r179 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 180] = r180;
	output_rf2[i + 180] = r180;
	output_rf3[i + 180] = r180;
	if (r180 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 181] = r181;
	output_rf2[i + 181] = r181;
	output_rf3[i + 181] = r181;
	if (r181 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 182] = r182;
	output_rf2[i + 182] = r182;
	output_rf3[i + 182] = r182;
	if (r182 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 183] = r183;
	output_rf2[i + 183] = r183;
	output_rf3[i + 183] = r183;
	if (r183 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 184] = r184;
	output_rf2[i + 184] = r184;
	output_rf3[i + 184] = r184;
	if (r184 != reg_data) {
		reg_errors++;
	}
	output_rf1[i + 185] = r185;
	output_rf2[i + 185] = r185;
	output_rf3[i + 185] = r185;
	if (r185 != reg_data) {
		reg_errors++;
	}

	if (reg_errors != 0) {
		atomicAdd(&register_file_errors1, reg_errors);
		atomicAdd(&register_file_errors2, reg_errors);
		atomicAdd(&register_file_errors3, reg_errors);
	}
	__syncthreads();
}

#endif /* REGISTER_KERNEL_H_ */
