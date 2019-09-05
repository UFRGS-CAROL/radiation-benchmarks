#ifndef REGISTER_KERNEL_H_
#define REGISTER_KERNEL_H_

#include "utils.h"

__global__ void test_register_file_kernel_or(uint32 *rf1, uint32 *rf2, uint32 *rf3, uint32 *mem1, uint32 *mem2, uint32 *mem3, uint32 reg_data, const uint64 sleep_cycles) {
	const uint32 i =  (blockDim.x * blockIdx.x  + threadIdx.x) * 256;
	register uint32 r0 = mem1[0];
	register uint32 r1 = mem1[1];
	register uint32 r2 = mem1[2];
	register uint32 r3 = mem1[3];
	register uint32 r4 = mem1[4];
	register uint32 r5 = mem1[5];
	register uint32 r6 = mem1[6];
	register uint32 r7 = mem1[7];
	register uint32 r8 = mem1[8];
	register uint32 r9 = mem1[9];
	register uint32 r10 = mem1[10];
	register uint32 r11 = mem1[11];
	register uint32 r12 = mem1[12];
	register uint32 r13 = mem1[13];
	register uint32 r14 = mem1[14];
	register uint32 r15 = mem1[15];
	register uint32 r16 = mem1[16];
	register uint32 r17 = mem1[17];
	register uint32 r18 = mem1[18];
	register uint32 r19 = mem1[19];
	register uint32 r20 = mem1[20];
	register uint32 r21 = mem1[21];
	register uint32 r22 = mem1[22];
	register uint32 r23 = mem1[23];
	register uint32 r24 = mem1[24];
	register uint32 r25 = mem1[25];
	register uint32 r26 = mem1[26];
	register uint32 r27 = mem1[27];
	register uint32 r28 = mem1[28];
	register uint32 r29 = mem1[29];
	register uint32 r30 = mem1[30];
	register uint32 r31 = mem1[31];
	register uint32 r32 = mem1[32];
	register uint32 r33 = mem1[33];
	register uint32 r34 = mem1[34];
	register uint32 r35 = mem1[35];
	register uint32 r36 = mem1[36];
	register uint32 r37 = mem1[37];
	register uint32 r38 = mem1[38];
	register uint32 r39 = mem1[39];
	register uint32 r40 = mem1[40];
	register uint32 r41 = mem1[41];
	register uint32 r42 = mem1[42];
	register uint32 r43 = mem1[43];
	register uint32 r44 = mem1[44];
	register uint32 r45 = mem1[45];
	register uint32 r46 = mem1[46];
	register uint32 r47 = mem1[47];
	register uint32 r48 = mem1[48];
	register uint32 r49 = mem1[49];
	register uint32 r50 = mem1[50];
	register uint32 r51 = mem1[51];
	register uint32 r52 = mem1[52];
	register uint32 r53 = mem1[53];
	register uint32 r54 = mem1[54];
	register uint32 r55 = mem1[55];
	register uint32 r56 = mem1[56];
	register uint32 r57 = mem1[57];
	register uint32 r58 = mem1[58];
	register uint32 r59 = mem1[59];
	register uint32 r60 = mem1[60];
	register uint32 r61 = mem1[61];
	register uint32 r62 = mem1[62];
	register uint32 r63 = mem1[63];
	register uint32 r64 = mem1[64];
	register uint32 r65 = mem1[65];
	register uint32 r66 = mem1[66];
	register uint32 r67 = mem1[67];
	register uint32 r68 = mem1[68];
	register uint32 r69 = mem1[69];
	register uint32 r70 = mem1[70];
	register uint32 r71 = mem1[71];
	register uint32 r72 = mem1[72];
	register uint32 r73 = mem1[73];
	register uint32 r74 = mem1[74];
	register uint32 r75 = mem1[75];
	register uint32 r76 = mem1[76];
	register uint32 r77 = mem1[77];
	register uint32 r78 = mem1[78];
	register uint32 r79 = mem1[79];
	register uint32 r80 = mem1[80];
	register uint32 r81 = mem1[81];
	register uint32 r82 = mem1[82];
	register uint32 r83 = mem1[83];
	register uint32 r84 = mem1[84];
	register uint32 r85 = mem1[85];
	register uint32 r86 = mem1[86];
	register uint32 r87 = mem1[87];
	register uint32 r88 = mem1[88];
	register uint32 r89 = mem1[89];
	register uint32 r90 = mem1[90];
	register uint32 r91 = mem1[91];
	register uint32 r92 = mem1[92];
	register uint32 r93 = mem1[93];
	register uint32 r94 = mem1[94];
	register uint32 r95 = mem1[95];
	register uint32 r96 = mem1[96];
	register uint32 r97 = mem1[97];
	register uint32 r98 = mem1[98];
	register uint32 r99 = mem1[99];
	register uint32 r100 = mem1[100];
	register uint32 r101 = mem1[101];
	register uint32 r102 = mem1[102];
	register uint32 r103 = mem1[103];
	register uint32 r104 = mem1[104];
	register uint32 r105 = mem1[105];
	register uint32 r106 = mem1[106];
	register uint32 r107 = mem1[107];
	register uint32 r108 = mem1[108];
	register uint32 r109 = mem1[109];
	register uint32 r110 = mem1[110];
	register uint32 r111 = mem1[111];
	register uint32 r112 = mem1[112];
	register uint32 r113 = mem1[113];
	register uint32 r114 = mem1[114];
	register uint32 r115 = mem1[115];
	register uint32 r116 = mem1[116];
	register uint32 r117 = mem1[117];
	register uint32 r118 = mem1[118];
	register uint32 r119 = mem1[119];
	register uint32 r120 = mem1[120];
	register uint32 r121 = mem1[121];
	register uint32 r122 = mem1[122];
	register uint32 r123 = mem1[123];
	register uint32 r124 = mem1[124];
	register uint32 r125 = mem1[125];
	register uint32 r126 = mem1[126];
	register uint32 r127 = mem1[127];
	register uint32 r128 = mem1[128];
	register uint32 r129 = mem1[129];
	register uint32 r130 = mem1[130];
	register uint32 r131 = mem1[131];
	register uint32 r132 = mem1[132];
	register uint32 r133 = mem1[133];
	register uint32 r134 = mem1[134];
	register uint32 r135 = mem1[135];
	register uint32 r136 = mem1[136];
	register uint32 r137 = mem1[137];
	register uint32 r138 = mem1[138];
	register uint32 r139 = mem1[139];
	register uint32 r140 = mem1[140];
	register uint32 r141 = mem1[141];
	register uint32 r142 = mem1[142];
	register uint32 r143 = mem1[143];
	register uint32 r144 = mem1[144];
	register uint32 r145 = mem1[145];
	register uint32 r146 = mem1[146];
	register uint32 r147 = mem1[147];
	register uint32 r148 = mem1[148];
	register uint32 r149 = mem1[149];
	register uint32 r150 = mem1[150];
	register uint32 r151 = mem1[151];
	register uint32 r152 = mem1[152];
	register uint32 r153 = mem1[153];
	register uint32 r154 = mem1[154];
	register uint32 r155 = mem1[155];
	register uint32 r156 = mem1[156];
	register uint32 r157 = mem1[157];
	register uint32 r158 = mem1[158];
	register uint32 r159 = mem1[159];
	register uint32 r160 = mem1[160];
	register uint32 r161 = mem1[161];
	register uint32 r162 = mem1[162];
	register uint32 r163 = mem1[163];
	register uint32 r164 = mem1[164];
	register uint32 r165 = mem1[165];
	register uint32 r166 = mem1[166];
	register uint32 r167 = mem1[167];
	register uint32 r168 = mem1[168];
	register uint32 r169 = mem1[169];
	register uint32 r170 = mem1[170];
	register uint32 r171 = mem1[171];
	register uint32 r172 = mem1[172];
	register uint32 r173 = mem1[173];
	register uint32 r174 = mem1[174];
	register uint32 r175 = mem1[175];
	register uint32 r176 = mem1[176];
	register uint32 r177 = mem1[177];
	register uint32 r178 = mem1[178];
	register uint32 r179 = mem1[179];
	register uint32 r180 = mem1[180];
	register uint32 r181 = mem1[181];
	register uint32 r182 = mem1[182];
	register uint32 r183 = mem1[183];
	register uint32 r184 = mem1[184];
	register uint32 r185 = mem1[185];
	register uint32 r186 = mem1[186];
	register uint32 r187 = mem1[187];
	register uint32 r188 = mem1[188];
	register uint32 r189 = mem1[189];
	register uint32 r190 = mem1[190];
	register uint32 r191 = mem1[191];
	register uint32 r192 = mem1[192];
	register uint32 r193 = mem1[193];
	register uint32 r194 = mem1[194];
	register uint32 r195 = mem1[195];
	register uint32 r196 = mem1[196];
	register uint32 r197 = mem1[197];
	register uint32 r198 = mem1[198];
	register uint32 r199 = mem1[199];
	register uint32 r200 = mem1[200];
	register uint32 r201 = mem1[201];
	register uint32 r202 = mem1[202];
	register uint32 r203 = mem1[203];
	register uint32 r204 = mem1[204];
	register uint32 r205 = mem1[205];
	register uint32 r206 = mem1[206];
	register uint32 r207 = mem1[207];
	register uint32 r208 = mem1[208];
	register uint32 r209 = mem1[209];
	register uint32 r210 = mem1[210];
	register uint32 r211 = mem1[211];
	register uint32 r212 = mem1[212];
	register uint32 r213 = mem1[213];
	register uint32 r214 = mem1[214];
	register uint32 r215 = mem1[215];
	register uint32 r216 = mem1[216];
	register uint32 r217 = mem1[217];
	register uint32 r218 = mem1[218];
	register uint32 r219 = mem1[219];
	register uint32 r220 = mem1[220];
	register uint32 r221 = mem1[221];
	register uint32 r222 = mem1[222];
	register uint32 r223 = mem1[223];
	register uint32 r224 = mem1[224];
	register uint32 r225 = mem1[225];
	register uint32 r226 = mem1[226];
	register uint32 r227 = mem1[227];
	register uint32 r228 = mem1[228];
	register uint32 r229 = mem1[229];
	register uint32 r230 = mem1[230];
	register uint32 r231 = mem1[231];
	register uint32 r232 = mem1[232];
	register uint32 r233 = mem1[233];
	register uint32 r234 = mem1[234];
	register uint32 r235 = mem1[235];
	register uint32 r236 = mem1[236];
	register uint32 r237 = mem1[237];
	register uint32 r238 = mem1[238];
	register uint32 r239 = mem1[239];
	register uint32 r240 = mem1[240];
	register uint32 r241 = mem1[241];
	register uint32 r242 = mem1[242];
	register uint32 r243 = mem1[243];
	register uint32 r244 = mem1[244];
	register uint32 r245 = mem1[245];
	register uint32 r246 = mem1[246];
	register uint32 r247 = mem1[247];
	register uint32 r248 = mem1[248];
	register uint32 r249 = mem1[249];
	register uint32 r250 = mem1[250];
	register uint32 r251 = mem1[251];
	register uint32 r252 = mem1[252];
	register uint32 r253 = mem1[253];
	register uint32 r254 = mem1[254];
	register uint32 r255 = mem1[255];

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

#endif /* REGISTER_KERNEL_H_ */
