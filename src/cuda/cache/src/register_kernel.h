#ifndef REGISTER_KERNEL_H_
#define REGISTER_KERNEL_H_

__device__ uint64 register_file_errors;

__global__ void test_register_file_kernel(uint32 *output_rf, const uint32 reg_data, const uint64 sleep_cycles) {
	const uint32 i = blockIdx.x * blockIdx.y + threadIdx.x;
	uint32 reg_errors = 0;
	register uint32 r0 = output_rf[i + 0], r1 = output_rf[i + 1];
	register uint32 r2 = output_rf[i + 2], r3 = output_rf[i + 3];
	register uint32 r4 = output_rf[i + 4], r5 = output_rf[i + 5];
	register uint32 r6 = output_rf[i + 6], r7 = output_rf[i + 7];
	register uint32 r8 = output_rf[i + 8], r9 = output_rf[i + 9];
	register uint32 r10 = output_rf[i + 10], r11 = output_rf[i + 11];
	register uint32 r12 = output_rf[i + 12], r13 = output_rf[i + 13];
	register uint32 r14 = output_rf[i + 14], r15 = output_rf[i + 15];
	register uint32 r16 = output_rf[i + 16], r17 = output_rf[i + 17];
	register uint32 r18 = output_rf[i + 18], r19 = output_rf[i + 19];
	register uint32 r20 = output_rf[i + 20], r21 = output_rf[i + 21];
	register uint32 r22 = output_rf[i + 22], r23 = output_rf[i + 23];
	register uint32 r24 = output_rf[i + 24], r25 = output_rf[i + 25];
	register uint32 r26 = output_rf[i + 26], r27 = output_rf[i + 27];
	register uint32 r28 = output_rf[i + 28], r29 = output_rf[i + 29];
	register uint32 r30 = output_rf[i + 30], r31 = output_rf[i + 31];
	register uint32 r32 = output_rf[i + 32], r33 = output_rf[i + 33];
	register uint32 r34 = output_rf[i + 34], r35 = output_rf[i + 35];
	register uint32 r36 = output_rf[i + 36], r37 = output_rf[i + 37];
	register uint32 r38 = output_rf[i + 38], r39 = output_rf[i + 39];
	register uint32 r40 = output_rf[i + 40], r41 = output_rf[i + 41];
	register uint32 r42 = output_rf[i + 42], r43 = output_rf[i + 43];
	register uint32 r44 = output_rf[i + 44], r45 = output_rf[i + 45];
	register uint32 r46 = output_rf[i + 46], r47 = output_rf[i + 47];
	register uint32 r48 = output_rf[i + 48], r49 = output_rf[i + 49];
	register uint32 r50 = output_rf[i + 50], r51 = output_rf[i + 51];
	register uint32 r52 = output_rf[i + 52], r53 = output_rf[i + 53];
	register uint32 r54 = output_rf[i + 54], r55 = output_rf[i + 55];
	register uint32 r56 = output_rf[i + 56], r57 = output_rf[i + 57];
	register uint32 r58 = output_rf[i + 58], r59 = output_rf[i + 59];
	register uint32 r60 = output_rf[i + 60], r61 = output_rf[i + 61];
	register uint32 r62 = output_rf[i + 62], r63 = output_rf[i + 63];
	register uint32 r64 = output_rf[i + 64], r65 = output_rf[i + 65];
	register uint32 r66 = output_rf[i + 66], r67 = output_rf[i + 67];
	register uint32 r68 = output_rf[i + 68], r69 = output_rf[i + 69];
	register uint32 r70 = output_rf[i + 70], r71 = output_rf[i + 71];
	register uint32 r72 = output_rf[i + 72], r73 = output_rf[i + 73];
	register uint32 r74 = output_rf[i + 74], r75 = output_rf[i + 75];
	register uint32 r76 = output_rf[i + 76], r77 = output_rf[i + 77];
	register uint32 r78 = output_rf[i + 78], r79 = output_rf[i + 79];
	register uint32 r80 = output_rf[i + 80], r81 = output_rf[i + 81];
	register uint32 r82 = output_rf[i + 82], r83 = output_rf[i + 83];
	register uint32 r84 = output_rf[i + 84], r85 = output_rf[i + 85];
	register uint32 r86 = output_rf[i + 86], r87 = output_rf[i + 87];
	register uint32 r88 = output_rf[i + 88], r89 = output_rf[i + 89];
	register uint32 r90 = output_rf[i + 90], r91 = output_rf[i + 91];
	register uint32 r92 = output_rf[i + 92], r93 = output_rf[i + 93];
	register uint32 r94 = output_rf[i + 94], r95 = output_rf[i + 95];
	register uint32 r96 = output_rf[i + 96], r97 = output_rf[i + 97];
	register uint32 r98 = output_rf[i + 98], r99 = output_rf[i + 99];
	register uint32 r100 = output_rf[i + 100], r101 = output_rf[i + 101];
	register uint32 r102 = output_rf[i + 102], r103 = output_rf[i + 103];
	register uint32 r104 = output_rf[i + 104], r105 = output_rf[i + 105];
	register uint32 r106 = output_rf[i + 106], r107 = output_rf[i + 107];
	register uint32 r108 = output_rf[i + 108], r109 = output_rf[i + 109];
	register uint32 r110 = output_rf[i + 110], r111 = output_rf[i + 111];
	register uint32 r112 = output_rf[i + 112], r113 = output_rf[i + 113];
	register uint32 r114 = output_rf[i + 114], r115 = output_rf[i + 115];
	register uint32 r116 = output_rf[i + 116], r117 = output_rf[i + 117];
	register uint32 r118 = output_rf[i + 118], r119 = output_rf[i + 119];
	register uint32 r120 = output_rf[i + 120], r121 = output_rf[i + 121];
	register uint32 r122 = output_rf[i + 122], r123 = output_rf[i + 123];
	register uint32 r124 = output_rf[i + 124], r125 = output_rf[i + 125];
	register uint32 r126 = output_rf[i + 126], r127 = output_rf[i + 127];
	register uint32 r128 = output_rf[i + 128], r129 = output_rf[i + 129];
	register uint32 r130 = output_rf[i + 130], r131 = output_rf[i + 131];
	register uint32 r132 = output_rf[i + 132], r133 = output_rf[i + 133];
	register uint32 r134 = output_rf[i + 134], r135 = output_rf[i + 135];
	register uint32 r136 = output_rf[i + 136], r137 = output_rf[i + 137];
	register uint32 r138 = output_rf[i + 138], r139 = output_rf[i + 139];
	register uint32 r140 = output_rf[i + 140], r141 = output_rf[i + 141];
	register uint32 r142 = output_rf[i + 142], r143 = output_rf[i + 143];
	register uint32 r144 = output_rf[i + 144], r145 = output_rf[i + 145];
	register uint32 r146 = output_rf[i + 146], r147 = output_rf[i + 147];
	register uint32 r148 = output_rf[i + 148], r149 = output_rf[i + 149];
	register uint32 r150 = output_rf[i + 150], r151 = output_rf[i + 151];
	register uint32 r152 = output_rf[i + 152], r153 = output_rf[i + 153];
	register uint32 r154 = output_rf[i + 154], r155 = output_rf[i + 155];
	register uint32 r156 = output_rf[i + 156], r157 = output_rf[i + 157];
	register uint32 r158 = output_rf[i + 158], r159 = output_rf[i + 159];
	register uint32 r160 = output_rf[i + 160], r161 = output_rf[i + 161];
	register uint32 r162 = output_rf[i + 162], r163 = output_rf[i + 163];
	register uint32 r164 = output_rf[i + 164], r165 = output_rf[i + 165];
	register uint32 r166 = output_rf[i + 166], r167 = output_rf[i + 167];
	register uint32 r168 = output_rf[i + 168], r169 = output_rf[i + 169];
	register uint32 r170 = output_rf[i + 170], r171 = output_rf[i + 171];
	register uint32 r172 = output_rf[i + 172], r173 = output_rf[i + 173];
	register uint32 r174 = output_rf[i + 174], r175 = output_rf[i + 175];
	register uint32 r176 = output_rf[i + 176], r177 = output_rf[i + 177];
	register uint32 r178 = output_rf[i + 178], r179 = output_rf[i + 179];
	register uint32 r180 = output_rf[i + 180], r181 = output_rf[i + 181];
	register uint32 r182 = output_rf[i + 182], r183 = output_rf[i + 183];
	register uint32 r184 = output_rf[i + 184], r185 = output_rf[i + 185];

	sleep_cuda(sleep_cycles);

	output_rf[i + 0] = r0;
	if (r0 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 1] = r1;
	if (r1 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 2] = r2;
	if (r2 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 3] = r3;
	if (r3 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 4] = r4;
	if (r4 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 5] = r5;
	if (r5 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 6] = r6;
	if (r6 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 7] = r7;
	if (r7 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 8] = r8;
	if (r8 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 9] = r9;
	if (r9 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 10] = r10;
	if (r10 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 11] = r11;
	if (r11 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 12] = r12;
	if (r12 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 13] = r13;
	if (r13 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 14] = r14;
	if (r14 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 15] = r15;
	if (r15 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 16] = r16;
	if (r16 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 17] = r17;
	if (r17 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 18] = r18;
	if (r18 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 19] = r19;
	if (r19 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 20] = r20;
	if (r20 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 21] = r21;
	if (r21 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 22] = r22;
	if (r22 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 23] = r23;
	if (r23 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 24] = r24;
	if (r24 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 25] = r25;
	if (r25 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 26] = r26;
	if (r26 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 27] = r27;
	if (r27 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 28] = r28;
	if (r28 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 29] = r29;
	if (r29 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 30] = r30;
	if (r30 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 31] = r31;
	if (r31 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 32] = r32;
	if (r32 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 33] = r33;
	if (r33 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 34] = r34;
	if (r34 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 35] = r35;
	if (r35 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 36] = r36;
	if (r36 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 37] = r37;
	if (r37 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 38] = r38;
	if (r38 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 39] = r39;
	if (r39 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 40] = r40;
	if (r40 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 41] = r41;
	if (r41 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 42] = r42;
	if (r42 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 43] = r43;
	if (r43 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 44] = r44;
	if (r44 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 45] = r45;
	if (r45 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 46] = r46;
	if (r46 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 47] = r47;
	if (r47 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 48] = r48;
	if (r48 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 49] = r49;
	if (r49 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 50] = r50;
	if (r50 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 51] = r51;
	if (r51 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 52] = r52;
	if (r52 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 53] = r53;
	if (r53 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 54] = r54;
	if (r54 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 55] = r55;
	if (r55 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 56] = r56;
	if (r56 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 57] = r57;
	if (r57 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 58] = r58;
	if (r58 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 59] = r59;
	if (r59 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 60] = r60;
	if (r60 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 61] = r61;
	if (r61 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 62] = r62;
	if (r62 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 63] = r63;
	if (r63 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 64] = r64;
	if (r64 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 65] = r65;
	if (r65 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 66] = r66;
	if (r66 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 67] = r67;
	if (r67 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 68] = r68;
	if (r68 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 69] = r69;
	if (r69 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 70] = r70;
	if (r70 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 71] = r71;
	if (r71 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 72] = r72;
	if (r72 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 73] = r73;
	if (r73 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 74] = r74;
	if (r74 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 75] = r75;
	if (r75 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 76] = r76;
	if (r76 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 77] = r77;
	if (r77 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 78] = r78;
	if (r78 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 79] = r79;
	if (r79 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 80] = r80;
	if (r80 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 81] = r81;
	if (r81 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 82] = r82;
	if (r82 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 83] = r83;
	if (r83 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 84] = r84;
	if (r84 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 85] = r85;
	if (r85 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 86] = r86;
	if (r86 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 87] = r87;
	if (r87 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 88] = r88;
	if (r88 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 89] = r89;
	if (r89 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 90] = r90;
	if (r90 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 91] = r91;
	if (r91 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 92] = r92;
	if (r92 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 93] = r93;
	if (r93 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 94] = r94;
	if (r94 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 95] = r95;
	if (r95 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 96] = r96;
	if (r96 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 97] = r97;
	if (r97 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 98] = r98;
	if (r98 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 99] = r99;
	if (r99 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 100] = r100;
	if (r100 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 101] = r101;
	if (r101 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 102] = r102;
	if (r102 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 103] = r103;
	if (r103 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 104] = r104;
	if (r104 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 105] = r105;
	if (r105 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 106] = r106;
	if (r106 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 107] = r107;
	if (r107 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 108] = r108;
	if (r108 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 109] = r109;
	if (r109 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 110] = r110;
	if (r110 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 111] = r111;
	if (r111 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 112] = r112;
	if (r112 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 113] = r113;
	if (r113 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 114] = r114;
	if (r114 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 115] = r115;
	if (r115 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 116] = r116;
	if (r116 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 117] = r117;
	if (r117 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 118] = r118;
	if (r118 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 119] = r119;
	if (r119 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 120] = r120;
	if (r120 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 121] = r121;
	if (r121 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 122] = r122;
	if (r122 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 123] = r123;
	if (r123 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 124] = r124;
	if (r124 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 125] = r125;
	if (r125 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 126] = r126;
	if (r126 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 127] = r127;
	if (r127 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 128] = r128;
	if (r128 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 129] = r129;
	if (r129 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 130] = r130;
	if (r130 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 131] = r131;
	if (r131 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 132] = r132;
	if (r132 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 133] = r133;
	if (r133 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 134] = r134;
	if (r134 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 135] = r135;
	if (r135 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 136] = r136;
	if (r136 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 137] = r137;
	if (r137 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 138] = r138;
	if (r138 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 139] = r139;
	if (r139 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 140] = r140;
	if (r140 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 141] = r141;
	if (r141 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 142] = r142;
	if (r142 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 143] = r143;
	if (r143 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 144] = r144;
	if (r144 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 145] = r145;
	if (r145 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 146] = r146;
	if (r146 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 147] = r147;
	if (r147 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 148] = r148;
	if (r148 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 149] = r149;
	if (r149 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 150] = r150;
	if (r150 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 151] = r151;
	if (r151 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 152] = r152;
	if (r152 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 153] = r153;
	if (r153 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 154] = r154;
	if (r154 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 155] = r155;
	if (r155 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 156] = r156;
	if (r156 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 157] = r157;
	if (r157 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 158] = r158;
	if (r158 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 159] = r159;
	if (r159 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 160] = r160;
	if (r160 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 161] = r161;
	if (r161 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 162] = r162;
	if (r162 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 163] = r163;
	if (r163 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 164] = r164;
	if (r164 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 165] = r165;
	if (r165 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 166] = r166;
	if (r166 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 167] = r167;
	if (r167 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 168] = r168;
	if (r168 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 169] = r169;
	if (r169 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 170] = r170;
	if (r170 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 171] = r171;
	if (r171 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 172] = r172;
	if (r172 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 173] = r173;
	if (r173 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 174] = r174;
	if (r174 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 175] = r175;
	if (r175 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 176] = r176;
	if (r176 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 177] = r177;
	if (r177 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 178] = r178;
	if (r178 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 179] = r179;
	if (r179 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 180] = r180;
	if (r180 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 181] = r181;
	if (r181 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 182] = r182;
	if (r182 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 183] = r183;
	if (r183 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 184] = r184;
	if (r184 != reg_data) {
		reg_errors++;
	}
	output_rf[i + 185] = r185;
	if (r185 != reg_data) {
		reg_errors++;
	}
	if (reg_errors != 0) {
		atomicAdd(&register_file_errors, reg_errors);
	}
	__syncthreads();
}

#endif /* REGISTER_KERNEL_H_ */
