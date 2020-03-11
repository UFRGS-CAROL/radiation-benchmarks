#ifndef BRANCH_KERNEL_H_
#define BRANCH_KERNEL_H_

template<uint32_t UNROLL_MAX, typename int_t>
__global__ void branch_int_kernel(int_t* src, int_t* dst, uint32_t op) {
	const uint32_t i = (blockDim.x * blockIdx.x + threadIdx.x);

	if (threadIdx.x == 0) {
		dst[i] = 0;
	} else if (threadIdx.x == 1) {
		dst[i] = 1;
	} else if (threadIdx.x == 2) {
		dst[i] = 2;
	} else if (threadIdx.x == 3) {
		dst[i] = 3;
	} else if (threadIdx.x == 4) {
		dst[i] = 4;
	} else if (threadIdx.x == 5) {
		dst[i] = 5;
	} else if (threadIdx.x == 6) {
		dst[i] = 6;
	} else if (threadIdx.x == 7) {
		dst[i] = 7;
	} else if (threadIdx.x == 8) {
		dst[i] = 8;
	} else if (threadIdx.x == 9) {
		dst[i] = 9;
	} else if (threadIdx.x == 10) {
		dst[i] = 10;
	} else if (threadIdx.x == 11) {
		dst[i] = 11;
	} else if (threadIdx.x == 12) {
		dst[i] = 12;
	} else if (threadIdx.x == 13) {
		dst[i] = 13;
	} else if (threadIdx.x == 14) {
		dst[i] = 14;
	} else if (threadIdx.x == 15) {
		dst[i] = 15;
	} else if (threadIdx.x == 16) {
		dst[i] = 16;
	} else if (threadIdx.x == 17) {
		dst[i] = 17;
	} else if (threadIdx.x == 18) {
		dst[i] = 18;
	} else if (threadIdx.x == 19) {
		dst[i] = 19;
	} else if (threadIdx.x == 20) {
		dst[i] = 20;
	} else if (threadIdx.x == 21) {
		dst[i] = 21;
	} else if (threadIdx.x == 22) {
		dst[i] = 22;
	} else if (threadIdx.x == 23) {
		dst[i] = 23;
	} else if (threadIdx.x == 24) {
		dst[i] = 24;
	} else if (threadIdx.x == 25) {
		dst[i] = 25;
	} else if (threadIdx.x == 26) {
		dst[i] = 26;
	} else if (threadIdx.x == 27) {
		dst[i] = 27;
	} else if (threadIdx.x == 28) {
		dst[i] = 28;
	} else if (threadIdx.x == 29) {
		dst[i] = 29;
	} else if (threadIdx.x == 30) {
		dst[i] = 30;
	} else if (threadIdx.x == 31) {
		dst[i] = 31;
	} else if (threadIdx.x == 32) {
		dst[i] = 32;
	} else if (threadIdx.x == 33) {
		dst[i] = 33;
	} else if (threadIdx.x == 34) {
		dst[i] = 34;
	} else if (threadIdx.x == 35) {
		dst[i] = 35;
	} else if (threadIdx.x == 36) {
		dst[i] = 36;
	} else if (threadIdx.x == 37) {
		dst[i] = 37;
	} else if (threadIdx.x == 38) {
		dst[i] = 38;
	} else if (threadIdx.x == 39) {
		dst[i] = 39;
	} else if (threadIdx.x == 40) {
		dst[i] = 40;
	} else if (threadIdx.x == 41) {
		dst[i] = 41;
	} else if (threadIdx.x == 42) {
		dst[i] = 42;
	} else if (threadIdx.x == 43) {
		dst[i] = 43;
	} else if (threadIdx.x == 44) {
		dst[i] = 44;
	} else if (threadIdx.x == 45) {
		dst[i] = 45;
	} else if (threadIdx.x == 46) {
		dst[i] = 46;
	} else if (threadIdx.x == 47) {
		dst[i] = 47;
	} else if (threadIdx.x == 48) {
		dst[i] = 48;
	} else if (threadIdx.x == 49) {
		dst[i] = 49;
	} else if (threadIdx.x == 50) {
		dst[i] = 50;
	} else if (threadIdx.x == 51) {
		dst[i] = 51;
	} else if (threadIdx.x == 52) {
		dst[i] = 52;
	} else if (threadIdx.x == 53) {
		dst[i] = 53;
	} else if (threadIdx.x == 54) {
		dst[i] = 54;
	} else if (threadIdx.x == 55) {
		dst[i] = 55;
	} else if (threadIdx.x == 56) {
		dst[i] = 56;
	} else if (threadIdx.x == 57) {
		dst[i] = 57;
	} else if (threadIdx.x == 58) {
		dst[i] = 58;
	} else if (threadIdx.x == 59) {
		dst[i] = 59;
	} else if (threadIdx.x == 60) {
		dst[i] = 60;
	} else if (threadIdx.x == 61) {
		dst[i] = 61;
	} else if (threadIdx.x == 62) {
		dst[i] = 62;
	} else if (threadIdx.x == 63) {
		dst[i] = 63;
	} else if (threadIdx.x == 64) {
		dst[i] = 64;
	} else if (threadIdx.x == 65) {
		dst[i] = 65;
	} else if (threadIdx.x == 66) {
		dst[i] = 66;
	} else if (threadIdx.x == 67) {
		dst[i] = 67;
	} else if (threadIdx.x == 68) {
		dst[i] = 68;
	} else if (threadIdx.x == 69) {
		dst[i] = 69;
	} else if (threadIdx.x == 70) {
		dst[i] = 70;
	} else if (threadIdx.x == 71) {
		dst[i] = 71;
	} else if (threadIdx.x == 72) {
		dst[i] = 72;
	} else if (threadIdx.x == 73) {
		dst[i] = 73;
	} else if (threadIdx.x == 74) {
		dst[i] = 74;
	} else if (threadIdx.x == 75) {
		dst[i] = 75;
	} else if (threadIdx.x == 76) {
		dst[i] = 76;
	} else if (threadIdx.x == 77) {
		dst[i] = 77;
	} else if (threadIdx.x == 78) {
		dst[i] = 78;
	} else if (threadIdx.x == 79) {
		dst[i] = 79;
	} else if (threadIdx.x == 80) {
		dst[i] = 80;
	} else if (threadIdx.x == 81) {
		dst[i] = 81;
	} else if (threadIdx.x == 82) {
		dst[i] = 82;
	} else if (threadIdx.x == 83) {
		dst[i] = 83;
	} else if (threadIdx.x == 84) {
		dst[i] = 84;
	} else if (threadIdx.x == 85) {
		dst[i] = 85;
	} else if (threadIdx.x == 86) {
		dst[i] = 86;
	} else if (threadIdx.x == 87) {
		dst[i] = 87;
	} else if (threadIdx.x == 88) {
		dst[i] = 88;
	} else if (threadIdx.x == 89) {
		dst[i] = 89;
	} else if (threadIdx.x == 90) {
		dst[i] = 90;
	} else if (threadIdx.x == 91) {
		dst[i] = 91;
	} else if (threadIdx.x == 92) {
		dst[i] = 92;
	} else if (threadIdx.x == 93) {
		dst[i] = 93;
	} else if (threadIdx.x == 94) {
		dst[i] = 94;
	} else if (threadIdx.x == 95) {
		dst[i] = 95;
	} else if (threadIdx.x == 96) {
		dst[i] = 96;
	} else if (threadIdx.x == 97) {
		dst[i] = 97;
	} else if (threadIdx.x == 98) {
		dst[i] = 98;
	} else if (threadIdx.x == 99) {
		dst[i] = 99;
	} else if (threadIdx.x == 100) {
		dst[i] = 100;
	} else if (threadIdx.x == 101) {
		dst[i] = 101;
	} else if (threadIdx.x == 102) {
		dst[i] = 102;
	} else if (threadIdx.x == 103) {
		dst[i] = 103;
	} else if (threadIdx.x == 104) {
		dst[i] = 104;
	} else if (threadIdx.x == 105) {
		dst[i] = 105;
	} else if (threadIdx.x == 106) {
		dst[i] = 106;
	} else if (threadIdx.x == 107) {
		dst[i] = 107;
	} else if (threadIdx.x == 108) {
		dst[i] = 108;
	} else if (threadIdx.x == 109) {
		dst[i] = 109;
	} else if (threadIdx.x == 110) {
		dst[i] = 110;
	} else if (threadIdx.x == 111) {
		dst[i] = 111;
	} else if (threadIdx.x == 112) {
		dst[i] = 112;
	} else if (threadIdx.x == 113) {
		dst[i] = 113;
	} else if (threadIdx.x == 114) {
		dst[i] = 114;
	} else if (threadIdx.x == 115) {
		dst[i] = 115;
	} else if (threadIdx.x == 116) {
		dst[i] = 116;
	} else if (threadIdx.x == 117) {
		dst[i] = 117;
	} else if (threadIdx.x == 118) {
		dst[i] = 118;
	} else if (threadIdx.x == 119) {
		dst[i] = 119;
	} else if (threadIdx.x == 120) {
		dst[i] = 120;
	} else if (threadIdx.x == 121) {
		dst[i] = 121;
	} else if (threadIdx.x == 122) {
		dst[i] = 122;
	} else if (threadIdx.x == 123) {
		dst[i] = 123;
	} else if (threadIdx.x == 124) {
		dst[i] = 124;
	} else if (threadIdx.x == 125) {
		dst[i] = 125;
	} else if (threadIdx.x == 126) {
		dst[i] = 126;
	} else if (threadIdx.x == 127) {
		dst[i] = 127;
	} else if (threadIdx.x == 128) {
		dst[i] = 128;
	} else if (threadIdx.x == 129) {
		dst[i] = 129;
	} else if (threadIdx.x == 130) {
		dst[i] = 130;
	} else if (threadIdx.x == 131) {
		dst[i] = 131;
	} else if (threadIdx.x == 132) {
		dst[i] = 132;
	} else if (threadIdx.x == 133) {
		dst[i] = 133;
	} else if (threadIdx.x == 134) {
		dst[i] = 134;
	} else if (threadIdx.x == 135) {
		dst[i] = 135;
	} else if (threadIdx.x == 136) {
		dst[i] = 136;
	} else if (threadIdx.x == 137) {
		dst[i] = 137;
	} else if (threadIdx.x == 138) {
		dst[i] = 138;
	} else if (threadIdx.x == 139) {
		dst[i] = 139;
	} else if (threadIdx.x == 140) {
		dst[i] = 140;
	} else if (threadIdx.x == 141) {
		dst[i] = 141;
	} else if (threadIdx.x == 142) {
		dst[i] = 142;
	} else if (threadIdx.x == 143) {
		dst[i] = 143;
	} else if (threadIdx.x == 144) {
		dst[i] = 144;
	} else if (threadIdx.x == 145) {
		dst[i] = 145;
	} else if (threadIdx.x == 146) {
		dst[i] = 146;
	} else if (threadIdx.x == 147) {
		dst[i] = 147;
	} else if (threadIdx.x == 148) {
		dst[i] = 148;
	} else if (threadIdx.x == 149) {
		dst[i] = 149;
	} else if (threadIdx.x == 150) {
		dst[i] = 150;
	} else if (threadIdx.x == 151) {
		dst[i] = 151;
	} else if (threadIdx.x == 152) {
		dst[i] = 152;
	} else if (threadIdx.x == 153) {
		dst[i] = 153;
	} else if (threadIdx.x == 154) {
		dst[i] = 154;
	} else if (threadIdx.x == 155) {
		dst[i] = 155;
	} else if (threadIdx.x == 156) {
		dst[i] = 156;
	} else if (threadIdx.x == 157) {
		dst[i] = 157;
	} else if (threadIdx.x == 158) {
		dst[i] = 158;
	} else if (threadIdx.x == 159) {
		dst[i] = 159;
	} else if (threadIdx.x == 160) {
		dst[i] = 160;
	} else if (threadIdx.x == 161) {
		dst[i] = 161;
	} else if (threadIdx.x == 162) {
		dst[i] = 162;
	} else if (threadIdx.x == 163) {
		dst[i] = 163;
	} else if (threadIdx.x == 164) {
		dst[i] = 164;
	} else if (threadIdx.x == 165) {
		dst[i] = 165;
	} else if (threadIdx.x == 166) {
		dst[i] = 166;
	} else if (threadIdx.x == 167) {
		dst[i] = 167;
	} else if (threadIdx.x == 168) {
		dst[i] = 168;
	} else if (threadIdx.x == 169) {
		dst[i] = 169;
	} else if (threadIdx.x == 170) {
		dst[i] = 170;
	} else if (threadIdx.x == 171) {
		dst[i] = 171;
	} else if (threadIdx.x == 172) {
		dst[i] = 172;
	} else if (threadIdx.x == 173) {
		dst[i] = 173;
	} else if (threadIdx.x == 174) {
		dst[i] = 174;
	} else if (threadIdx.x == 175) {
		dst[i] = 175;
	} else if (threadIdx.x == 176) {
		dst[i] = 176;
	} else if (threadIdx.x == 177) {
		dst[i] = 177;
	} else if (threadIdx.x == 178) {
		dst[i] = 178;
	} else if (threadIdx.x == 179) {
		dst[i] = 179;
	} else if (threadIdx.x == 180) {
		dst[i] = 180;
	} else if (threadIdx.x == 181) {
		dst[i] = 181;
	} else if (threadIdx.x == 182) {
		dst[i] = 182;
	} else if (threadIdx.x == 183) {
		dst[i] = 183;
	} else if (threadIdx.x == 184) {
		dst[i] = 184;
	} else if (threadIdx.x == 185) {
		dst[i] = 185;
	} else if (threadIdx.x == 186) {
		dst[i] = 186;
	} else if (threadIdx.x == 187) {
		dst[i] = 187;
	} else if (threadIdx.x == 188) {
		dst[i] = 188;
	} else if (threadIdx.x == 189) {
		dst[i] = 189;
	} else if (threadIdx.x == 190) {
		dst[i] = 190;
	} else if (threadIdx.x == 191) {
		dst[i] = 191;
	} else if (threadIdx.x == 192) {
		dst[i] = 192;
	} else if (threadIdx.x == 193) {
		dst[i] = 193;
	} else if (threadIdx.x == 194) {
		dst[i] = 194;
	} else if (threadIdx.x == 195) {
		dst[i] = 195;
	} else if (threadIdx.x == 196) {
		dst[i] = 196;
	} else if (threadIdx.x == 197) {
		dst[i] = 197;
	} else if (threadIdx.x == 198) {
		dst[i] = 198;
	} else if (threadIdx.x == 199) {
		dst[i] = 199;
	} else if (threadIdx.x == 200) {
		dst[i] = 200;
	} else if (threadIdx.x == 201) {
		dst[i] = 201;
	} else if (threadIdx.x == 202) {
		dst[i] = 202;
	} else if (threadIdx.x == 203) {
		dst[i] = 203;
	} else if (threadIdx.x == 204) {
		dst[i] = 204;
	} else if (threadIdx.x == 205) {
		dst[i] = 205;
	} else if (threadIdx.x == 206) {
		dst[i] = 206;
	} else if (threadIdx.x == 207) {
		dst[i] = 207;
	} else if (threadIdx.x == 208) {
		dst[i] = 208;
	} else if (threadIdx.x == 209) {
		dst[i] = 209;
	} else if (threadIdx.x == 210) {
		dst[i] = 210;
	} else if (threadIdx.x == 211) {
		dst[i] = 211;
	} else if (threadIdx.x == 212) {
		dst[i] = 212;
	} else if (threadIdx.x == 213) {
		dst[i] = 213;
	} else if (threadIdx.x == 214) {
		dst[i] = 214;
	} else if (threadIdx.x == 215) {
		dst[i] = 215;
	} else if (threadIdx.x == 216) {
		dst[i] = 216;
	} else if (threadIdx.x == 217) {
		dst[i] = 217;
	} else if (threadIdx.x == 218) {
		dst[i] = 218;
	} else if (threadIdx.x == 219) {
		dst[i] = 219;
	} else if (threadIdx.x == 220) {
		dst[i] = 220;
	} else if (threadIdx.x == 221) {
		dst[i] = 221;
	} else if (threadIdx.x == 222) {
		dst[i] = 222;
	} else if (threadIdx.x == 223) {
		dst[i] = 223;
	} else if (threadIdx.x == 224) {
		dst[i] = 224;
	} else if (threadIdx.x == 225) {
		dst[i] = 225;
	} else if (threadIdx.x == 226) {
		dst[i] = 226;
	} else if (threadIdx.x == 227) {
		dst[i] = 227;
	} else if (threadIdx.x == 228) {
		dst[i] = 228;
	} else if (threadIdx.x == 229) {
		dst[i] = 229;
	} else if (threadIdx.x == 230) {
		dst[i] = 230;
	} else if (threadIdx.x == 231) {
		dst[i] = 231;
	} else if (threadIdx.x == 232) {
		dst[i] = 232;
	} else if (threadIdx.x == 233) {
		dst[i] = 233;
	} else if (threadIdx.x == 234) {
		dst[i] = 234;
	} else if (threadIdx.x == 235) {
		dst[i] = 235;
	} else if (threadIdx.x == 236) {
		dst[i] = 236;
	} else if (threadIdx.x == 237) {
		dst[i] = 237;
	} else if (threadIdx.x == 238) {
		dst[i] = 238;
	} else if (threadIdx.x == 239) {
		dst[i] = 239;
	} else if (threadIdx.x == 240) {
		dst[i] = 240;
	} else if (threadIdx.x == 241) {
		dst[i] = 241;
	} else if (threadIdx.x == 242) {
		dst[i] = 242;
	} else if (threadIdx.x == 243) {
		dst[i] = 243;
	} else if (threadIdx.x == 244) {
		dst[i] = 244;
	} else if (threadIdx.x == 245) {
		dst[i] = 245;
	} else if (threadIdx.x == 246) {
		dst[i] = 246;
	} else if (threadIdx.x == 247) {
		dst[i] = 247;
	} else if (threadIdx.x == 248) {
		dst[i] = 248;
	} else if (threadIdx.x == 249) {
		dst[i] = 249;
	} else if (threadIdx.x == 250) {
		dst[i] = 250;
	} else if (threadIdx.x == 251) {
		dst[i] = 251;
	} else if (threadIdx.x == 252) {
		dst[i] = 252;
	} else if (threadIdx.x == 253) {
		dst[i] = 253;
	} else if (threadIdx.x == 254) {
		dst[i] = 254;
	} else if (threadIdx.x == 255) {
		dst[i] = 255;
	} else if (threadIdx.x == 256) {
		dst[i] = 256;
	} else if (threadIdx.x == 257) {
		dst[i] = 257;
	} else if (threadIdx.x == 258) {
		dst[i] = 258;
	} else if (threadIdx.x == 259) {
		dst[i] = 259;
	} else if (threadIdx.x == 260) {
		dst[i] = 260;
	} else if (threadIdx.x == 261) {
		dst[i] = 261;
	} else if (threadIdx.x == 262) {
		dst[i] = 262;
	} else if (threadIdx.x == 263) {
		dst[i] = 263;
	} else if (threadIdx.x == 264) {
		dst[i] = 264;
	} else if (threadIdx.x == 265) {
		dst[i] = 265;
	} else if (threadIdx.x == 266) {
		dst[i] = 266;
	} else if (threadIdx.x == 267) {
		dst[i] = 267;
	} else if (threadIdx.x == 268) {
		dst[i] = 268;
	} else if (threadIdx.x == 269) {
		dst[i] = 269;
	} else if (threadIdx.x == 270) {
		dst[i] = 270;
	} else if (threadIdx.x == 271) {
		dst[i] = 271;
	} else if (threadIdx.x == 272) {
		dst[i] = 272;
	} else if (threadIdx.x == 273) {
		dst[i] = 273;
	} else if (threadIdx.x == 274) {
		dst[i] = 274;
	} else if (threadIdx.x == 275) {
		dst[i] = 275;
	} else if (threadIdx.x == 276) {
		dst[i] = 276;
	} else if (threadIdx.x == 277) {
		dst[i] = 277;
	} else if (threadIdx.x == 278) {
		dst[i] = 278;
	} else if (threadIdx.x == 279) {
		dst[i] = 279;
	} else if (threadIdx.x == 280) {
		dst[i] = 280;
	} else if (threadIdx.x == 281) {
		dst[i] = 281;
	} else if (threadIdx.x == 282) {
		dst[i] = 282;
	} else if (threadIdx.x == 283) {
		dst[i] = 283;
	} else if (threadIdx.x == 284) {
		dst[i] = 284;
	} else if (threadIdx.x == 285) {
		dst[i] = 285;
	} else if (threadIdx.x == 286) {
		dst[i] = 286;
	} else if (threadIdx.x == 287) {
		dst[i] = 287;
	} else if (threadIdx.x == 288) {
		dst[i] = 288;
	} else if (threadIdx.x == 289) {
		dst[i] = 289;
	} else if (threadIdx.x == 290) {
		dst[i] = 290;
	} else if (threadIdx.x == 291) {
		dst[i] = 291;
	} else if (threadIdx.x == 292) {
		dst[i] = 292;
	} else if (threadIdx.x == 293) {
		dst[i] = 293;
	} else if (threadIdx.x == 294) {
		dst[i] = 294;
	} else if (threadIdx.x == 295) {
		dst[i] = 295;
	} else if (threadIdx.x == 296) {
		dst[i] = 296;
	} else if (threadIdx.x == 297) {
		dst[i] = 297;
	} else if (threadIdx.x == 298) {
		dst[i] = 298;
	} else if (threadIdx.x == 299) {
		dst[i] = 299;
	} else if (threadIdx.x == 300) {
		dst[i] = 300;
	} else if (threadIdx.x == 301) {
		dst[i] = 301;
	} else if (threadIdx.x == 302) {
		dst[i] = 302;
	} else if (threadIdx.x == 303) {
		dst[i] = 303;
	} else if (threadIdx.x == 304) {
		dst[i] = 304;
	} else if (threadIdx.x == 305) {
		dst[i] = 305;
	} else if (threadIdx.x == 306) {
		dst[i] = 306;
	} else if (threadIdx.x == 307) {
		dst[i] = 307;
	} else if (threadIdx.x == 308) {
		dst[i] = 308;
	} else if (threadIdx.x == 309) {
		dst[i] = 309;
	} else if (threadIdx.x == 310) {
		dst[i] = 310;
	} else if (threadIdx.x == 311) {
		dst[i] = 311;
	} else if (threadIdx.x == 312) {
		dst[i] = 312;
	} else if (threadIdx.x == 313) {
		dst[i] = 313;
	} else if (threadIdx.x == 314) {
		dst[i] = 314;
	} else if (threadIdx.x == 315) {
		dst[i] = 315;
	} else if (threadIdx.x == 316) {
		dst[i] = 316;
	} else if (threadIdx.x == 317) {
		dst[i] = 317;
	} else if (threadIdx.x == 318) {
		dst[i] = 318;
	} else if (threadIdx.x == 319) {
		dst[i] = 319;
	} else if (threadIdx.x == 320) {
		dst[i] = 320;
	} else if (threadIdx.x == 321) {
		dst[i] = 321;
	} else if (threadIdx.x == 322) {
		dst[i] = 322;
	} else if (threadIdx.x == 323) {
		dst[i] = 323;
	} else if (threadIdx.x == 324) {
		dst[i] = 324;
	} else if (threadIdx.x == 325) {
		dst[i] = 325;
	} else if (threadIdx.x == 326) {
		dst[i] = 326;
	} else if (threadIdx.x == 327) {
		dst[i] = 327;
	} else if (threadIdx.x == 328) {
		dst[i] = 328;
	} else if (threadIdx.x == 329) {
		dst[i] = 329;
	} else if (threadIdx.x == 330) {
		dst[i] = 330;
	} else if (threadIdx.x == 331) {
		dst[i] = 331;
	} else if (threadIdx.x == 332) {
		dst[i] = 332;
	} else if (threadIdx.x == 333) {
		dst[i] = 333;
	} else if (threadIdx.x == 334) {
		dst[i] = 334;
	} else if (threadIdx.x == 335) {
		dst[i] = 335;
	} else if (threadIdx.x == 336) {
		dst[i] = 336;
	} else if (threadIdx.x == 337) {
		dst[i] = 337;
	} else if (threadIdx.x == 338) {
		dst[i] = 338;
	} else if (threadIdx.x == 339) {
		dst[i] = 339;
	} else if (threadIdx.x == 340) {
		dst[i] = 340;
	} else if (threadIdx.x == 341) {
		dst[i] = 341;
	} else if (threadIdx.x == 342) {
		dst[i] = 342;
	} else if (threadIdx.x == 343) {
		dst[i] = 343;
	} else if (threadIdx.x == 344) {
		dst[i] = 344;
	} else if (threadIdx.x == 345) {
		dst[i] = 345;
	} else if (threadIdx.x == 346) {
		dst[i] = 346;
	} else if (threadIdx.x == 347) {
		dst[i] = 347;
	} else if (threadIdx.x == 348) {
		dst[i] = 348;
	} else if (threadIdx.x == 349) {
		dst[i] = 349;
	} else if (threadIdx.x == 350) {
		dst[i] = 350;
	} else if (threadIdx.x == 351) {
		dst[i] = 351;
	} else if (threadIdx.x == 352) {
		dst[i] = 352;
	} else if (threadIdx.x == 353) {
		dst[i] = 353;
	} else if (threadIdx.x == 354) {
		dst[i] = 354;
	} else if (threadIdx.x == 355) {
		dst[i] = 355;
	} else if (threadIdx.x == 356) {
		dst[i] = 356;
	} else if (threadIdx.x == 357) {
		dst[i] = 357;
	} else if (threadIdx.x == 358) {
		dst[i] = 358;
	} else if (threadIdx.x == 359) {
		dst[i] = 359;
	} else if (threadIdx.x == 360) {
		dst[i] = 360;
	} else if (threadIdx.x == 361) {
		dst[i] = 361;
	} else if (threadIdx.x == 362) {
		dst[i] = 362;
	} else if (threadIdx.x == 363) {
		dst[i] = 363;
	} else if (threadIdx.x == 364) {
		dst[i] = 364;
	} else if (threadIdx.x == 365) {
		dst[i] = 365;
	} else if (threadIdx.x == 366) {
		dst[i] = 366;
	} else if (threadIdx.x == 367) {
		dst[i] = 367;
	} else if (threadIdx.x == 368) {
		dst[i] = 368;
	} else if (threadIdx.x == 369) {
		dst[i] = 369;
	} else if (threadIdx.x == 370) {
		dst[i] = 370;
	} else if (threadIdx.x == 371) {
		dst[i] = 371;
	} else if (threadIdx.x == 372) {
		dst[i] = 372;
	} else if (threadIdx.x == 373) {
		dst[i] = 373;
	} else if (threadIdx.x == 374) {
		dst[i] = 374;
	} else if (threadIdx.x == 375) {
		dst[i] = 375;
	} else if (threadIdx.x == 376) {
		dst[i] = 376;
	} else if (threadIdx.x == 377) {
		dst[i] = 377;
	} else if (threadIdx.x == 378) {
		dst[i] = 378;
	} else if (threadIdx.x == 379) {
		dst[i] = 379;
	} else if (threadIdx.x == 380) {
		dst[i] = 380;
	} else if (threadIdx.x == 381) {
		dst[i] = 381;
	} else if (threadIdx.x == 382) {
		dst[i] = 382;
	} else if (threadIdx.x == 383) {
		dst[i] = 383;
	} else if (threadIdx.x == 384) {
		dst[i] = 384;
	} else if (threadIdx.x == 385) {
		dst[i] = 385;
	} else if (threadIdx.x == 386) {
		dst[i] = 386;
	} else if (threadIdx.x == 387) {
		dst[i] = 387;
	} else if (threadIdx.x == 388) {
		dst[i] = 388;
	} else if (threadIdx.x == 389) {
		dst[i] = 389;
	} else if (threadIdx.x == 390) {
		dst[i] = 390;
	} else if (threadIdx.x == 391) {
		dst[i] = 391;
	} else if (threadIdx.x == 392) {
		dst[i] = 392;
	} else if (threadIdx.x == 393) {
		dst[i] = 393;
	} else if (threadIdx.x == 394) {
		dst[i] = 394;
	} else if (threadIdx.x == 395) {
		dst[i] = 395;
	} else if (threadIdx.x == 396) {
		dst[i] = 396;
	} else if (threadIdx.x == 397) {
		dst[i] = 397;
	} else if (threadIdx.x == 398) {
		dst[i] = 398;
	} else if (threadIdx.x == 399) {
		dst[i] = 399;
	} else if (threadIdx.x == 400) {
		dst[i] = 400;
	} else if (threadIdx.x == 401) {
		dst[i] = 401;
	} else if (threadIdx.x == 402) {
		dst[i] = 402;
	} else if (threadIdx.x == 403) {
		dst[i] = 403;
	} else if (threadIdx.x == 404) {
		dst[i] = 404;
	} else if (threadIdx.x == 405) {
		dst[i] = 405;
	} else if (threadIdx.x == 406) {
		dst[i] = 406;
	} else if (threadIdx.x == 407) {
		dst[i] = 407;
	} else if (threadIdx.x == 408) {
		dst[i] = 408;
	} else if (threadIdx.x == 409) {
		dst[i] = 409;
	} else if (threadIdx.x == 410) {
		dst[i] = 410;
	} else if (threadIdx.x == 411) {
		dst[i] = 411;
	} else if (threadIdx.x == 412) {
		dst[i] = 412;
	} else if (threadIdx.x == 413) {
		dst[i] = 413;
	} else if (threadIdx.x == 414) {
		dst[i] = 414;
	} else if (threadIdx.x == 415) {
		dst[i] = 415;
	} else if (threadIdx.x == 416) {
		dst[i] = 416;
	} else if (threadIdx.x == 417) {
		dst[i] = 417;
	} else if (threadIdx.x == 418) {
		dst[i] = 418;
	} else if (threadIdx.x == 419) {
		dst[i] = 419;
	} else if (threadIdx.x == 420) {
		dst[i] = 420;
	} else if (threadIdx.x == 421) {
		dst[i] = 421;
	} else if (threadIdx.x == 422) {
		dst[i] = 422;
	} else if (threadIdx.x == 423) {
		dst[i] = 423;
	} else if (threadIdx.x == 424) {
		dst[i] = 424;
	} else if (threadIdx.x == 425) {
		dst[i] = 425;
	} else if (threadIdx.x == 426) {
		dst[i] = 426;
	} else if (threadIdx.x == 427) {
		dst[i] = 427;
	} else if (threadIdx.x == 428) {
		dst[i] = 428;
	} else if (threadIdx.x == 429) {
		dst[i] = 429;
	} else if (threadIdx.x == 430) {
		dst[i] = 430;
	} else if (threadIdx.x == 431) {
		dst[i] = 431;
	} else if (threadIdx.x == 432) {
		dst[i] = 432;
	} else if (threadIdx.x == 433) {
		dst[i] = 433;
	} else if (threadIdx.x == 434) {
		dst[i] = 434;
	} else if (threadIdx.x == 435) {
		dst[i] = 435;
	} else if (threadIdx.x == 436) {
		dst[i] = 436;
	} else if (threadIdx.x == 437) {
		dst[i] = 437;
	} else if (threadIdx.x == 438) {
		dst[i] = 438;
	} else if (threadIdx.x == 439) {
		dst[i] = 439;
	} else if (threadIdx.x == 440) {
		dst[i] = 440;
	} else if (threadIdx.x == 441) {
		dst[i] = 441;
	} else if (threadIdx.x == 442) {
		dst[i] = 442;
	} else if (threadIdx.x == 443) {
		dst[i] = 443;
	} else if (threadIdx.x == 444) {
		dst[i] = 444;
	} else if (threadIdx.x == 445) {
		dst[i] = 445;
	} else if (threadIdx.x == 446) {
		dst[i] = 446;
	} else if (threadIdx.x == 447) {
		dst[i] = 447;
	} else if (threadIdx.x == 448) {
		dst[i] = 448;
	} else if (threadIdx.x == 449) {
		dst[i] = 449;
	} else if (threadIdx.x == 450) {
		dst[i] = 450;
	} else if (threadIdx.x == 451) {
		dst[i] = 451;
	} else if (threadIdx.x == 452) {
		dst[i] = 452;
	} else if (threadIdx.x == 453) {
		dst[i] = 453;
	} else if (threadIdx.x == 454) {
		dst[i] = 454;
	} else if (threadIdx.x == 455) {
		dst[i] = 455;
	} else if (threadIdx.x == 456) {
		dst[i] = 456;
	} else if (threadIdx.x == 457) {
		dst[i] = 457;
	} else if (threadIdx.x == 458) {
		dst[i] = 458;
	} else if (threadIdx.x == 459) {
		dst[i] = 459;
	} else if (threadIdx.x == 460) {
		dst[i] = 460;
	} else if (threadIdx.x == 461) {
		dst[i] = 461;
	} else if (threadIdx.x == 462) {
		dst[i] = 462;
	} else if (threadIdx.x == 463) {
		dst[i] = 463;
	} else if (threadIdx.x == 464) {
		dst[i] = 464;
	} else if (threadIdx.x == 465) {
		dst[i] = 465;
	} else if (threadIdx.x == 466) {
		dst[i] = 466;
	} else if (threadIdx.x == 467) {
		dst[i] = 467;
	} else if (threadIdx.x == 468) {
		dst[i] = 468;
	} else if (threadIdx.x == 469) {
		dst[i] = 469;
	} else if (threadIdx.x == 470) {
		dst[i] = 470;
	} else if (threadIdx.x == 471) {
		dst[i] = 471;
	} else if (threadIdx.x == 472) {
		dst[i] = 472;
	} else if (threadIdx.x == 473) {
		dst[i] = 473;
	} else if (threadIdx.x == 474) {
		dst[i] = 474;
	} else if (threadIdx.x == 475) {
		dst[i] = 475;
	} else if (threadIdx.x == 476) {
		dst[i] = 476;
	} else if (threadIdx.x == 477) {
		dst[i] = 477;
	} else if (threadIdx.x == 478) {
		dst[i] = 478;
	} else if (threadIdx.x == 479) {
		dst[i] = 479;
	} else if (threadIdx.x == 480) {
		dst[i] = 480;
	} else if (threadIdx.x == 481) {
		dst[i] = 481;
	} else if (threadIdx.x == 482) {
		dst[i] = 482;
	} else if (threadIdx.x == 483) {
		dst[i] = 483;
	} else if (threadIdx.x == 484) {
		dst[i] = 484;
	} else if (threadIdx.x == 485) {
		dst[i] = 485;
	} else if (threadIdx.x == 486) {
		dst[i] = 486;
	} else if (threadIdx.x == 487) {
		dst[i] = 487;
	} else if (threadIdx.x == 488) {
		dst[i] = 488;
	} else if (threadIdx.x == 489) {
		dst[i] = 489;
	} else if (threadIdx.x == 490) {
		dst[i] = 490;
	} else if (threadIdx.x == 491) {
		dst[i] = 491;
	} else if (threadIdx.x == 492) {
		dst[i] = 492;
	} else if (threadIdx.x == 493) {
		dst[i] = 493;
	} else if (threadIdx.x == 494) {
		dst[i] = 494;
	} else if (threadIdx.x == 495) {
		dst[i] = 495;
	} else if (threadIdx.x == 496) {
		dst[i] = 496;
	} else if (threadIdx.x == 497) {
		dst[i] = 497;
	} else if (threadIdx.x == 498) {
		dst[i] = 498;
	} else if (threadIdx.x == 499) {
		dst[i] = 499;
	} else if (threadIdx.x == 500) {
		dst[i] = 500;
	} else if (threadIdx.x == 501) {
		dst[i] = 501;
	} else if (threadIdx.x == 502) {
		dst[i] = 502;
	} else if (threadIdx.x == 503) {
		dst[i] = 503;
	} else if (threadIdx.x == 504) {
		dst[i] = 504;
	} else if (threadIdx.x == 505) {
		dst[i] = 505;
	} else if (threadIdx.x == 506) {
		dst[i] = 506;
	} else if (threadIdx.x == 507) {
		dst[i] = 507;
	} else if (threadIdx.x == 508) {
		dst[i] = 508;
	} else if (threadIdx.x == 509) {
		dst[i] = 509;
	} else if (threadIdx.x == 510) {
		dst[i] = 510;
	} else if (threadIdx.x == 511) {
		dst[i] = 511;
	} else if (threadIdx.x == 512) {
		dst[i] = 512;
	} else if (threadIdx.x == 513) {
		dst[i] = 513;
	} else if (threadIdx.x == 514) {
		dst[i] = 514;
	} else if (threadIdx.x == 515) {
		dst[i] = 515;
	} else if (threadIdx.x == 516) {
		dst[i] = 516;
	} else if (threadIdx.x == 517) {
		dst[i] = 517;
	} else if (threadIdx.x == 518) {
		dst[i] = 518;
	} else if (threadIdx.x == 519) {
		dst[i] = 519;
	} else if (threadIdx.x == 520) {
		dst[i] = 520;
	} else if (threadIdx.x == 521) {
		dst[i] = 521;
	} else if (threadIdx.x == 522) {
		dst[i] = 522;
	} else if (threadIdx.x == 523) {
		dst[i] = 523;
	} else if (threadIdx.x == 524) {
		dst[i] = 524;
	} else if (threadIdx.x == 525) {
		dst[i] = 525;
	} else if (threadIdx.x == 526) {
		dst[i] = 526;
	} else if (threadIdx.x == 527) {
		dst[i] = 527;
	} else if (threadIdx.x == 528) {
		dst[i] = 528;
	} else if (threadIdx.x == 529) {
		dst[i] = 529;
	} else if (threadIdx.x == 530) {
		dst[i] = 530;
	} else if (threadIdx.x == 531) {
		dst[i] = 531;
	} else if (threadIdx.x == 532) {
		dst[i] = 532;
	} else if (threadIdx.x == 533) {
		dst[i] = 533;
	} else if (threadIdx.x == 534) {
		dst[i] = 534;
	} else if (threadIdx.x == 535) {
		dst[i] = 535;
	} else if (threadIdx.x == 536) {
		dst[i] = 536;
	} else if (threadIdx.x == 537) {
		dst[i] = 537;
	} else if (threadIdx.x == 538) {
		dst[i] = 538;
	} else if (threadIdx.x == 539) {
		dst[i] = 539;
	} else if (threadIdx.x == 540) {
		dst[i] = 540;
	} else if (threadIdx.x == 541) {
		dst[i] = 541;
	} else if (threadIdx.x == 542) {
		dst[i] = 542;
	} else if (threadIdx.x == 543) {
		dst[i] = 543;
	} else if (threadIdx.x == 544) {
		dst[i] = 544;
	} else if (threadIdx.x == 545) {
		dst[i] = 545;
	} else if (threadIdx.x == 546) {
		dst[i] = 546;
	} else if (threadIdx.x == 547) {
		dst[i] = 547;
	} else if (threadIdx.x == 548) {
		dst[i] = 548;
	} else if (threadIdx.x == 549) {
		dst[i] = 549;
	} else if (threadIdx.x == 550) {
		dst[i] = 550;
	} else if (threadIdx.x == 551) {
		dst[i] = 551;
	} else if (threadIdx.x == 552) {
		dst[i] = 552;
	} else if (threadIdx.x == 553) {
		dst[i] = 553;
	} else if (threadIdx.x == 554) {
		dst[i] = 554;
	} else if (threadIdx.x == 555) {
		dst[i] = 555;
	} else if (threadIdx.x == 556) {
		dst[i] = 556;
	} else if (threadIdx.x == 557) {
		dst[i] = 557;
	} else if (threadIdx.x == 558) {
		dst[i] = 558;
	} else if (threadIdx.x == 559) {
		dst[i] = 559;
	} else if (threadIdx.x == 560) {
		dst[i] = 560;
	} else if (threadIdx.x == 561) {
		dst[i] = 561;
	} else if (threadIdx.x == 562) {
		dst[i] = 562;
	} else if (threadIdx.x == 563) {
		dst[i] = 563;
	} else if (threadIdx.x == 564) {
		dst[i] = 564;
	} else if (threadIdx.x == 565) {
		dst[i] = 565;
	} else if (threadIdx.x == 566) {
		dst[i] = 566;
	} else if (threadIdx.x == 567) {
		dst[i] = 567;
	} else if (threadIdx.x == 568) {
		dst[i] = 568;
	} else if (threadIdx.x == 569) {
		dst[i] = 569;
	} else if (threadIdx.x == 570) {
		dst[i] = 570;
	} else if (threadIdx.x == 571) {
		dst[i] = 571;
	} else if (threadIdx.x == 572) {
		dst[i] = 572;
	} else if (threadIdx.x == 573) {
		dst[i] = 573;
	} else if (threadIdx.x == 574) {
		dst[i] = 574;
	} else if (threadIdx.x == 575) {
		dst[i] = 575;
	} else if (threadIdx.x == 576) {
		dst[i] = 576;
	} else if (threadIdx.x == 577) {
		dst[i] = 577;
	} else if (threadIdx.x == 578) {
		dst[i] = 578;
	} else if (threadIdx.x == 579) {
		dst[i] = 579;
	} else if (threadIdx.x == 580) {
		dst[i] = 580;
	} else if (threadIdx.x == 581) {
		dst[i] = 581;
	} else if (threadIdx.x == 582) {
		dst[i] = 582;
	} else if (threadIdx.x == 583) {
		dst[i] = 583;
	} else if (threadIdx.x == 584) {
		dst[i] = 584;
	} else if (threadIdx.x == 585) {
		dst[i] = 585;
	} else if (threadIdx.x == 586) {
		dst[i] = 586;
	} else if (threadIdx.x == 587) {
		dst[i] = 587;
	} else if (threadIdx.x == 588) {
		dst[i] = 588;
	} else if (threadIdx.x == 589) {
		dst[i] = 589;
	} else if (threadIdx.x == 590) {
		dst[i] = 590;
	} else if (threadIdx.x == 591) {
		dst[i] = 591;
	} else if (threadIdx.x == 592) {
		dst[i] = 592;
	} else if (threadIdx.x == 593) {
		dst[i] = 593;
	} else if (threadIdx.x == 594) {
		dst[i] = 594;
	} else if (threadIdx.x == 595) {
		dst[i] = 595;
	} else if (threadIdx.x == 596) {
		dst[i] = 596;
	} else if (threadIdx.x == 597) {
		dst[i] = 597;
	} else if (threadIdx.x == 598) {
		dst[i] = 598;
	} else if (threadIdx.x == 599) {
		dst[i] = 599;
	} else if (threadIdx.x == 600) {
		dst[i] = 600;
	} else if (threadIdx.x == 601) {
		dst[i] = 601;
	} else if (threadIdx.x == 602) {
		dst[i] = 602;
	} else if (threadIdx.x == 603) {
		dst[i] = 603;
	} else if (threadIdx.x == 604) {
		dst[i] = 604;
	} else if (threadIdx.x == 605) {
		dst[i] = 605;
	} else if (threadIdx.x == 606) {
		dst[i] = 606;
	} else if (threadIdx.x == 607) {
		dst[i] = 607;
	} else if (threadIdx.x == 608) {
		dst[i] = 608;
	} else if (threadIdx.x == 609) {
		dst[i] = 609;
	} else if (threadIdx.x == 610) {
		dst[i] = 610;
	} else if (threadIdx.x == 611) {
		dst[i] = 611;
	} else if (threadIdx.x == 612) {
		dst[i] = 612;
	} else if (threadIdx.x == 613) {
		dst[i] = 613;
	} else if (threadIdx.x == 614) {
		dst[i] = 614;
	} else if (threadIdx.x == 615) {
		dst[i] = 615;
	} else if (threadIdx.x == 616) {
		dst[i] = 616;
	} else if (threadIdx.x == 617) {
		dst[i] = 617;
	} else if (threadIdx.x == 618) {
		dst[i] = 618;
	} else if (threadIdx.x == 619) {
		dst[i] = 619;
	} else if (threadIdx.x == 620) {
		dst[i] = 620;
	} else if (threadIdx.x == 621) {
		dst[i] = 621;
	} else if (threadIdx.x == 622) {
		dst[i] = 622;
	} else if (threadIdx.x == 623) {
		dst[i] = 623;
	} else if (threadIdx.x == 624) {
		dst[i] = 624;
	} else if (threadIdx.x == 625) {
		dst[i] = 625;
	} else if (threadIdx.x == 626) {
		dst[i] = 626;
	} else if (threadIdx.x == 627) {
		dst[i] = 627;
	} else if (threadIdx.x == 628) {
		dst[i] = 628;
	} else if (threadIdx.x == 629) {
		dst[i] = 629;
	} else if (threadIdx.x == 630) {
		dst[i] = 630;
	} else if (threadIdx.x == 631) {
		dst[i] = 631;
	} else if (threadIdx.x == 632) {
		dst[i] = 632;
	} else if (threadIdx.x == 633) {
		dst[i] = 633;
	} else if (threadIdx.x == 634) {
		dst[i] = 634;
	} else if (threadIdx.x == 635) {
		dst[i] = 635;
	} else if (threadIdx.x == 636) {
		dst[i] = 636;
	} else if (threadIdx.x == 637) {
		dst[i] = 637;
	} else if (threadIdx.x == 638) {
		dst[i] = 638;
	} else if (threadIdx.x == 639) {
		dst[i] = 639;
	} else if (threadIdx.x == 640) {
		dst[i] = 640;
	} else if (threadIdx.x == 641) {
		dst[i] = 641;
	} else if (threadIdx.x == 642) {
		dst[i] = 642;
	} else if (threadIdx.x == 643) {
		dst[i] = 643;
	} else if (threadIdx.x == 644) {
		dst[i] = 644;
	} else if (threadIdx.x == 645) {
		dst[i] = 645;
	} else if (threadIdx.x == 646) {
		dst[i] = 646;
	} else if (threadIdx.x == 647) {
		dst[i] = 647;
	} else if (threadIdx.x == 648) {
		dst[i] = 648;
	} else if (threadIdx.x == 649) {
		dst[i] = 649;
	} else if (threadIdx.x == 650) {
		dst[i] = 650;
	} else if (threadIdx.x == 651) {
		dst[i] = 651;
	} else if (threadIdx.x == 652) {
		dst[i] = 652;
	} else if (threadIdx.x == 653) {
		dst[i] = 653;
	} else if (threadIdx.x == 654) {
		dst[i] = 654;
	} else if (threadIdx.x == 655) {
		dst[i] = 655;
	} else if (threadIdx.x == 656) {
		dst[i] = 656;
	} else if (threadIdx.x == 657) {
		dst[i] = 657;
	} else if (threadIdx.x == 658) {
		dst[i] = 658;
	} else if (threadIdx.x == 659) {
		dst[i] = 659;
	} else if (threadIdx.x == 660) {
		dst[i] = 660;
	} else if (threadIdx.x == 661) {
		dst[i] = 661;
	} else if (threadIdx.x == 662) {
		dst[i] = 662;
	} else if (threadIdx.x == 663) {
		dst[i] = 663;
	} else if (threadIdx.x == 664) {
		dst[i] = 664;
	} else if (threadIdx.x == 665) {
		dst[i] = 665;
	} else if (threadIdx.x == 666) {
		dst[i] = 666;
	} else if (threadIdx.x == 667) {
		dst[i] = 667;
	} else if (threadIdx.x == 668) {
		dst[i] = 668;
	} else if (threadIdx.x == 669) {
		dst[i] = 669;
	} else if (threadIdx.x == 670) {
		dst[i] = 670;
	} else if (threadIdx.x == 671) {
		dst[i] = 671;
	} else if (threadIdx.x == 672) {
		dst[i] = 672;
	} else if (threadIdx.x == 673) {
		dst[i] = 673;
	} else if (threadIdx.x == 674) {
		dst[i] = 674;
	} else if (threadIdx.x == 675) {
		dst[i] = 675;
	} else if (threadIdx.x == 676) {
		dst[i] = 676;
	} else if (threadIdx.x == 677) {
		dst[i] = 677;
	} else if (threadIdx.x == 678) {
		dst[i] = 678;
	} else if (threadIdx.x == 679) {
		dst[i] = 679;
	} else if (threadIdx.x == 680) {
		dst[i] = 680;
	} else if (threadIdx.x == 681) {
		dst[i] = 681;
	} else if (threadIdx.x == 682) {
		dst[i] = 682;
	} else if (threadIdx.x == 683) {
		dst[i] = 683;
	} else if (threadIdx.x == 684) {
		dst[i] = 684;
	} else if (threadIdx.x == 685) {
		dst[i] = 685;
	} else if (threadIdx.x == 686) {
		dst[i] = 686;
	} else if (threadIdx.x == 687) {
		dst[i] = 687;
	} else if (threadIdx.x == 688) {
		dst[i] = 688;
	} else if (threadIdx.x == 689) {
		dst[i] = 689;
	} else if (threadIdx.x == 690) {
		dst[i] = 690;
	} else if (threadIdx.x == 691) {
		dst[i] = 691;
	} else if (threadIdx.x == 692) {
		dst[i] = 692;
	} else if (threadIdx.x == 693) {
		dst[i] = 693;
	} else if (threadIdx.x == 694) {
		dst[i] = 694;
	} else if (threadIdx.x == 695) {
		dst[i] = 695;
	} else if (threadIdx.x == 696) {
		dst[i] = 696;
	} else if (threadIdx.x == 697) {
		dst[i] = 697;
	} else if (threadIdx.x == 698) {
		dst[i] = 698;
	} else if (threadIdx.x == 699) {
		dst[i] = 699;
	} else if (threadIdx.x == 700) {
		dst[i] = 700;
	} else if (threadIdx.x == 701) {
		dst[i] = 701;
	} else if (threadIdx.x == 702) {
		dst[i] = 702;
	} else if (threadIdx.x == 703) {
		dst[i] = 703;
	} else if (threadIdx.x == 704) {
		dst[i] = 704;
	} else if (threadIdx.x == 705) {
		dst[i] = 705;
	} else if (threadIdx.x == 706) {
		dst[i] = 706;
	} else if (threadIdx.x == 707) {
		dst[i] = 707;
	} else if (threadIdx.x == 708) {
		dst[i] = 708;
	} else if (threadIdx.x == 709) {
		dst[i] = 709;
	} else if (threadIdx.x == 710) {
		dst[i] = 710;
	} else if (threadIdx.x == 711) {
		dst[i] = 711;
	} else if (threadIdx.x == 712) {
		dst[i] = 712;
	} else if (threadIdx.x == 713) {
		dst[i] = 713;
	} else if (threadIdx.x == 714) {
		dst[i] = 714;
	} else if (threadIdx.x == 715) {
		dst[i] = 715;
	} else if (threadIdx.x == 716) {
		dst[i] = 716;
	} else if (threadIdx.x == 717) {
		dst[i] = 717;
	} else if (threadIdx.x == 718) {
		dst[i] = 718;
	} else if (threadIdx.x == 719) {
		dst[i] = 719;
	} else if (threadIdx.x == 720) {
		dst[i] = 720;
	} else if (threadIdx.x == 721) {
		dst[i] = 721;
	} else if (threadIdx.x == 722) {
		dst[i] = 722;
	} else if (threadIdx.x == 723) {
		dst[i] = 723;
	} else if (threadIdx.x == 724) {
		dst[i] = 724;
	} else if (threadIdx.x == 725) {
		dst[i] = 725;
	} else if (threadIdx.x == 726) {
		dst[i] = 726;
	} else if (threadIdx.x == 727) {
		dst[i] = 727;
	} else if (threadIdx.x == 728) {
		dst[i] = 728;
	} else if (threadIdx.x == 729) {
		dst[i] = 729;
	} else if (threadIdx.x == 730) {
		dst[i] = 730;
	} else if (threadIdx.x == 731) {
		dst[i] = 731;
	} else if (threadIdx.x == 732) {
		dst[i] = 732;
	} else if (threadIdx.x == 733) {
		dst[i] = 733;
	} else if (threadIdx.x == 734) {
		dst[i] = 734;
	} else if (threadIdx.x == 735) {
		dst[i] = 735;
	} else if (threadIdx.x == 736) {
		dst[i] = 736;
	} else if (threadIdx.x == 737) {
		dst[i] = 737;
	} else if (threadIdx.x == 738) {
		dst[i] = 738;
	} else if (threadIdx.x == 739) {
		dst[i] = 739;
	} else if (threadIdx.x == 740) {
		dst[i] = 740;
	} else if (threadIdx.x == 741) {
		dst[i] = 741;
	} else if (threadIdx.x == 742) {
		dst[i] = 742;
	} else if (threadIdx.x == 743) {
		dst[i] = 743;
	} else if (threadIdx.x == 744) {
		dst[i] = 744;
	} else if (threadIdx.x == 745) {
		dst[i] = 745;
	} else if (threadIdx.x == 746) {
		dst[i] = 746;
	} else if (threadIdx.x == 747) {
		dst[i] = 747;
	} else if (threadIdx.x == 748) {
		dst[i] = 748;
	} else if (threadIdx.x == 749) {
		dst[i] = 749;
	} else if (threadIdx.x == 750) {
		dst[i] = 750;
	} else if (threadIdx.x == 751) {
		dst[i] = 751;
	} else if (threadIdx.x == 752) {
		dst[i] = 752;
	} else if (threadIdx.x == 753) {
		dst[i] = 753;
	} else if (threadIdx.x == 754) {
		dst[i] = 754;
	} else if (threadIdx.x == 755) {
		dst[i] = 755;
	} else if (threadIdx.x == 756) {
		dst[i] = 756;
	} else if (threadIdx.x == 757) {
		dst[i] = 757;
	} else if (threadIdx.x == 758) {
		dst[i] = 758;
	} else if (threadIdx.x == 759) {
		dst[i] = 759;
	} else if (threadIdx.x == 760) {
		dst[i] = 760;
	} else if (threadIdx.x == 761) {
		dst[i] = 761;
	} else if (threadIdx.x == 762) {
		dst[i] = 762;
	} else if (threadIdx.x == 763) {
		dst[i] = 763;
	} else if (threadIdx.x == 764) {
		dst[i] = 764;
	} else if (threadIdx.x == 765) {
		dst[i] = 765;
	} else if (threadIdx.x == 766) {
		dst[i] = 766;
	} else if (threadIdx.x == 767) {
		dst[i] = 767;
	} else if (threadIdx.x == 768) {
		dst[i] = 768;
	} else if (threadIdx.x == 769) {
		dst[i] = 769;
	} else if (threadIdx.x == 770) {
		dst[i] = 770;
	} else if (threadIdx.x == 771) {
		dst[i] = 771;
	} else if (threadIdx.x == 772) {
		dst[i] = 772;
	} else if (threadIdx.x == 773) {
		dst[i] = 773;
	} else if (threadIdx.x == 774) {
		dst[i] = 774;
	} else if (threadIdx.x == 775) {
		dst[i] = 775;
	} else if (threadIdx.x == 776) {
		dst[i] = 776;
	} else if (threadIdx.x == 777) {
		dst[i] = 777;
	} else if (threadIdx.x == 778) {
		dst[i] = 778;
	} else if (threadIdx.x == 779) {
		dst[i] = 779;
	} else if (threadIdx.x == 780) {
		dst[i] = 780;
	} else if (threadIdx.x == 781) {
		dst[i] = 781;
	} else if (threadIdx.x == 782) {
		dst[i] = 782;
	} else if (threadIdx.x == 783) {
		dst[i] = 783;
	} else if (threadIdx.x == 784) {
		dst[i] = 784;
	} else if (threadIdx.x == 785) {
		dst[i] = 785;
	} else if (threadIdx.x == 786) {
		dst[i] = 786;
	} else if (threadIdx.x == 787) {
		dst[i] = 787;
	} else if (threadIdx.x == 788) {
		dst[i] = 788;
	} else if (threadIdx.x == 789) {
		dst[i] = 789;
	} else if (threadIdx.x == 790) {
		dst[i] = 790;
	} else if (threadIdx.x == 791) {
		dst[i] = 791;
	} else if (threadIdx.x == 792) {
		dst[i] = 792;
	} else if (threadIdx.x == 793) {
		dst[i] = 793;
	} else if (threadIdx.x == 794) {
		dst[i] = 794;
	} else if (threadIdx.x == 795) {
		dst[i] = 795;
	} else if (threadIdx.x == 796) {
		dst[i] = 796;
	} else if (threadIdx.x == 797) {
		dst[i] = 797;
	} else if (threadIdx.x == 798) {
		dst[i] = 798;
	} else if (threadIdx.x == 799) {
		dst[i] = 799;
	} else if (threadIdx.x == 800) {
		dst[i] = 800;
	} else if (threadIdx.x == 801) {
		dst[i] = 801;
	} else if (threadIdx.x == 802) {
		dst[i] = 802;
	} else if (threadIdx.x == 803) {
		dst[i] = 803;
	} else if (threadIdx.x == 804) {
		dst[i] = 804;
	} else if (threadIdx.x == 805) {
		dst[i] = 805;
	} else if (threadIdx.x == 806) {
		dst[i] = 806;
	} else if (threadIdx.x == 807) {
		dst[i] = 807;
	} else if (threadIdx.x == 808) {
		dst[i] = 808;
	} else if (threadIdx.x == 809) {
		dst[i] = 809;
	} else if (threadIdx.x == 810) {
		dst[i] = 810;
	} else if (threadIdx.x == 811) {
		dst[i] = 811;
	} else if (threadIdx.x == 812) {
		dst[i] = 812;
	} else if (threadIdx.x == 813) {
		dst[i] = 813;
	} else if (threadIdx.x == 814) {
		dst[i] = 814;
	} else if (threadIdx.x == 815) {
		dst[i] = 815;
	} else if (threadIdx.x == 816) {
		dst[i] = 816;
	} else if (threadIdx.x == 817) {
		dst[i] = 817;
	} else if (threadIdx.x == 818) {
		dst[i] = 818;
	} else if (threadIdx.x == 819) {
		dst[i] = 819;
	} else if (threadIdx.x == 820) {
		dst[i] = 820;
	} else if (threadIdx.x == 821) {
		dst[i] = 821;
	} else if (threadIdx.x == 822) {
		dst[i] = 822;
	} else if (threadIdx.x == 823) {
		dst[i] = 823;
	} else if (threadIdx.x == 824) {
		dst[i] = 824;
	} else if (threadIdx.x == 825) {
		dst[i] = 825;
	} else if (threadIdx.x == 826) {
		dst[i] = 826;
	} else if (threadIdx.x == 827) {
		dst[i] = 827;
	} else if (threadIdx.x == 828) {
		dst[i] = 828;
	} else if (threadIdx.x == 829) {
		dst[i] = 829;
	} else if (threadIdx.x == 830) {
		dst[i] = 830;
	} else if (threadIdx.x == 831) {
		dst[i] = 831;
	} else if (threadIdx.x == 832) {
		dst[i] = 832;
	} else if (threadIdx.x == 833) {
		dst[i] = 833;
	} else if (threadIdx.x == 834) {
		dst[i] = 834;
	} else if (threadIdx.x == 835) {
		dst[i] = 835;
	} else if (threadIdx.x == 836) {
		dst[i] = 836;
	} else if (threadIdx.x == 837) {
		dst[i] = 837;
	} else if (threadIdx.x == 838) {
		dst[i] = 838;
	} else if (threadIdx.x == 839) {
		dst[i] = 839;
	} else if (threadIdx.x == 840) {
		dst[i] = 840;
	} else if (threadIdx.x == 841) {
		dst[i] = 841;
	} else if (threadIdx.x == 842) {
		dst[i] = 842;
	} else if (threadIdx.x == 843) {
		dst[i] = 843;
	} else if (threadIdx.x == 844) {
		dst[i] = 844;
	} else if (threadIdx.x == 845) {
		dst[i] = 845;
	} else if (threadIdx.x == 846) {
		dst[i] = 846;
	} else if (threadIdx.x == 847) {
		dst[i] = 847;
	} else if (threadIdx.x == 848) {
		dst[i] = 848;
	} else if (threadIdx.x == 849) {
		dst[i] = 849;
	} else if (threadIdx.x == 850) {
		dst[i] = 850;
	} else if (threadIdx.x == 851) {
		dst[i] = 851;
	} else if (threadIdx.x == 852) {
		dst[i] = 852;
	} else if (threadIdx.x == 853) {
		dst[i] = 853;
	} else if (threadIdx.x == 854) {
		dst[i] = 854;
	} else if (threadIdx.x == 855) {
		dst[i] = 855;
	} else if (threadIdx.x == 856) {
		dst[i] = 856;
	} else if (threadIdx.x == 857) {
		dst[i] = 857;
	} else if (threadIdx.x == 858) {
		dst[i] = 858;
	} else if (threadIdx.x == 859) {
		dst[i] = 859;
	} else if (threadIdx.x == 860) {
		dst[i] = 860;
	} else if (threadIdx.x == 861) {
		dst[i] = 861;
	} else if (threadIdx.x == 862) {
		dst[i] = 862;
	} else if (threadIdx.x == 863) {
		dst[i] = 863;
	} else if (threadIdx.x == 864) {
		dst[i] = 864;
	} else if (threadIdx.x == 865) {
		dst[i] = 865;
	} else if (threadIdx.x == 866) {
		dst[i] = 866;
	} else if (threadIdx.x == 867) {
		dst[i] = 867;
	} else if (threadIdx.x == 868) {
		dst[i] = 868;
	} else if (threadIdx.x == 869) {
		dst[i] = 869;
	} else if (threadIdx.x == 870) {
		dst[i] = 870;
	} else if (threadIdx.x == 871) {
		dst[i] = 871;
	} else if (threadIdx.x == 872) {
		dst[i] = 872;
	} else if (threadIdx.x == 873) {
		dst[i] = 873;
	} else if (threadIdx.x == 874) {
		dst[i] = 874;
	} else if (threadIdx.x == 875) {
		dst[i] = 875;
	} else if (threadIdx.x == 876) {
		dst[i] = 876;
	} else if (threadIdx.x == 877) {
		dst[i] = 877;
	} else if (threadIdx.x == 878) {
		dst[i] = 878;
	} else if (threadIdx.x == 879) {
		dst[i] = 879;
	} else if (threadIdx.x == 880) {
		dst[i] = 880;
	} else if (threadIdx.x == 881) {
		dst[i] = 881;
	} else if (threadIdx.x == 882) {
		dst[i] = 882;
	} else if (threadIdx.x == 883) {
		dst[i] = 883;
	} else if (threadIdx.x == 884) {
		dst[i] = 884;
	} else if (threadIdx.x == 885) {
		dst[i] = 885;
	} else if (threadIdx.x == 886) {
		dst[i] = 886;
	} else if (threadIdx.x == 887) {
		dst[i] = 887;
	} else if (threadIdx.x == 888) {
		dst[i] = 888;
	} else if (threadIdx.x == 889) {
		dst[i] = 889;
	} else if (threadIdx.x == 890) {
		dst[i] = 890;
	} else if (threadIdx.x == 891) {
		dst[i] = 891;
	} else if (threadIdx.x == 892) {
		dst[i] = 892;
	} else if (threadIdx.x == 893) {
		dst[i] = 893;
	} else if (threadIdx.x == 894) {
		dst[i] = 894;
	} else if (threadIdx.x == 895) {
		dst[i] = 895;
	} else if (threadIdx.x == 896) {
		dst[i] = 896;
	} else if (threadIdx.x == 897) {
		dst[i] = 897;
	} else if (threadIdx.x == 898) {
		dst[i] = 898;
	} else if (threadIdx.x == 899) {
		dst[i] = 899;
	} else if (threadIdx.x == 900) {
		dst[i] = 900;
	} else if (threadIdx.x == 901) {
		dst[i] = 901;
	} else if (threadIdx.x == 902) {
		dst[i] = 902;
	} else if (threadIdx.x == 903) {
		dst[i] = 903;
	} else if (threadIdx.x == 904) {
		dst[i] = 904;
	} else if (threadIdx.x == 905) {
		dst[i] = 905;
	} else if (threadIdx.x == 906) {
		dst[i] = 906;
	} else if (threadIdx.x == 907) {
		dst[i] = 907;
	} else if (threadIdx.x == 908) {
		dst[i] = 908;
	} else if (threadIdx.x == 909) {
		dst[i] = 909;
	} else if (threadIdx.x == 910) {
		dst[i] = 910;
	} else if (threadIdx.x == 911) {
		dst[i] = 911;
	} else if (threadIdx.x == 912) {
		dst[i] = 912;
	} else if (threadIdx.x == 913) {
		dst[i] = 913;
	} else if (threadIdx.x == 914) {
		dst[i] = 914;
	} else if (threadIdx.x == 915) {
		dst[i] = 915;
	} else if (threadIdx.x == 916) {
		dst[i] = 916;
	} else if (threadIdx.x == 917) {
		dst[i] = 917;
	} else if (threadIdx.x == 918) {
		dst[i] = 918;
	} else if (threadIdx.x == 919) {
		dst[i] = 919;
	} else if (threadIdx.x == 920) {
		dst[i] = 920;
	} else if (threadIdx.x == 921) {
		dst[i] = 921;
	} else if (threadIdx.x == 922) {
		dst[i] = 922;
	} else if (threadIdx.x == 923) {
		dst[i] = 923;
	} else if (threadIdx.x == 924) {
		dst[i] = 924;
	} else if (threadIdx.x == 925) {
		dst[i] = 925;
	} else if (threadIdx.x == 926) {
		dst[i] = 926;
	} else if (threadIdx.x == 927) {
		dst[i] = 927;
	} else if (threadIdx.x == 928) {
		dst[i] = 928;
	} else if (threadIdx.x == 929) {
		dst[i] = 929;
	} else if (threadIdx.x == 930) {
		dst[i] = 930;
	} else if (threadIdx.x == 931) {
		dst[i] = 931;
	} else if (threadIdx.x == 932) {
		dst[i] = 932;
	} else if (threadIdx.x == 933) {
		dst[i] = 933;
	} else if (threadIdx.x == 934) {
		dst[i] = 934;
	} else if (threadIdx.x == 935) {
		dst[i] = 935;
	} else if (threadIdx.x == 936) {
		dst[i] = 936;
	} else if (threadIdx.x == 937) {
		dst[i] = 937;
	} else if (threadIdx.x == 938) {
		dst[i] = 938;
	} else if (threadIdx.x == 939) {
		dst[i] = 939;
	} else if (threadIdx.x == 940) {
		dst[i] = 940;
	} else if (threadIdx.x == 941) {
		dst[i] = 941;
	} else if (threadIdx.x == 942) {
		dst[i] = 942;
	} else if (threadIdx.x == 943) {
		dst[i] = 943;
	} else if (threadIdx.x == 944) {
		dst[i] = 944;
	} else if (threadIdx.x == 945) {
		dst[i] = 945;
	} else if (threadIdx.x == 946) {
		dst[i] = 946;
	} else if (threadIdx.x == 947) {
		dst[i] = 947;
	} else if (threadIdx.x == 948) {
		dst[i] = 948;
	} else if (threadIdx.x == 949) {
		dst[i] = 949;
	} else if (threadIdx.x == 950) {
		dst[i] = 950;
	} else if (threadIdx.x == 951) {
		dst[i] = 951;
	} else if (threadIdx.x == 952) {
		dst[i] = 952;
	} else if (threadIdx.x == 953) {
		dst[i] = 953;
	} else if (threadIdx.x == 954) {
		dst[i] = 954;
	} else if (threadIdx.x == 955) {
		dst[i] = 955;
	} else if (threadIdx.x == 956) {
		dst[i] = 956;
	} else if (threadIdx.x == 957) {
		dst[i] = 957;
	} else if (threadIdx.x == 958) {
		dst[i] = 958;
	} else if (threadIdx.x == 959) {
		dst[i] = 959;
	} else if (threadIdx.x == 960) {
		dst[i] = 960;
	} else if (threadIdx.x == 961) {
		dst[i] = 961;
	} else if (threadIdx.x == 962) {
		dst[i] = 962;
	} else if (threadIdx.x == 963) {
		dst[i] = 963;
	} else if (threadIdx.x == 964) {
		dst[i] = 964;
	} else if (threadIdx.x == 965) {
		dst[i] = 965;
	} else if (threadIdx.x == 966) {
		dst[i] = 966;
	} else if (threadIdx.x == 967) {
		dst[i] = 967;
	} else if (threadIdx.x == 968) {
		dst[i] = 968;
	} else if (threadIdx.x == 969) {
		dst[i] = 969;
	} else if (threadIdx.x == 970) {
		dst[i] = 970;
	} else if (threadIdx.x == 971) {
		dst[i] = 971;
	} else if (threadIdx.x == 972) {
		dst[i] = 972;
	} else if (threadIdx.x == 973) {
		dst[i] = 973;
	} else if (threadIdx.x == 974) {
		dst[i] = 974;
	} else if (threadIdx.x == 975) {
		dst[i] = 975;
	} else if (threadIdx.x == 976) {
		dst[i] = 976;
	} else if (threadIdx.x == 977) {
		dst[i] = 977;
	} else if (threadIdx.x == 978) {
		dst[i] = 978;
	} else if (threadIdx.x == 979) {
		dst[i] = 979;
	} else if (threadIdx.x == 980) {
		dst[i] = 980;
	} else if (threadIdx.x == 981) {
		dst[i] = 981;
	} else if (threadIdx.x == 982) {
		dst[i] = 982;
	} else if (threadIdx.x == 983) {
		dst[i] = 983;
	} else if (threadIdx.x == 984) {
		dst[i] = 984;
	} else if (threadIdx.x == 985) {
		dst[i] = 985;
	} else if (threadIdx.x == 986) {
		dst[i] = 986;
	} else if (threadIdx.x == 987) {
		dst[i] = 987;
	} else if (threadIdx.x == 988) {
		dst[i] = 988;
	} else if (threadIdx.x == 989) {
		dst[i] = 989;
	} else if (threadIdx.x == 990) {
		dst[i] = 990;
	} else if (threadIdx.x == 991) {
		dst[i] = 991;
	} else if (threadIdx.x == 992) {
		dst[i] = 992;
	} else if (threadIdx.x == 993) {
		dst[i] = 993;
	} else if (threadIdx.x == 994) {
		dst[i] = 994;
	} else if (threadIdx.x == 995) {
		dst[i] = 995;
	} else if (threadIdx.x == 996) {
		dst[i] = 996;
	} else if (threadIdx.x == 997) {
		dst[i] = 997;
	} else if (threadIdx.x == 998) {
		dst[i] = 998;
	} else if (threadIdx.x == 999) {
		dst[i] = 999;
	} else if (threadIdx.x == 1000) {
		dst[i] = 1000;
	} else if (threadIdx.x == 1001) {
		dst[i] = 1001;
	} else if (threadIdx.x == 1002) {
		dst[i] = 1002;
	} else if (threadIdx.x == 1003) {
		dst[i] = 1003;
	} else if (threadIdx.x == 1004) {
		dst[i] = 1004;
	} else if (threadIdx.x == 1005) {
		dst[i] = 1005;
	} else if (threadIdx.x == 1006) {
		dst[i] = 1006;
	} else if (threadIdx.x == 1007) {
		dst[i] = 1007;
	} else if (threadIdx.x == 1008) {
		dst[i] = 1008;
	} else if (threadIdx.x == 1009) {
		dst[i] = 1009;
	} else if (threadIdx.x == 1010) {
		dst[i] = 1010;
	} else if (threadIdx.x == 1011) {
		dst[i] = 1011;
	} else if (threadIdx.x == 1012) {
		dst[i] = 1012;
	} else if (threadIdx.x == 1013) {
		dst[i] = 1013;
	} else if (threadIdx.x == 1014) {
		dst[i] = 1014;
	} else if (threadIdx.x == 1015) {
		dst[i] = 1015;
	} else if (threadIdx.x == 1016) {
		dst[i] = 1016;
	} else if (threadIdx.x == 1017) {
		dst[i] = 1017;
	} else if (threadIdx.x == 1018) {
		dst[i] = 1018;
	} else if (threadIdx.x == 1019) {
		dst[i] = 1019;
	} else if (threadIdx.x == 1020) {
		dst[i] = 1020;
	} else if (threadIdx.x == 1021) {
		dst[i] = 1021;
	} else if (threadIdx.x == 1022) {
		dst[i] = 1022;
	} else if (threadIdx.x == 1023) {
		dst[i] = 1023;
	}
}

#endif /* BRANCH_KERNEL_H_ */
