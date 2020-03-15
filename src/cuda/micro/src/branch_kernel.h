#ifndef BRANCH_KERNEL_H_
#define BRANCH_KERNEL_H_

#include <cstdint>


template<typename int_t>
__global__ void int_branch_kernel(int_t* dst, uint32_t op) {
	const int_t i = (blockDim.x * blockIdx.x + threadIdx.x);
	int_t value = i;

	if (threadIdx.x == 0) {
		value = 0;
	} else if (threadIdx.x == 1) {
		value = 1;
	} else if (threadIdx.x == 2) {
		value = 2;
	} else if (threadIdx.x == 3) {
		value = 3;
	} else if (threadIdx.x == 4) {
		value = 4;
	} else if (threadIdx.x == 5) {
		value = 5;
	} else if (threadIdx.x == 6) {
		value = 6;
	} else if (threadIdx.x == 7) {
		value = 7;
	} else if (threadIdx.x == 8) {
		value = 8;
	} else if (threadIdx.x == 9) {
		value = 9;
	} else if (threadIdx.x == 10) {
		value = 10;
	} else if (threadIdx.x == 11) {
		value = 11;
	} else if (threadIdx.x == 12) {
		value = 12;
	} else if (threadIdx.x == 13) {
		value = 13;
	} else if (threadIdx.x == 14) {
		value = 14;
	} else if (threadIdx.x == 15) {
		value = 15;
	} else if (threadIdx.x == 16) {
		value = 16;
	} else if (threadIdx.x == 17) {
		value = 17;
	} else if (threadIdx.x == 18) {
		value = 18;
	} else if (threadIdx.x == 19) {
		value = 19;
	} else if (threadIdx.x == 20) {
		value = 20;
	} else if (threadIdx.x == 21) {
		value = 21;
	} else if (threadIdx.x == 22) {
		value = 22;
	} else if (threadIdx.x == 23) {
		value = 23;
	} else if (threadIdx.x == 24) {
		value = 24;
	} else if (threadIdx.x == 25) {
		value = 25;
	} else if (threadIdx.x == 26) {
		value = 26;
	} else if (threadIdx.x == 27) {
		value = 27;
	} else if (threadIdx.x == 28) {
		value = 28;
	} else if (threadIdx.x == 29) {
		value = 29;
	} else if (threadIdx.x == 30) {
		value = 30;
	} else if (threadIdx.x == 31) {
		value = 31;
	} else if (threadIdx.x == 32) {
		value = 32;
	} else if (threadIdx.x == 33) {
		value = 33;
	} else if (threadIdx.x == 34) {
		value = 34;
	} else if (threadIdx.x == 35) {
		value = 35;
	} else if (threadIdx.x == 36) {
		value = 36;
	} else if (threadIdx.x == 37) {
		value = 37;
	} else if (threadIdx.x == 38) {
		value = 38;
	} else if (threadIdx.x == 39) {
		value = 39;
	} else if (threadIdx.x == 40) {
		value = 40;
	} else if (threadIdx.x == 41) {
		value = 41;
	} else if (threadIdx.x == 42) {
		value = 42;
	} else if (threadIdx.x == 43) {
		value = 43;
	} else if (threadIdx.x == 44) {
		value = 44;
	} else if (threadIdx.x == 45) {
		value = 45;
	} else if (threadIdx.x == 46) {
		value = 46;
	} else if (threadIdx.x == 47) {
		value = 47;
	} else if (threadIdx.x == 48) {
		value = 48;
	} else if (threadIdx.x == 49) {
		value = 49;
	} else if (threadIdx.x == 50) {
		value = 50;
	} else if (threadIdx.x == 51) {
		value = 51;
	} else if (threadIdx.x == 52) {
		value = 52;
	} else if (threadIdx.x == 53) {
		value = 53;
	} else if (threadIdx.x == 54) {
		value = 54;
	} else if (threadIdx.x == 55) {
		value = 55;
	} else if (threadIdx.x == 56) {
		value = 56;
	} else if (threadIdx.x == 57) {
		value = 57;
	} else if (threadIdx.x == 58) {
		value = 58;
	} else if (threadIdx.x == 59) {
		value = 59;
	} else if (threadIdx.x == 60) {
		value = 60;
	} else if (threadIdx.x == 61) {
		value = 61;
	} else if (threadIdx.x == 62) {
		value = 62;
	} else if (threadIdx.x == 63) {
		value = 63;
	} else if (threadIdx.x == 64) {
		value = 64;
	} else if (threadIdx.x == 65) {
		value = 65;
	} else if (threadIdx.x == 66) {
		value = 66;
	} else if (threadIdx.x == 67) {
		value = 67;
	} else if (threadIdx.x == 68) {
		value = 68;
	} else if (threadIdx.x == 69) {
		value = 69;
	} else if (threadIdx.x == 70) {
		value = 70;
	} else if (threadIdx.x == 71) {
		value = 71;
	} else if (threadIdx.x == 72) {
		value = 72;
	} else if (threadIdx.x == 73) {
		value = 73;
	} else if (threadIdx.x == 74) {
		value = 74;
	} else if (threadIdx.x == 75) {
		value = 75;
	} else if (threadIdx.x == 76) {
		value = 76;
	} else if (threadIdx.x == 77) {
		value = 77;
	} else if (threadIdx.x == 78) {
		value = 78;
	} else if (threadIdx.x == 79) {
		value = 79;
	} else if (threadIdx.x == 80) {
		value = 80;
	} else if (threadIdx.x == 81) {
		value = 81;
	} else if (threadIdx.x == 82) {
		value = 82;
	} else if (threadIdx.x == 83) {
		value = 83;
	} else if (threadIdx.x == 84) {
		value = 84;
	} else if (threadIdx.x == 85) {
		value = 85;
	} else if (threadIdx.x == 86) {
		value = 86;
	} else if (threadIdx.x == 87) {
		value = 87;
	} else if (threadIdx.x == 88) {
		value = 88;
	} else if (threadIdx.x == 89) {
		value = 89;
	} else if (threadIdx.x == 90) {
		value = 90;
	} else if (threadIdx.x == 91) {
		value = 91;
	} else if (threadIdx.x == 92) {
		value = 92;
	} else if (threadIdx.x == 93) {
		value = 93;
	} else if (threadIdx.x == 94) {
		value = 94;
	} else if (threadIdx.x == 95) {
		value = 95;
	} else if (threadIdx.x == 96) {
		value = 96;
	} else if (threadIdx.x == 97) {
		value = 97;
	} else if (threadIdx.x == 98) {
		value = 98;
	} else if (threadIdx.x == 99) {
		value = 99;
	} else if (threadIdx.x == 100) {
		value = 100;
	} else if (threadIdx.x == 101) {
		value = 101;
	} else if (threadIdx.x == 102) {
		value = 102;
	} else if (threadIdx.x == 103) {
		value = 103;
	} else if (threadIdx.x == 104) {
		value = 104;
	} else if (threadIdx.x == 105) {
		value = 105;
	} else if (threadIdx.x == 106) {
		value = 106;
	} else if (threadIdx.x == 107) {
		value = 107;
	} else if (threadIdx.x == 108) {
		value = 108;
	} else if (threadIdx.x == 109) {
		value = 109;
	} else if (threadIdx.x == 110) {
		value = 110;
	} else if (threadIdx.x == 111) {
		value = 111;
	} else if (threadIdx.x == 112) {
		value = 112;
	} else if (threadIdx.x == 113) {
		value = 113;
	} else if (threadIdx.x == 114) {
		value = 114;
	} else if (threadIdx.x == 115) {
		value = 115;
	} else if (threadIdx.x == 116) {
		value = 116;
	} else if (threadIdx.x == 117) {
		value = 117;
	} else if (threadIdx.x == 118) {
		value = 118;
	} else if (threadIdx.x == 119) {
		value = 119;
	} else if (threadIdx.x == 120) {
		value = 120;
	} else if (threadIdx.x == 121) {
		value = 121;
	} else if (threadIdx.x == 122) {
		value = 122;
	} else if (threadIdx.x == 123) {
		value = 123;
	} else if (threadIdx.x == 124) {
		value = 124;
	} else if (threadIdx.x == 125) {
		value = 125;
	} else if (threadIdx.x == 126) {
		value = 126;
	} else if (threadIdx.x == 127) {
		value = 127;
	} else if (threadIdx.x == 128) {
		value = 128;
	} else if (threadIdx.x == 129) {
		value = 129;
	} else if (threadIdx.x == 130) {
		value = 130;
	} else if (threadIdx.x == 131) {
		value = 131;
	} else if (threadIdx.x == 132) {
		value = 132;
	} else if (threadIdx.x == 133) {
		value = 133;
	} else if (threadIdx.x == 134) {
		value = 134;
	} else if (threadIdx.x == 135) {
		value = 135;
	} else if (threadIdx.x == 136) {
		value = 136;
	} else if (threadIdx.x == 137) {
		value = 137;
	} else if (threadIdx.x == 138) {
		value = 138;
	} else if (threadIdx.x == 139) {
		value = 139;
	} else if (threadIdx.x == 140) {
		value = 140;
	} else if (threadIdx.x == 141) {
		value = 141;
	} else if (threadIdx.x == 142) {
		value = 142;
	} else if (threadIdx.x == 143) {
		value = 143;
	} else if (threadIdx.x == 144) {
		value = 144;
	} else if (threadIdx.x == 145) {
		value = 145;
	} else if (threadIdx.x == 146) {
		value = 146;
	} else if (threadIdx.x == 147) {
		value = 147;
	} else if (threadIdx.x == 148) {
		value = 148;
	} else if (threadIdx.x == 149) {
		value = 149;
	} else if (threadIdx.x == 150) {
		value = 150;
	} else if (threadIdx.x == 151) {
		value = 151;
	} else if (threadIdx.x == 152) {
		value = 152;
	} else if (threadIdx.x == 153) {
		value = 153;
	} else if (threadIdx.x == 154) {
		value = 154;
	} else if (threadIdx.x == 155) {
		value = 155;
	} else if (threadIdx.x == 156) {
		value = 156;
	} else if (threadIdx.x == 157) {
		value = 157;
	} else if (threadIdx.x == 158) {
		value = 158;
	} else if (threadIdx.x == 159) {
		value = 159;
	} else if (threadIdx.x == 160) {
		value = 160;
	} else if (threadIdx.x == 161) {
		value = 161;
	} else if (threadIdx.x == 162) {
		value = 162;
	} else if (threadIdx.x == 163) {
		value = 163;
	} else if (threadIdx.x == 164) {
		value = 164;
	} else if (threadIdx.x == 165) {
		value = 165;
	} else if (threadIdx.x == 166) {
		value = 166;
	} else if (threadIdx.x == 167) {
		value = 167;
	} else if (threadIdx.x == 168) {
		value = 168;
	} else if (threadIdx.x == 169) {
		value = 169;
	} else if (threadIdx.x == 170) {
		value = 170;
	} else if (threadIdx.x == 171) {
		value = 171;
	} else if (threadIdx.x == 172) {
		value = 172;
	} else if (threadIdx.x == 173) {
		value = 173;
	} else if (threadIdx.x == 174) {
		value = 174;
	} else if (threadIdx.x == 175) {
		value = 175;
	} else if (threadIdx.x == 176) {
		value = 176;
	} else if (threadIdx.x == 177) {
		value = 177;
	} else if (threadIdx.x == 178) {
		value = 178;
	} else if (threadIdx.x == 179) {
		value = 179;
	} else if (threadIdx.x == 180) {
		value = 180;
	} else if (threadIdx.x == 181) {
		value = 181;
	} else if (threadIdx.x == 182) {
		value = 182;
	} else if (threadIdx.x == 183) {
		value = 183;
	} else if (threadIdx.x == 184) {
		value = 184;
	} else if (threadIdx.x == 185) {
		value = 185;
	} else if (threadIdx.x == 186) {
		value = 186;
	} else if (threadIdx.x == 187) {
		value = 187;
	} else if (threadIdx.x == 188) {
		value = 188;
	} else if (threadIdx.x == 189) {
		value = 189;
	} else if (threadIdx.x == 190) {
		value = 190;
	} else if (threadIdx.x == 191) {
		value = 191;
	} else if (threadIdx.x == 192) {
		value = 192;
	} else if (threadIdx.x == 193) {
		value = 193;
	} else if (threadIdx.x == 194) {
		value = 194;
	} else if (threadIdx.x == 195) {
		value = 195;
	} else if (threadIdx.x == 196) {
		value = 196;
	} else if (threadIdx.x == 197) {
		value = 197;
	} else if (threadIdx.x == 198) {
		value = 198;
	} else if (threadIdx.x == 199) {
		value = 199;
	} else if (threadIdx.x == 200) {
		value = 200;
	} else if (threadIdx.x == 201) {
		value = 201;
	} else if (threadIdx.x == 202) {
		value = 202;
	} else if (threadIdx.x == 203) {
		value = 203;
	} else if (threadIdx.x == 204) {
		value = 204;
	} else if (threadIdx.x == 205) {
		value = 205;
	} else if (threadIdx.x == 206) {
		value = 206;
	} else if (threadIdx.x == 207) {
		value = 207;
	} else if (threadIdx.x == 208) {
		value = 208;
	} else if (threadIdx.x == 209) {
		value = 209;
	} else if (threadIdx.x == 210) {
		value = 210;
	} else if (threadIdx.x == 211) {
		value = 211;
	} else if (threadIdx.x == 212) {
		value = 212;
	} else if (threadIdx.x == 213) {
		value = 213;
	} else if (threadIdx.x == 214) {
		value = 214;
	} else if (threadIdx.x == 215) {
		value = 215;
	} else if (threadIdx.x == 216) {
		value = 216;
	} else if (threadIdx.x == 217) {
		value = 217;
	} else if (threadIdx.x == 218) {
		value = 218;
	} else if (threadIdx.x == 219) {
		value = 219;
	} else if (threadIdx.x == 220) {
		value = 220;
	} else if (threadIdx.x == 221) {
		value = 221;
	} else if (threadIdx.x == 222) {
		value = 222;
	} else if (threadIdx.x == 223) {
		value = 223;
	} else if (threadIdx.x == 224) {
		value = 224;
	} else if (threadIdx.x == 225) {
		value = 225;
	} else if (threadIdx.x == 226) {
		value = 226;
	} else if (threadIdx.x == 227) {
		value = 227;
	} else if (threadIdx.x == 228) {
		value = 228;
	} else if (threadIdx.x == 229) {
		value = 229;
	} else if (threadIdx.x == 230) {
		value = 230;
	} else if (threadIdx.x == 231) {
		value = 231;
	} else if (threadIdx.x == 232) {
		value = 232;
	} else if (threadIdx.x == 233) {
		value = 233;
	} else if (threadIdx.x == 234) {
		value = 234;
	} else if (threadIdx.x == 235) {
		value = 235;
	} else if (threadIdx.x == 236) {
		value = 236;
	} else if (threadIdx.x == 237) {
		value = 237;
	} else if (threadIdx.x == 238) {
		value = 238;
	} else if (threadIdx.x == 239) {
		value = 239;
	} else if (threadIdx.x == 240) {
		value = 240;
	} else if (threadIdx.x == 241) {
		value = 241;
	} else if (threadIdx.x == 242) {
		value = 242;
	} else if (threadIdx.x == 243) {
		value = 243;
	} else if (threadIdx.x == 244) {
		value = 244;
	} else if (threadIdx.x == 245) {
		value = 245;
	} else if (threadIdx.x == 246) {
		value = 246;
	} else if (threadIdx.x == 247) {
		value = 247;
	} else if (threadIdx.x == 248) {
		value = 248;
	} else if (threadIdx.x == 249) {
		value = 249;
	} else if (threadIdx.x == 250) {
		value = 250;
	} else if (threadIdx.x == 251) {
		value = 251;
	} else if (threadIdx.x == 252) {
		value = 252;
	} else if (threadIdx.x == 253) {
		value = 253;
	} else if (threadIdx.x == 254) {
		value = 254;
	} else if (threadIdx.x == 255) {
		value = 255;
	} else if (threadIdx.x == 256) {
		value = 256;
	} else if (threadIdx.x == 257) {
		value = 257;
	} else if (threadIdx.x == 258) {
		value = 258;
	} else if (threadIdx.x == 259) {
		value = 259;
	} else if (threadIdx.x == 260) {
		value = 260;
	} else if (threadIdx.x == 261) {
		value = 261;
	} else if (threadIdx.x == 262) {
		value = 262;
	} else if (threadIdx.x == 263) {
		value = 263;
	} else if (threadIdx.x == 264) {
		value = 264;
	} else if (threadIdx.x == 265) {
		value = 265;
	} else if (threadIdx.x == 266) {
		value = 266;
	} else if (threadIdx.x == 267) {
		value = 267;
	} else if (threadIdx.x == 268) {
		value = 268;
	} else if (threadIdx.x == 269) {
		value = 269;
	} else if (threadIdx.x == 270) {
		value = 270;
	} else if (threadIdx.x == 271) {
		value = 271;
	} else if (threadIdx.x == 272) {
		value = 272;
	} else if (threadIdx.x == 273) {
		value = 273;
	} else if (threadIdx.x == 274) {
		value = 274;
	} else if (threadIdx.x == 275) {
		value = 275;
	} else if (threadIdx.x == 276) {
		value = 276;
	} else if (threadIdx.x == 277) {
		value = 277;
	} else if (threadIdx.x == 278) {
		value = 278;
	} else if (threadIdx.x == 279) {
		value = 279;
	} else if (threadIdx.x == 280) {
		value = 280;
	} else if (threadIdx.x == 281) {
		value = 281;
	} else if (threadIdx.x == 282) {
		value = 282;
	} else if (threadIdx.x == 283) {
		value = 283;
	} else if (threadIdx.x == 284) {
		value = 284;
	} else if (threadIdx.x == 285) {
		value = 285;
	} else if (threadIdx.x == 286) {
		value = 286;
	} else if (threadIdx.x == 287) {
		value = 287;
	} else if (threadIdx.x == 288) {
		value = 288;
	} else if (threadIdx.x == 289) {
		value = 289;
	} else if (threadIdx.x == 290) {
		value = 290;
	} else if (threadIdx.x == 291) {
		value = 291;
	} else if (threadIdx.x == 292) {
		value = 292;
	} else if (threadIdx.x == 293) {
		value = 293;
	} else if (threadIdx.x == 294) {
		value = 294;
	} else if (threadIdx.x == 295) {
		value = 295;
	} else if (threadIdx.x == 296) {
		value = 296;
	} else if (threadIdx.x == 297) {
		value = 297;
	} else if (threadIdx.x == 298) {
		value = 298;
	} else if (threadIdx.x == 299) {
		value = 299;
	} else if (threadIdx.x == 300) {
		value = 300;
	} else if (threadIdx.x == 301) {
		value = 301;
	} else if (threadIdx.x == 302) {
		value = 302;
	} else if (threadIdx.x == 303) {
		value = 303;
	} else if (threadIdx.x == 304) {
		value = 304;
	} else if (threadIdx.x == 305) {
		value = 305;
	} else if (threadIdx.x == 306) {
		value = 306;
	} else if (threadIdx.x == 307) {
		value = 307;
	} else if (threadIdx.x == 308) {
		value = 308;
	} else if (threadIdx.x == 309) {
		value = 309;
	} else if (threadIdx.x == 310) {
		value = 310;
	} else if (threadIdx.x == 311) {
		value = 311;
	} else if (threadIdx.x == 312) {
		value = 312;
	} else if (threadIdx.x == 313) {
		value = 313;
	} else if (threadIdx.x == 314) {
		value = 314;
	} else if (threadIdx.x == 315) {
		value = 315;
	} else if (threadIdx.x == 316) {
		value = 316;
	} else if (threadIdx.x == 317) {
		value = 317;
	} else if (threadIdx.x == 318) {
		value = 318;
	} else if (threadIdx.x == 319) {
		value = 319;
	} else if (threadIdx.x == 320) {
		value = 320;
	} else if (threadIdx.x == 321) {
		value = 321;
	} else if (threadIdx.x == 322) {
		value = 322;
	} else if (threadIdx.x == 323) {
		value = 323;
	} else if (threadIdx.x == 324) {
		value = 324;
	} else if (threadIdx.x == 325) {
		value = 325;
	} else if (threadIdx.x == 326) {
		value = 326;
	} else if (threadIdx.x == 327) {
		value = 327;
	} else if (threadIdx.x == 328) {
		value = 328;
	} else if (threadIdx.x == 329) {
		value = 329;
	} else if (threadIdx.x == 330) {
		value = 330;
	} else if (threadIdx.x == 331) {
		value = 331;
	} else if (threadIdx.x == 332) {
		value = 332;
	} else if (threadIdx.x == 333) {
		value = 333;
	} else if (threadIdx.x == 334) {
		value = 334;
	} else if (threadIdx.x == 335) {
		value = 335;
	} else if (threadIdx.x == 336) {
		value = 336;
	} else if (threadIdx.x == 337) {
		value = 337;
	} else if (threadIdx.x == 338) {
		value = 338;
	} else if (threadIdx.x == 339) {
		value = 339;
	} else if (threadIdx.x == 340) {
		value = 340;
	} else if (threadIdx.x == 341) {
		value = 341;
	} else if (threadIdx.x == 342) {
		value = 342;
	} else if (threadIdx.x == 343) {
		value = 343;
	} else if (threadIdx.x == 344) {
		value = 344;
	} else if (threadIdx.x == 345) {
		value = 345;
	} else if (threadIdx.x == 346) {
		value = 346;
	} else if (threadIdx.x == 347) {
		value = 347;
	} else if (threadIdx.x == 348) {
		value = 348;
	} else if (threadIdx.x == 349) {
		value = 349;
	} else if (threadIdx.x == 350) {
		value = 350;
	} else if (threadIdx.x == 351) {
		value = 351;
	} else if (threadIdx.x == 352) {
		value = 352;
	} else if (threadIdx.x == 353) {
		value = 353;
	} else if (threadIdx.x == 354) {
		value = 354;
	} else if (threadIdx.x == 355) {
		value = 355;
	} else if (threadIdx.x == 356) {
		value = 356;
	} else if (threadIdx.x == 357) {
		value = 357;
	} else if (threadIdx.x == 358) {
		value = 358;
	} else if (threadIdx.x == 359) {
		value = 359;
	} else if (threadIdx.x == 360) {
		value = 360;
	} else if (threadIdx.x == 361) {
		value = 361;
	} else if (threadIdx.x == 362) {
		value = 362;
	} else if (threadIdx.x == 363) {
		value = 363;
	} else if (threadIdx.x == 364) {
		value = 364;
	} else if (threadIdx.x == 365) {
		value = 365;
	} else if (threadIdx.x == 366) {
		value = 366;
	} else if (threadIdx.x == 367) {
		value = 367;
	} else if (threadIdx.x == 368) {
		value = 368;
	} else if (threadIdx.x == 369) {
		value = 369;
	} else if (threadIdx.x == 370) {
		value = 370;
	} else if (threadIdx.x == 371) {
		value = 371;
	} else if (threadIdx.x == 372) {
		value = 372;
	} else if (threadIdx.x == 373) {
		value = 373;
	} else if (threadIdx.x == 374) {
		value = 374;
	} else if (threadIdx.x == 375) {
		value = 375;
	} else if (threadIdx.x == 376) {
		value = 376;
	} else if (threadIdx.x == 377) {
		value = 377;
	} else if (threadIdx.x == 378) {
		value = 378;
	} else if (threadIdx.x == 379) {
		value = 379;
	} else if (threadIdx.x == 380) {
		value = 380;
	} else if (threadIdx.x == 381) {
		value = 381;
	} else if (threadIdx.x == 382) {
		value = 382;
	} else if (threadIdx.x == 383) {
		value = 383;
	} else if (threadIdx.x == 384) {
		value = 384;
	} else if (threadIdx.x == 385) {
		value = 385;
	} else if (threadIdx.x == 386) {
		value = 386;
	} else if (threadIdx.x == 387) {
		value = 387;
	} else if (threadIdx.x == 388) {
		value = 388;
	} else if (threadIdx.x == 389) {
		value = 389;
	} else if (threadIdx.x == 390) {
		value = 390;
	} else if (threadIdx.x == 391) {
		value = 391;
	} else if (threadIdx.x == 392) {
		value = 392;
	} else if (threadIdx.x == 393) {
		value = 393;
	} else if (threadIdx.x == 394) {
		value = 394;
	} else if (threadIdx.x == 395) {
		value = 395;
	} else if (threadIdx.x == 396) {
		value = 396;
	} else if (threadIdx.x == 397) {
		value = 397;
	} else if (threadIdx.x == 398) {
		value = 398;
	} else if (threadIdx.x == 399) {
		value = 399;
	} else if (threadIdx.x == 400) {
		value = 400;
	} else if (threadIdx.x == 401) {
		value = 401;
	} else if (threadIdx.x == 402) {
		value = 402;
	} else if (threadIdx.x == 403) {
		value = 403;
	} else if (threadIdx.x == 404) {
		value = 404;
	} else if (threadIdx.x == 405) {
		value = 405;
	} else if (threadIdx.x == 406) {
		value = 406;
	} else if (threadIdx.x == 407) {
		value = 407;
	} else if (threadIdx.x == 408) {
		value = 408;
	} else if (threadIdx.x == 409) {
		value = 409;
	} else if (threadIdx.x == 410) {
		value = 410;
	} else if (threadIdx.x == 411) {
		value = 411;
	} else if (threadIdx.x == 412) {
		value = 412;
	} else if (threadIdx.x == 413) {
		value = 413;
	} else if (threadIdx.x == 414) {
		value = 414;
	} else if (threadIdx.x == 415) {
		value = 415;
	} else if (threadIdx.x == 416) {
		value = 416;
	} else if (threadIdx.x == 417) {
		value = 417;
	} else if (threadIdx.x == 418) {
		value = 418;
	} else if (threadIdx.x == 419) {
		value = 419;
	} else if (threadIdx.x == 420) {
		value = 420;
	} else if (threadIdx.x == 421) {
		value = 421;
	} else if (threadIdx.x == 422) {
		value = 422;
	} else if (threadIdx.x == 423) {
		value = 423;
	} else if (threadIdx.x == 424) {
		value = 424;
	} else if (threadIdx.x == 425) {
		value = 425;
	} else if (threadIdx.x == 426) {
		value = 426;
	} else if (threadIdx.x == 427) {
		value = 427;
	} else if (threadIdx.x == 428) {
		value = 428;
	} else if (threadIdx.x == 429) {
		value = 429;
	} else if (threadIdx.x == 430) {
		value = 430;
	} else if (threadIdx.x == 431) {
		value = 431;
	} else if (threadIdx.x == 432) {
		value = 432;
	} else if (threadIdx.x == 433) {
		value = 433;
	} else if (threadIdx.x == 434) {
		value = 434;
	} else if (threadIdx.x == 435) {
		value = 435;
	} else if (threadIdx.x == 436) {
		value = 436;
	} else if (threadIdx.x == 437) {
		value = 437;
	} else if (threadIdx.x == 438) {
		value = 438;
	} else if (threadIdx.x == 439) {
		value = 439;
	} else if (threadIdx.x == 440) {
		value = 440;
	} else if (threadIdx.x == 441) {
		value = 441;
	} else if (threadIdx.x == 442) {
		value = 442;
	} else if (threadIdx.x == 443) {
		value = 443;
	} else if (threadIdx.x == 444) {
		value = 444;
	} else if (threadIdx.x == 445) {
		value = 445;
	} else if (threadIdx.x == 446) {
		value = 446;
	} else if (threadIdx.x == 447) {
		value = 447;
	} else if (threadIdx.x == 448) {
		value = 448;
	} else if (threadIdx.x == 449) {
		value = 449;
	} else if (threadIdx.x == 450) {
		value = 450;
	} else if (threadIdx.x == 451) {
		value = 451;
	} else if (threadIdx.x == 452) {
		value = 452;
	} else if (threadIdx.x == 453) {
		value = 453;
	} else if (threadIdx.x == 454) {
		value = 454;
	} else if (threadIdx.x == 455) {
		value = 455;
	} else if (threadIdx.x == 456) {
		value = 456;
	} else if (threadIdx.x == 457) {
		value = 457;
	} else if (threadIdx.x == 458) {
		value = 458;
	} else if (threadIdx.x == 459) {
		value = 459;
	} else if (threadIdx.x == 460) {
		value = 460;
	} else if (threadIdx.x == 461) {
		value = 461;
	} else if (threadIdx.x == 462) {
		value = 462;
	} else if (threadIdx.x == 463) {
		value = 463;
	} else if (threadIdx.x == 464) {
		value = 464;
	} else if (threadIdx.x == 465) {
		value = 465;
	} else if (threadIdx.x == 466) {
		value = 466;
	} else if (threadIdx.x == 467) {
		value = 467;
	} else if (threadIdx.x == 468) {
		value = 468;
	} else if (threadIdx.x == 469) {
		value = 469;
	} else if (threadIdx.x == 470) {
		value = 470;
	} else if (threadIdx.x == 471) {
		value = 471;
	} else if (threadIdx.x == 472) {
		value = 472;
	} else if (threadIdx.x == 473) {
		value = 473;
	} else if (threadIdx.x == 474) {
		value = 474;
	} else if (threadIdx.x == 475) {
		value = 475;
	} else if (threadIdx.x == 476) {
		value = 476;
	} else if (threadIdx.x == 477) {
		value = 477;
	} else if (threadIdx.x == 478) {
		value = 478;
	} else if (threadIdx.x == 479) {
		value = 479;
	} else if (threadIdx.x == 480) {
		value = 480;
	} else if (threadIdx.x == 481) {
		value = 481;
	} else if (threadIdx.x == 482) {
		value = 482;
	} else if (threadIdx.x == 483) {
		value = 483;
	} else if (threadIdx.x == 484) {
		value = 484;
	} else if (threadIdx.x == 485) {
		value = 485;
	} else if (threadIdx.x == 486) {
		value = 486;
	} else if (threadIdx.x == 487) {
		value = 487;
	} else if (threadIdx.x == 488) {
		value = 488;
	} else if (threadIdx.x == 489) {
		value = 489;
	} else if (threadIdx.x == 490) {
		value = 490;
	} else if (threadIdx.x == 491) {
		value = 491;
	} else if (threadIdx.x == 492) {
		value = 492;
	} else if (threadIdx.x == 493) {
		value = 493;
	} else if (threadIdx.x == 494) {
		value = 494;
	} else if (threadIdx.x == 495) {
		value = 495;
	} else if (threadIdx.x == 496) {
		value = 496;
	} else if (threadIdx.x == 497) {
		value = 497;
	} else if (threadIdx.x == 498) {
		value = 498;
	} else if (threadIdx.x == 499) {
		value = 499;
	} else if (threadIdx.x == 500) {
		value = 500;
	} else if (threadIdx.x == 501) {
		value = 501;
	} else if (threadIdx.x == 502) {
		value = 502;
	} else if (threadIdx.x == 503) {
		value = 503;
	} else if (threadIdx.x == 504) {
		value = 504;
	} else if (threadIdx.x == 505) {
		value = 505;
	} else if (threadIdx.x == 506) {
		value = 506;
	} else if (threadIdx.x == 507) {
		value = 507;
	} else if (threadIdx.x == 508) {
		value = 508;
	} else if (threadIdx.x == 509) {
		value = 509;
	} else if (threadIdx.x == 510) {
		value = 510;
	} else if (threadIdx.x == 511) {
		value = 511;
	} else if (threadIdx.x == 512) {
		value = 512;
	} else if (threadIdx.x == 513) {
		value = 513;
	} else if (threadIdx.x == 514) {
		value = 514;
	} else if (threadIdx.x == 515) {
		value = 515;
	} else if (threadIdx.x == 516) {
		value = 516;
	} else if (threadIdx.x == 517) {
		value = 517;
	} else if (threadIdx.x == 518) {
		value = 518;
	} else if (threadIdx.x == 519) {
		value = 519;
	} else if (threadIdx.x == 520) {
		value = 520;
	} else if (threadIdx.x == 521) {
		value = 521;
	} else if (threadIdx.x == 522) {
		value = 522;
	} else if (threadIdx.x == 523) {
		value = 523;
	} else if (threadIdx.x == 524) {
		value = 524;
	} else if (threadIdx.x == 525) {
		value = 525;
	} else if (threadIdx.x == 526) {
		value = 526;
	} else if (threadIdx.x == 527) {
		value = 527;
	} else if (threadIdx.x == 528) {
		value = 528;
	} else if (threadIdx.x == 529) {
		value = 529;
	} else if (threadIdx.x == 530) {
		value = 530;
	} else if (threadIdx.x == 531) {
		value = 531;
	} else if (threadIdx.x == 532) {
		value = 532;
	} else if (threadIdx.x == 533) {
		value = 533;
	} else if (threadIdx.x == 534) {
		value = 534;
	} else if (threadIdx.x == 535) {
		value = 535;
	} else if (threadIdx.x == 536) {
		value = 536;
	} else if (threadIdx.x == 537) {
		value = 537;
	} else if (threadIdx.x == 538) {
		value = 538;
	} else if (threadIdx.x == 539) {
		value = 539;
	} else if (threadIdx.x == 540) {
		value = 540;
	} else if (threadIdx.x == 541) {
		value = 541;
	} else if (threadIdx.x == 542) {
		value = 542;
	} else if (threadIdx.x == 543) {
		value = 543;
	} else if (threadIdx.x == 544) {
		value = 544;
	} else if (threadIdx.x == 545) {
		value = 545;
	} else if (threadIdx.x == 546) {
		value = 546;
	} else if (threadIdx.x == 547) {
		value = 547;
	} else if (threadIdx.x == 548) {
		value = 548;
	} else if (threadIdx.x == 549) {
		value = 549;
	} else if (threadIdx.x == 550) {
		value = 550;
	} else if (threadIdx.x == 551) {
		value = 551;
	} else if (threadIdx.x == 552) {
		value = 552;
	} else if (threadIdx.x == 553) {
		value = 553;
	} else if (threadIdx.x == 554) {
		value = 554;
	} else if (threadIdx.x == 555) {
		value = 555;
	} else if (threadIdx.x == 556) {
		value = 556;
	} else if (threadIdx.x == 557) {
		value = 557;
	} else if (threadIdx.x == 558) {
		value = 558;
	} else if (threadIdx.x == 559) {
		value = 559;
	} else if (threadIdx.x == 560) {
		value = 560;
	} else if (threadIdx.x == 561) {
		value = 561;
	} else if (threadIdx.x == 562) {
		value = 562;
	} else if (threadIdx.x == 563) {
		value = 563;
	} else if (threadIdx.x == 564) {
		value = 564;
	} else if (threadIdx.x == 565) {
		value = 565;
	} else if (threadIdx.x == 566) {
		value = 566;
	} else if (threadIdx.x == 567) {
		value = 567;
	} else if (threadIdx.x == 568) {
		value = 568;
	} else if (threadIdx.x == 569) {
		value = 569;
	} else if (threadIdx.x == 570) {
		value = 570;
	} else if (threadIdx.x == 571) {
		value = 571;
	} else if (threadIdx.x == 572) {
		value = 572;
	} else if (threadIdx.x == 573) {
		value = 573;
	} else if (threadIdx.x == 574) {
		value = 574;
	} else if (threadIdx.x == 575) {
		value = 575;
	} else if (threadIdx.x == 576) {
		value = 576;
	} else if (threadIdx.x == 577) {
		value = 577;
	} else if (threadIdx.x == 578) {
		value = 578;
	} else if (threadIdx.x == 579) {
		value = 579;
	} else if (threadIdx.x == 580) {
		value = 580;
	} else if (threadIdx.x == 581) {
		value = 581;
	} else if (threadIdx.x == 582) {
		value = 582;
	} else if (threadIdx.x == 583) {
		value = 583;
	} else if (threadIdx.x == 584) {
		value = 584;
	} else if (threadIdx.x == 585) {
		value = 585;
	} else if (threadIdx.x == 586) {
		value = 586;
	} else if (threadIdx.x == 587) {
		value = 587;
	} else if (threadIdx.x == 588) {
		value = 588;
	} else if (threadIdx.x == 589) {
		value = 589;
	} else if (threadIdx.x == 590) {
		value = 590;
	} else if (threadIdx.x == 591) {
		value = 591;
	} else if (threadIdx.x == 592) {
		value = 592;
	} else if (threadIdx.x == 593) {
		value = 593;
	} else if (threadIdx.x == 594) {
		value = 594;
	} else if (threadIdx.x == 595) {
		value = 595;
	} else if (threadIdx.x == 596) {
		value = 596;
	} else if (threadIdx.x == 597) {
		value = 597;
	} else if (threadIdx.x == 598) {
		value = 598;
	} else if (threadIdx.x == 599) {
		value = 599;
	} else if (threadIdx.x == 600) {
		value = 600;
	} else if (threadIdx.x == 601) {
		value = 601;
	} else if (threadIdx.x == 602) {
		value = 602;
	} else if (threadIdx.x == 603) {
		value = 603;
	} else if (threadIdx.x == 604) {
		value = 604;
	} else if (threadIdx.x == 605) {
		value = 605;
	} else if (threadIdx.x == 606) {
		value = 606;
	} else if (threadIdx.x == 607) {
		value = 607;
	} else if (threadIdx.x == 608) {
		value = 608;
	} else if (threadIdx.x == 609) {
		value = 609;
	} else if (threadIdx.x == 610) {
		value = 610;
	} else if (threadIdx.x == 611) {
		value = 611;
	} else if (threadIdx.x == 612) {
		value = 612;
	} else if (threadIdx.x == 613) {
		value = 613;
	} else if (threadIdx.x == 614) {
		value = 614;
	} else if (threadIdx.x == 615) {
		value = 615;
	} else if (threadIdx.x == 616) {
		value = 616;
	} else if (threadIdx.x == 617) {
		value = 617;
	} else if (threadIdx.x == 618) {
		value = 618;
	} else if (threadIdx.x == 619) {
		value = 619;
	} else if (threadIdx.x == 620) {
		value = 620;
	} else if (threadIdx.x == 621) {
		value = 621;
	} else if (threadIdx.x == 622) {
		value = 622;
	} else if (threadIdx.x == 623) {
		value = 623;
	} else if (threadIdx.x == 624) {
		value = 624;
	} else if (threadIdx.x == 625) {
		value = 625;
	} else if (threadIdx.x == 626) {
		value = 626;
	} else if (threadIdx.x == 627) {
		value = 627;
	} else if (threadIdx.x == 628) {
		value = 628;
	} else if (threadIdx.x == 629) {
		value = 629;
	} else if (threadIdx.x == 630) {
		value = 630;
	} else if (threadIdx.x == 631) {
		value = 631;
	} else if (threadIdx.x == 632) {
		value = 632;
	} else if (threadIdx.x == 633) {
		value = 633;
	} else if (threadIdx.x == 634) {
		value = 634;
	} else if (threadIdx.x == 635) {
		value = 635;
	} else if (threadIdx.x == 636) {
		value = 636;
	} else if (threadIdx.x == 637) {
		value = 637;
	} else if (threadIdx.x == 638) {
		value = 638;
	} else if (threadIdx.x == 639) {
		value = 639;
	} else if (threadIdx.x == 640) {
		value = 640;
	} else if (threadIdx.x == 641) {
		value = 641;
	} else if (threadIdx.x == 642) {
		value = 642;
	} else if (threadIdx.x == 643) {
		value = 643;
	} else if (threadIdx.x == 644) {
		value = 644;
	} else if (threadIdx.x == 645) {
		value = 645;
	} else if (threadIdx.x == 646) {
		value = 646;
	} else if (threadIdx.x == 647) {
		value = 647;
	} else if (threadIdx.x == 648) {
		value = 648;
	} else if (threadIdx.x == 649) {
		value = 649;
	} else if (threadIdx.x == 650) {
		value = 650;
	} else if (threadIdx.x == 651) {
		value = 651;
	} else if (threadIdx.x == 652) {
		value = 652;
	} else if (threadIdx.x == 653) {
		value = 653;
	} else if (threadIdx.x == 654) {
		value = 654;
	} else if (threadIdx.x == 655) {
		value = 655;
	} else if (threadIdx.x == 656) {
		value = 656;
	} else if (threadIdx.x == 657) {
		value = 657;
	} else if (threadIdx.x == 658) {
		value = 658;
	} else if (threadIdx.x == 659) {
		value = 659;
	} else if (threadIdx.x == 660) {
		value = 660;
	} else if (threadIdx.x == 661) {
		value = 661;
	} else if (threadIdx.x == 662) {
		value = 662;
	} else if (threadIdx.x == 663) {
		value = 663;
	} else if (threadIdx.x == 664) {
		value = 664;
	} else if (threadIdx.x == 665) {
		value = 665;
	} else if (threadIdx.x == 666) {
		value = 666;
	} else if (threadIdx.x == 667) {
		value = 667;
	} else if (threadIdx.x == 668) {
		value = 668;
	} else if (threadIdx.x == 669) {
		value = 669;
	} else if (threadIdx.x == 670) {
		value = 670;
	} else if (threadIdx.x == 671) {
		value = 671;
	} else if (threadIdx.x == 672) {
		value = 672;
	} else if (threadIdx.x == 673) {
		value = 673;
	} else if (threadIdx.x == 674) {
		value = 674;
	} else if (threadIdx.x == 675) {
		value = 675;
	} else if (threadIdx.x == 676) {
		value = 676;
	} else if (threadIdx.x == 677) {
		value = 677;
	} else if (threadIdx.x == 678) {
		value = 678;
	} else if (threadIdx.x == 679) {
		value = 679;
	} else if (threadIdx.x == 680) {
		value = 680;
	} else if (threadIdx.x == 681) {
		value = 681;
	} else if (threadIdx.x == 682) {
		value = 682;
	} else if (threadIdx.x == 683) {
		value = 683;
	} else if (threadIdx.x == 684) {
		value = 684;
	} else if (threadIdx.x == 685) {
		value = 685;
	} else if (threadIdx.x == 686) {
		value = 686;
	} else if (threadIdx.x == 687) {
		value = 687;
	} else if (threadIdx.x == 688) {
		value = 688;
	} else if (threadIdx.x == 689) {
		value = 689;
	} else if (threadIdx.x == 690) {
		value = 690;
	} else if (threadIdx.x == 691) {
		value = 691;
	} else if (threadIdx.x == 692) {
		value = 692;
	} else if (threadIdx.x == 693) {
		value = 693;
	} else if (threadIdx.x == 694) {
		value = 694;
	} else if (threadIdx.x == 695) {
		value = 695;
	} else if (threadIdx.x == 696) {
		value = 696;
	} else if (threadIdx.x == 697) {
		value = 697;
	} else if (threadIdx.x == 698) {
		value = 698;
	} else if (threadIdx.x == 699) {
		value = 699;
	} else if (threadIdx.x == 700) {
		value = 700;
	} else if (threadIdx.x == 701) {
		value = 701;
	} else if (threadIdx.x == 702) {
		value = 702;
	} else if (threadIdx.x == 703) {
		value = 703;
	} else if (threadIdx.x == 704) {
		value = 704;
	} else if (threadIdx.x == 705) {
		value = 705;
	} else if (threadIdx.x == 706) {
		value = 706;
	} else if (threadIdx.x == 707) {
		value = 707;
	} else if (threadIdx.x == 708) {
		value = 708;
	} else if (threadIdx.x == 709) {
		value = 709;
	} else if (threadIdx.x == 710) {
		value = 710;
	} else if (threadIdx.x == 711) {
		value = 711;
	} else if (threadIdx.x == 712) {
		value = 712;
	} else if (threadIdx.x == 713) {
		value = 713;
	} else if (threadIdx.x == 714) {
		value = 714;
	} else if (threadIdx.x == 715) {
		value = 715;
	} else if (threadIdx.x == 716) {
		value = 716;
	} else if (threadIdx.x == 717) {
		value = 717;
	} else if (threadIdx.x == 718) {
		value = 718;
	} else if (threadIdx.x == 719) {
		value = 719;
	} else if (threadIdx.x == 720) {
		value = 720;
	} else if (threadIdx.x == 721) {
		value = 721;
	} else if (threadIdx.x == 722) {
		value = 722;
	} else if (threadIdx.x == 723) {
		value = 723;
	} else if (threadIdx.x == 724) {
		value = 724;
	} else if (threadIdx.x == 725) {
		value = 725;
	} else if (threadIdx.x == 726) {
		value = 726;
	} else if (threadIdx.x == 727) {
		value = 727;
	} else if (threadIdx.x == 728) {
		value = 728;
	} else if (threadIdx.x == 729) {
		value = 729;
	} else if (threadIdx.x == 730) {
		value = 730;
	} else if (threadIdx.x == 731) {
		value = 731;
	} else if (threadIdx.x == 732) {
		value = 732;
	} else if (threadIdx.x == 733) {
		value = 733;
	} else if (threadIdx.x == 734) {
		value = 734;
	} else if (threadIdx.x == 735) {
		value = 735;
	} else if (threadIdx.x == 736) {
		value = 736;
	} else if (threadIdx.x == 737) {
		value = 737;
	} else if (threadIdx.x == 738) {
		value = 738;
	} else if (threadIdx.x == 739) {
		value = 739;
	} else if (threadIdx.x == 740) {
		value = 740;
	} else if (threadIdx.x == 741) {
		value = 741;
	} else if (threadIdx.x == 742) {
		value = 742;
	} else if (threadIdx.x == 743) {
		value = 743;
	} else if (threadIdx.x == 744) {
		value = 744;
	} else if (threadIdx.x == 745) {
		value = 745;
	} else if (threadIdx.x == 746) {
		value = 746;
	} else if (threadIdx.x == 747) {
		value = 747;
	} else if (threadIdx.x == 748) {
		value = 748;
	} else if (threadIdx.x == 749) {
		value = 749;
	} else if (threadIdx.x == 750) {
		value = 750;
	} else if (threadIdx.x == 751) {
		value = 751;
	} else if (threadIdx.x == 752) {
		value = 752;
	} else if (threadIdx.x == 753) {
		value = 753;
	} else if (threadIdx.x == 754) {
		value = 754;
	} else if (threadIdx.x == 755) {
		value = 755;
	} else if (threadIdx.x == 756) {
		value = 756;
	} else if (threadIdx.x == 757) {
		value = 757;
	} else if (threadIdx.x == 758) {
		value = 758;
	} else if (threadIdx.x == 759) {
		value = 759;
	} else if (threadIdx.x == 760) {
		value = 760;
	} else if (threadIdx.x == 761) {
		value = 761;
	} else if (threadIdx.x == 762) {
		value = 762;
	} else if (threadIdx.x == 763) {
		value = 763;
	} else if (threadIdx.x == 764) {
		value = 764;
	} else if (threadIdx.x == 765) {
		value = 765;
	} else if (threadIdx.x == 766) {
		value = 766;
	} else if (threadIdx.x == 767) {
		value = 767;
	} else if (threadIdx.x == 768) {
		value = 768;
	} else if (threadIdx.x == 769) {
		value = 769;
	} else if (threadIdx.x == 770) {
		value = 770;
	} else if (threadIdx.x == 771) {
		value = 771;
	} else if (threadIdx.x == 772) {
		value = 772;
	} else if (threadIdx.x == 773) {
		value = 773;
	} else if (threadIdx.x == 774) {
		value = 774;
	} else if (threadIdx.x == 775) {
		value = 775;
	} else if (threadIdx.x == 776) {
		value = 776;
	} else if (threadIdx.x == 777) {
		value = 777;
	} else if (threadIdx.x == 778) {
		value = 778;
	} else if (threadIdx.x == 779) {
		value = 779;
	} else if (threadIdx.x == 780) {
		value = 780;
	} else if (threadIdx.x == 781) {
		value = 781;
	} else if (threadIdx.x == 782) {
		value = 782;
	} else if (threadIdx.x == 783) {
		value = 783;
	} else if (threadIdx.x == 784) {
		value = 784;
	} else if (threadIdx.x == 785) {
		value = 785;
	} else if (threadIdx.x == 786) {
		value = 786;
	} else if (threadIdx.x == 787) {
		value = 787;
	} else if (threadIdx.x == 788) {
		value = 788;
	} else if (threadIdx.x == 789) {
		value = 789;
	} else if (threadIdx.x == 790) {
		value = 790;
	} else if (threadIdx.x == 791) {
		value = 791;
	} else if (threadIdx.x == 792) {
		value = 792;
	} else if (threadIdx.x == 793) {
		value = 793;
	} else if (threadIdx.x == 794) {
		value = 794;
	} else if (threadIdx.x == 795) {
		value = 795;
	} else if (threadIdx.x == 796) {
		value = 796;
	} else if (threadIdx.x == 797) {
		value = 797;
	} else if (threadIdx.x == 798) {
		value = 798;
	} else if (threadIdx.x == 799) {
		value = 799;
	} else if (threadIdx.x == 800) {
		value = 800;
	} else if (threadIdx.x == 801) {
		value = 801;
	} else if (threadIdx.x == 802) {
		value = 802;
	} else if (threadIdx.x == 803) {
		value = 803;
	} else if (threadIdx.x == 804) {
		value = 804;
	} else if (threadIdx.x == 805) {
		value = 805;
	} else if (threadIdx.x == 806) {
		value = 806;
	} else if (threadIdx.x == 807) {
		value = 807;
	} else if (threadIdx.x == 808) {
		value = 808;
	} else if (threadIdx.x == 809) {
		value = 809;
	} else if (threadIdx.x == 810) {
		value = 810;
	} else if (threadIdx.x == 811) {
		value = 811;
	} else if (threadIdx.x == 812) {
		value = 812;
	} else if (threadIdx.x == 813) {
		value = 813;
	} else if (threadIdx.x == 814) {
		value = 814;
	} else if (threadIdx.x == 815) {
		value = 815;
	} else if (threadIdx.x == 816) {
		value = 816;
	} else if (threadIdx.x == 817) {
		value = 817;
	} else if (threadIdx.x == 818) {
		value = 818;
	} else if (threadIdx.x == 819) {
		value = 819;
	} else if (threadIdx.x == 820) {
		value = 820;
	} else if (threadIdx.x == 821) {
		value = 821;
	} else if (threadIdx.x == 822) {
		value = 822;
	} else if (threadIdx.x == 823) {
		value = 823;
	} else if (threadIdx.x == 824) {
		value = 824;
	} else if (threadIdx.x == 825) {
		value = 825;
	} else if (threadIdx.x == 826) {
		value = 826;
	} else if (threadIdx.x == 827) {
		value = 827;
	} else if (threadIdx.x == 828) {
		value = 828;
	} else if (threadIdx.x == 829) {
		value = 829;
	} else if (threadIdx.x == 830) {
		value = 830;
	} else if (threadIdx.x == 831) {
		value = 831;
	} else if (threadIdx.x == 832) {
		value = 832;
	} else if (threadIdx.x == 833) {
		value = 833;
	} else if (threadIdx.x == 834) {
		value = 834;
	} else if (threadIdx.x == 835) {
		value = 835;
	} else if (threadIdx.x == 836) {
		value = 836;
	} else if (threadIdx.x == 837) {
		value = 837;
	} else if (threadIdx.x == 838) {
		value = 838;
	} else if (threadIdx.x == 839) {
		value = 839;
	} else if (threadIdx.x == 840) {
		value = 840;
	} else if (threadIdx.x == 841) {
		value = 841;
	} else if (threadIdx.x == 842) {
		value = 842;
	} else if (threadIdx.x == 843) {
		value = 843;
	} else if (threadIdx.x == 844) {
		value = 844;
	} else if (threadIdx.x == 845) {
		value = 845;
	} else if (threadIdx.x == 846) {
		value = 846;
	} else if (threadIdx.x == 847) {
		value = 847;
	} else if (threadIdx.x == 848) {
		value = 848;
	} else if (threadIdx.x == 849) {
		value = 849;
	} else if (threadIdx.x == 850) {
		value = 850;
	} else if (threadIdx.x == 851) {
		value = 851;
	} else if (threadIdx.x == 852) {
		value = 852;
	} else if (threadIdx.x == 853) {
		value = 853;
	} else if (threadIdx.x == 854) {
		value = 854;
	} else if (threadIdx.x == 855) {
		value = 855;
	} else if (threadIdx.x == 856) {
		value = 856;
	} else if (threadIdx.x == 857) {
		value = 857;
	} else if (threadIdx.x == 858) {
		value = 858;
	} else if (threadIdx.x == 859) {
		value = 859;
	} else if (threadIdx.x == 860) {
		value = 860;
	} else if (threadIdx.x == 861) {
		value = 861;
	} else if (threadIdx.x == 862) {
		value = 862;
	} else if (threadIdx.x == 863) {
		value = 863;
	} else if (threadIdx.x == 864) {
		value = 864;
	} else if (threadIdx.x == 865) {
		value = 865;
	} else if (threadIdx.x == 866) {
		value = 866;
	} else if (threadIdx.x == 867) {
		value = 867;
	} else if (threadIdx.x == 868) {
		value = 868;
	} else if (threadIdx.x == 869) {
		value = 869;
	} else if (threadIdx.x == 870) {
		value = 870;
	} else if (threadIdx.x == 871) {
		value = 871;
	} else if (threadIdx.x == 872) {
		value = 872;
	} else if (threadIdx.x == 873) {
		value = 873;
	} else if (threadIdx.x == 874) {
		value = 874;
	} else if (threadIdx.x == 875) {
		value = 875;
	} else if (threadIdx.x == 876) {
		value = 876;
	} else if (threadIdx.x == 877) {
		value = 877;
	} else if (threadIdx.x == 878) {
		value = 878;
	} else if (threadIdx.x == 879) {
		value = 879;
	} else if (threadIdx.x == 880) {
		value = 880;
	} else if (threadIdx.x == 881) {
		value = 881;
	} else if (threadIdx.x == 882) {
		value = 882;
	} else if (threadIdx.x == 883) {
		value = 883;
	} else if (threadIdx.x == 884) {
		value = 884;
	} else if (threadIdx.x == 885) {
		value = 885;
	} else if (threadIdx.x == 886) {
		value = 886;
	} else if (threadIdx.x == 887) {
		value = 887;
	} else if (threadIdx.x == 888) {
		value = 888;
	} else if (threadIdx.x == 889) {
		value = 889;
	} else if (threadIdx.x == 890) {
		value = 890;
	} else if (threadIdx.x == 891) {
		value = 891;
	} else if (threadIdx.x == 892) {
		value = 892;
	} else if (threadIdx.x == 893) {
		value = 893;
	} else if (threadIdx.x == 894) {
		value = 894;
	} else if (threadIdx.x == 895) {
		value = 895;
	} else if (threadIdx.x == 896) {
		value = 896;
	} else if (threadIdx.x == 897) {
		value = 897;
	} else if (threadIdx.x == 898) {
		value = 898;
	} else if (threadIdx.x == 899) {
		value = 899;
	} else if (threadIdx.x == 900) {
		value = 900;
	} else if (threadIdx.x == 901) {
		value = 901;
	} else if (threadIdx.x == 902) {
		value = 902;
	} else if (threadIdx.x == 903) {
		value = 903;
	} else if (threadIdx.x == 904) {
		value = 904;
	} else if (threadIdx.x == 905) {
		value = 905;
	} else if (threadIdx.x == 906) {
		value = 906;
	} else if (threadIdx.x == 907) {
		value = 907;
	} else if (threadIdx.x == 908) {
		value = 908;
	} else if (threadIdx.x == 909) {
		value = 909;
	} else if (threadIdx.x == 910) {
		value = 910;
	} else if (threadIdx.x == 911) {
		value = 911;
	} else if (threadIdx.x == 912) {
		value = 912;
	} else if (threadIdx.x == 913) {
		value = 913;
	} else if (threadIdx.x == 914) {
		value = 914;
	} else if (threadIdx.x == 915) {
		value = 915;
	} else if (threadIdx.x == 916) {
		value = 916;
	} else if (threadIdx.x == 917) {
		value = 917;
	} else if (threadIdx.x == 918) {
		value = 918;
	} else if (threadIdx.x == 919) {
		value = 919;
	} else if (threadIdx.x == 920) {
		value = 920;
	} else if (threadIdx.x == 921) {
		value = 921;
	} else if (threadIdx.x == 922) {
		value = 922;
	} else if (threadIdx.x == 923) {
		value = 923;
	} else if (threadIdx.x == 924) {
		value = 924;
	} else if (threadIdx.x == 925) {
		value = 925;
	} else if (threadIdx.x == 926) {
		value = 926;
	} else if (threadIdx.x == 927) {
		value = 927;
	} else if (threadIdx.x == 928) {
		value = 928;
	} else if (threadIdx.x == 929) {
		value = 929;
	} else if (threadIdx.x == 930) {
		value = 930;
	} else if (threadIdx.x == 931) {
		value = 931;
	} else if (threadIdx.x == 932) {
		value = 932;
	} else if (threadIdx.x == 933) {
		value = 933;
	} else if (threadIdx.x == 934) {
		value = 934;
	} else if (threadIdx.x == 935) {
		value = 935;
	} else if (threadIdx.x == 936) {
		value = 936;
	} else if (threadIdx.x == 937) {
		value = 937;
	} else if (threadIdx.x == 938) {
		value = 938;
	} else if (threadIdx.x == 939) {
		value = 939;
	} else if (threadIdx.x == 940) {
		value = 940;
	} else if (threadIdx.x == 941) {
		value = 941;
	} else if (threadIdx.x == 942) {
		value = 942;
	} else if (threadIdx.x == 943) {
		value = 943;
	} else if (threadIdx.x == 944) {
		value = 944;
	} else if (threadIdx.x == 945) {
		value = 945;
	} else if (threadIdx.x == 946) {
		value = 946;
	} else if (threadIdx.x == 947) {
		value = 947;
	} else if (threadIdx.x == 948) {
		value = 948;
	} else if (threadIdx.x == 949) {
		value = 949;
	} else if (threadIdx.x == 950) {
		value = 950;
	} else if (threadIdx.x == 951) {
		value = 951;
	} else if (threadIdx.x == 952) {
		value = 952;
	} else if (threadIdx.x == 953) {
		value = 953;
	} else if (threadIdx.x == 954) {
		value = 954;
	} else if (threadIdx.x == 955) {
		value = 955;
	} else if (threadIdx.x == 956) {
		value = 956;
	} else if (threadIdx.x == 957) {
		value = 957;
	} else if (threadIdx.x == 958) {
		value = 958;
	} else if (threadIdx.x == 959) {
		value = 959;
	} else if (threadIdx.x == 960) {
		value = 960;
	} else if (threadIdx.x == 961) {
		value = 961;
	} else if (threadIdx.x == 962) {
		value = 962;
	} else if (threadIdx.x == 963) {
		value = 963;
	} else if (threadIdx.x == 964) {
		value = 964;
	} else if (threadIdx.x == 965) {
		value = 965;
	} else if (threadIdx.x == 966) {
		value = 966;
	} else if (threadIdx.x == 967) {
		value = 967;
	} else if (threadIdx.x == 968) {
		value = 968;
	} else if (threadIdx.x == 969) {
		value = 969;
	} else if (threadIdx.x == 970) {
		value = 970;
	} else if (threadIdx.x == 971) {
		value = 971;
	} else if (threadIdx.x == 972) {
		value = 972;
	} else if (threadIdx.x == 973) {
		value = 973;
	} else if (threadIdx.x == 974) {
		value = 974;
	} else if (threadIdx.x == 975) {
		value = 975;
	} else if (threadIdx.x == 976) {
		value = 976;
	} else if (threadIdx.x == 977) {
		value = 977;
	} else if (threadIdx.x == 978) {
		value = 978;
	} else if (threadIdx.x == 979) {
		value = 979;
	} else if (threadIdx.x == 980) {
		value = 980;
	} else if (threadIdx.x == 981) {
		value = 981;
	} else if (threadIdx.x == 982) {
		value = 982;
	} else if (threadIdx.x == 983) {
		value = 983;
	} else if (threadIdx.x == 984) {
		value = 984;
	} else if (threadIdx.x == 985) {
		value = 985;
	} else if (threadIdx.x == 986) {
		value = 986;
	} else if (threadIdx.x == 987) {
		value = 987;
	} else if (threadIdx.x == 988) {
		value = 988;
	} else if (threadIdx.x == 989) {
		value = 989;
	} else if (threadIdx.x == 990) {
		value = 990;
	} else if (threadIdx.x == 991) {
		value = 991;
	} else if (threadIdx.x == 992) {
		value = 992;
	} else if (threadIdx.x == 993) {
		value = 993;
	} else if (threadIdx.x == 994) {
		value = 994;
	} else if (threadIdx.x == 995) {
		value = 995;
	} else if (threadIdx.x == 996) {
		value = 996;
	} else if (threadIdx.x == 997) {
		value = 997;
	} else if (threadIdx.x == 998) {
		value = 998;
	} else if (threadIdx.x == 999) {
		value = 999;
	} else if (threadIdx.x == 1000) {
		value = 1000;
	} else if (threadIdx.x == 1001) {
		value = 1001;
	} else if (threadIdx.x == 1002) {
		value = 1002;
	} else if (threadIdx.x == 1003) {
		value = 1003;
	} else if (threadIdx.x == 1004) {
		value = 1004;
	} else if (threadIdx.x == 1005) {
		value = 1005;
	} else if (threadIdx.x == 1006) {
		value = 1006;
	} else if (threadIdx.x == 1007) {
		value = 1007;
	} else if (threadIdx.x == 1008) {
		value = 1008;
	} else if (threadIdx.x == 1009) {
		value = 1009;
	} else if (threadIdx.x == 1010) {
		value = 1010;
	} else if (threadIdx.x == 1011) {
		value = 1011;
	} else if (threadIdx.x == 1012) {
		value = 1012;
	} else if (threadIdx.x == 1013) {
		value = 1013;
	} else if (threadIdx.x == 1014) {
		value = 1014;
	} else if (threadIdx.x == 1015) {
		value = 1015;
	} else if (threadIdx.x == 1016) {
		value = 1016;
	} else if (threadIdx.x == 1017) {
		value = 1017;
	} else if (threadIdx.x == 1018) {
		value = 1018;
	} else if (threadIdx.x == 1019) {
		value = 1019;
	} else if (threadIdx.x == 1020) {
		value = 1020;
	} else if (threadIdx.x == 1021) {
		value = 1021;
	} else if (threadIdx.x == 1022) {
		value = 1022;
	} else if (threadIdx.x == 1023) {
		value = 1023;
	}
	dst[i] = value;

}

#endif /* BRANCH_KERNEL_H_ */
