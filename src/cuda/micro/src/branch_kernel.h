#ifndef BRANCH_KERNEL_H_
#define BRANCH_KERNEL_H_

#include <cstdint>


template<typename int_t>
__global__ void int_branch_kernel(int_t* dst_1, int_t* dst_2, int_t* dst_3, uint32_t op) {
	const int_t i = (blockDim.x * blockIdx.x + threadIdx.x);
	int_t value = i;
	int_t to_store = (i + 1) % op;
#pragma unroll 16
	for(int opi = 0; opi < op; opi++) {
		if (threadIdx.x == 0) {
			value = to_store;
		} else if (threadIdx.x == 1) {
			value = to_store;
		} else if (threadIdx.x == 2) {
			value = to_store;
		} else if (threadIdx.x == 3) {
			value = to_store;
		} else if (threadIdx.x == 4) {
			value = to_store;
		} else if (threadIdx.x == 5) {
			value = to_store;
		} else if (threadIdx.x == 6) {
			value = to_store;
		} else if (threadIdx.x == 7) {
			value = to_store;
		} else if (threadIdx.x == 8) {
			value = to_store;
		} else if (threadIdx.x == 9) {
			value = to_store;
		} else if (threadIdx.x == 10) {
			value = to_store;
		} else if (threadIdx.x == 11) {
			value = to_store;
		} else if (threadIdx.x == 12) {
			value = to_store;
		} else if (threadIdx.x == 13) {
			value = to_store;
		} else if (threadIdx.x == 14) {
			value = to_store;
		} else if (threadIdx.x == 15) {
			value = to_store;
		} else if (threadIdx.x == 16) {
			value = to_store;
		} else if (threadIdx.x == 17) {
			value = to_store;
		} else if (threadIdx.x == 18) {
			value = to_store;
		} else if (threadIdx.x == 19) {
			value = to_store;
		} else if (threadIdx.x == 20) {
			value = to_store;
		} else if (threadIdx.x == 21) {
			value = to_store;
		} else if (threadIdx.x == 22) {
			value = to_store;
		} else if (threadIdx.x == 23) {
			value = to_store;
		} else if (threadIdx.x == 24) {
			value = to_store;
		} else if (threadIdx.x == 25) {
			value = to_store;
			continue;
		} else if (threadIdx.x == 26) {
			value = to_store;
		} else if (threadIdx.x == 27) {
			value = to_store;
		} else if (threadIdx.x == 28) {
			value = to_store;
		} else if (threadIdx.x == 29) {
			value = to_store;
		} else if (threadIdx.x == 30) {
			value = to_store;
		} else if (threadIdx.x == 31) {
			value = to_store;
		} else if (threadIdx.x == 32) {
			value = to_store;
		} else if (threadIdx.x == 33) {
			value = to_store;
		} else if (threadIdx.x == 34) {
			value = to_store;
		} else if (threadIdx.x == 35) {
			value = to_store;
		} else if (threadIdx.x == 36) {
			value = to_store;
		} else if (threadIdx.x == 37) {
			value = to_store;
		} else if (threadIdx.x == 38) {
			value = to_store;
		} else if (threadIdx.x == 39) {
			value = to_store;
		} else if (threadIdx.x == 40) {
			value = to_store;
		} else if (threadIdx.x == 41) {
			value = to_store;
		} else if (threadIdx.x == 42) {
			value = to_store;
		} else if (threadIdx.x == 43) {
			value = to_store;
		} else if (threadIdx.x == 44) {
			value = to_store;
		} else if (threadIdx.x == 45) {
			value = to_store;
		} else if (threadIdx.x == 46) {
			value = to_store;
		} else if (threadIdx.x == 47) {
			value = to_store;
		} else if (threadIdx.x == 48) {
			value = to_store;
		} else if (threadIdx.x == 49) {
			value = to_store;
		} else if (threadIdx.x == 50) {
			value = to_store;
		} else if (threadIdx.x == 51) {
			value = to_store;
		} else if (threadIdx.x == 52) {
			value = to_store;
		} else if (threadIdx.x == 53) {
			value = to_store;
		} else if (threadIdx.x == 54) {
			value = to_store;
		} else if (threadIdx.x == 55) {
			value = to_store;
		} else if (threadIdx.x == 56) {
			value = to_store;
		} else if (threadIdx.x == 57) {
			value = to_store;
		} else if (threadIdx.x == 58) {
			value = to_store;
		} else if (threadIdx.x == 59) {
			value = to_store;
		} else if (threadIdx.x == 60) {
			value = to_store;
		} else if (threadIdx.x == 61) {
			value = to_store;
		} else if (threadIdx.x == 62) {
			value = to_store;
		} else if (threadIdx.x == 63) {
			value = to_store;
		} else if (threadIdx.x == 64) {
			value = to_store;
		} else if (threadIdx.x == 65) {
			value = to_store;
		} else if (threadIdx.x == 66) {
			value = to_store;
		} else if (threadIdx.x == 67) {
			value = to_store;
		} else if (threadIdx.x == 68) {
			value = to_store;
		} else if (threadIdx.x == 69) {
			value = to_store;
		} else if (threadIdx.x == 70) {
			value = to_store;
		} else if (threadIdx.x == 71) {
			value = to_store;
		} else if (threadIdx.x == 72) {
			value = to_store;
		} else if (threadIdx.x == 73) {
			value = to_store;
		} else if (threadIdx.x == 74) {
			value = to_store;
		} else if (threadIdx.x == 75) {
			value = to_store;
		} else if (threadIdx.x == 76) {
			value = to_store;
		} else if (threadIdx.x == 77) {
			value = to_store;
		} else if (threadIdx.x == 78) {
			value = to_store;
		} else if (threadIdx.x == 79) {
			value = to_store;
		} else if (threadIdx.x == 80) {
			value = to_store;
		} else if (threadIdx.x == 81) {
			value = to_store;
		} else if (threadIdx.x == 82) {
			value = to_store;
		} else if (threadIdx.x == 83) {
			value = to_store;
		} else if (threadIdx.x == 84) {
			value = to_store;
		} else if (threadIdx.x == 85) {
			value = to_store;
		} else if (threadIdx.x == 86) {
			value = to_store;
		} else if (threadIdx.x == 87) {
			value = to_store;
		} else if (threadIdx.x == 88) {
			value = to_store;
		} else if (threadIdx.x == 89) {
			value = to_store;
		} else if (threadIdx.x == 90) {
			value = to_store;
		} else if (threadIdx.x == 91) {
			value = to_store;
		} else if (threadIdx.x == 92) {
			value = to_store;
		} else if (threadIdx.x == 93) {
			value = to_store;
		} else if (threadIdx.x == 94) {
			value = to_store;
		} else if (threadIdx.x == 95) {
			value = to_store;
		} else if (threadIdx.x == 96) {
			value = to_store;
		} else if (threadIdx.x == 97) {
			value = to_store;
		} else if (threadIdx.x == 98) {
			value = to_store;
		} else if (threadIdx.x == 99) {
			value = to_store;
		} else if (threadIdx.x == 100) {
			value = to_store;
		} else if (threadIdx.x == 101) {
			value = to_store;
		} else if (threadIdx.x == 102) {
			value = to_store;
		} else if (threadIdx.x == 103) {
			value = to_store;
		} else if (threadIdx.x == 104) {
			value = to_store;
		} else if (threadIdx.x == 105) {
			value = to_store;
		} else if (threadIdx.x == 106) {
			value = to_store;
		} else if (threadIdx.x == 107) {
			value = to_store;
		} else if (threadIdx.x == 108) {
			value = to_store;
		} else if (threadIdx.x == 109) {
			value = to_store;
		} else if (threadIdx.x == 110) {
			value = to_store;
		} else if (threadIdx.x == 111) {
			value = to_store;
		} else if (threadIdx.x == 112) {
			value = to_store;
		} else if (threadIdx.x == 113) {
			value = to_store;
		} else if (threadIdx.x == 114) {
			value = to_store;
		} else if (threadIdx.x == 115) {
			value = to_store;
		} else if (threadIdx.x == 116) {
			value = to_store;
		} else if (threadIdx.x == 117) {
			value = to_store;
		} else if (threadIdx.x == 118) {
			value = to_store;
		} else if (threadIdx.x == 119) {
			value = to_store;
		} else if (threadIdx.x == 120) {
			value = to_store;
		} else if (threadIdx.x == 121) {
			value = to_store;
		} else if (threadIdx.x == 122) {
			value = to_store;
		} else if (threadIdx.x == 123) {
			value = to_store;
		} else if (threadIdx.x == 124) {
			value = to_store;
		} else if (threadIdx.x == 125) {
			value = to_store;
		} else if (threadIdx.x == 126) {
			value = to_store;
		} else if (threadIdx.x == 127) {
			value = to_store;
		} else if (threadIdx.x == 128) {
			value = to_store;
		} else if (threadIdx.x == 129) {
			value = to_store;
		} else if (threadIdx.x == 130) {
			value = to_store;
		} else if (threadIdx.x == 131) {
			value = to_store;
		} else if (threadIdx.x == 132) {
			value = to_store;
		} else if (threadIdx.x == 133) {
			value = to_store;
		} else if (threadIdx.x == 134) {
			value = to_store;
		} else if (threadIdx.x == 135) {
			value = to_store;
		} else if (threadIdx.x == 136) {
			value = to_store;
		} else if (threadIdx.x == 137) {
			value = to_store;
		} else if (threadIdx.x == 138) {
			value = to_store;
		} else if (threadIdx.x == 139) {
			value = to_store;
		} else if (threadIdx.x == 140) {
			value = to_store;
		} else if (threadIdx.x == 141) {
			value = to_store;
		} else if (threadIdx.x == 142) {
			value = to_store;
		} else if (threadIdx.x == 143) {
			value = to_store;
		} else if (threadIdx.x == 144) {
			value = to_store;
		} else if (threadIdx.x == 145) {
			value = to_store;
		} else if (threadIdx.x == 146) {
			value = to_store;
		} else if (threadIdx.x == 147) {
			value = to_store;
		} else if (threadIdx.x == 148) {
			value = to_store;
		} else if (threadIdx.x == 149) {
			value = to_store;
		} else if (threadIdx.x == 150) {
			value = to_store;
		} else if (threadIdx.x == 151) {
			value = to_store;
		} else if (threadIdx.x == 152) {
			value = to_store;
		} else if (threadIdx.x == 153) {
			value = to_store;
		} else if (threadIdx.x == 154) {
			value = to_store;
		} else if (threadIdx.x == 155) {
			value = to_store;
		} else if (threadIdx.x == 156) {
			value = to_store;
		} else if (threadIdx.x == 157) {
			value = to_store;
		} else if (threadIdx.x == 158) {
			value = to_store;
		} else if (threadIdx.x == 159) {
			value = to_store;
		} else if (threadIdx.x == 160) {
			value = to_store;
		} else if (threadIdx.x == 161) {
			value = to_store;
		} else if (threadIdx.x == 162) {
			value = to_store;
		} else if (threadIdx.x == 163) {
			value = to_store;
		} else if (threadIdx.x == 164) {
			value = to_store;
		} else if (threadIdx.x == 165) {
			value = to_store;
		} else if (threadIdx.x == 166) {
			value = to_store;
		} else if (threadIdx.x == 167) {
			value = to_store;
		} else if (threadIdx.x == 168) {
			value = to_store;
		} else if (threadIdx.x == 169) {
			value = to_store;
		} else if (threadIdx.x == 170) {
			value = to_store;
		} else if (threadIdx.x == 171) {
			value = to_store;
		} else if (threadIdx.x == 172) {
			value = to_store;
		} else if (threadIdx.x == 173) {
			value = to_store;
		} else if (threadIdx.x == 174) {
			value = to_store;
		} else if (threadIdx.x == 175) {
			value = to_store;
		} else if (threadIdx.x == 176) {
			value = to_store;
		} else if (threadIdx.x == 177) {
			value = to_store;
		} else if (threadIdx.x == 178) {
			value = to_store;
		} else if (threadIdx.x == 179) {
			value = to_store;
		} else if (threadIdx.x == 180) {
			value = to_store;
		} else if (threadIdx.x == 181) {
			value = to_store;
		} else if (threadIdx.x == 182) {
			value = to_store;
		} else if (threadIdx.x == 183) {
			value = to_store;
		} else if (threadIdx.x == 184) {
			value = to_store;
		} else if (threadIdx.x == 185) {
			value = to_store;
		} else if (threadIdx.x == 186) {
			value = to_store;
		} else if (threadIdx.x == 187) {
			value = to_store;
		} else if (threadIdx.x == 188) {
			value = to_store;
		} else if (threadIdx.x == 189) {
			value = to_store;
		} else if (threadIdx.x == 190) {
			value = to_store;
		} else if (threadIdx.x == 191) {
			value = to_store;
		} else if (threadIdx.x == 192) {
			value = to_store;
		} else if (threadIdx.x == 193) {
			value = to_store;
		} else if (threadIdx.x == 194) {
			value = to_store;
		} else if (threadIdx.x == 195) {
			value = to_store;
		} else if (threadIdx.x == 196) {
			value = to_store;
		} else if (threadIdx.x == 197) {
			value = to_store;
		} else if (threadIdx.x == 198) {
			value = to_store;
		} else if (threadIdx.x == 199) {
			value = to_store;
		} else if (threadIdx.x == 200) {
			value = to_store;
		} else if (threadIdx.x == 201) {
			value = to_store;
		} else if (threadIdx.x == 202) {
			value = to_store;
		} else if (threadIdx.x == 203) {
			value = to_store;
		} else if (threadIdx.x == 204) {
			value = to_store;
		} else if (threadIdx.x == 205) {
			value = to_store;
		} else if (threadIdx.x == 206) {
			value = to_store;
		} else if (threadIdx.x == 207) {
			value = to_store;
		} else if (threadIdx.x == 208) {
			value = to_store;
		} else if (threadIdx.x == 209) {
			value = to_store;
		} else if (threadIdx.x == 210) {
			value = to_store;
		} else if (threadIdx.x == 211) {
			value = to_store;
		} else if (threadIdx.x == 212) {
			value = to_store;
		} else if (threadIdx.x == 213) {
			value = to_store;
		} else if (threadIdx.x == 214) {
			value = to_store;
		} else if (threadIdx.x == 215) {
			value = to_store;
		} else if (threadIdx.x == 216) {
			value = to_store;
		} else if (threadIdx.x == 217) {
			value = to_store;
		} else if (threadIdx.x == 218) {
			value = to_store;
		} else if (threadIdx.x == 219) {
			value = to_store;
		} else if (threadIdx.x == 220) {
			value = to_store;
		} else if (threadIdx.x == 221) {
			value = to_store;
		} else if (threadIdx.x == 222) {
			value = to_store;
		} else if (threadIdx.x == 223) {
			value = to_store;
		} else if (threadIdx.x == 224) {
			value = to_store;
		} else if (threadIdx.x == 225) {
			value = to_store;
		} else if (threadIdx.x == 226) {
			value = to_store;
		} else if (threadIdx.x == 227) {
			value = to_store;
		} else if (threadIdx.x == 228) {
			value = to_store;
		} else if (threadIdx.x == 229) {
			value = to_store;
		} else if (threadIdx.x == 230) {
			value = to_store;
		} else if (threadIdx.x == 231) {
			value = to_store;
		} else if (threadIdx.x == 232) {
			value = to_store;
		} else if (threadIdx.x == 233) {
			value = to_store;
		} else if (threadIdx.x == 234) {
			value = to_store;
		} else if (threadIdx.x == 235) {
			value = to_store;
		} else if (threadIdx.x == 236) {
			value = to_store;
		} else if (threadIdx.x == 237) {
			value = to_store;
		} else if (threadIdx.x == 238) {
			value = to_store;
		} else if (threadIdx.x == 239) {
			value = to_store;
		} else if (threadIdx.x == 240) {
			value = to_store;
		} else if (threadIdx.x == 241) {
			value = to_store;
		} else if (threadIdx.x == 242) {
			value = to_store;
		} else if (threadIdx.x == 243) {
			value = to_store;
		} else if (threadIdx.x == 244) {
			value = to_store;
		} else if (threadIdx.x == 245) {
			value = to_store;
		} else if (threadIdx.x == 246) {
			value = to_store;
		} else if (threadIdx.x == 247) {
			value = to_store;
		} else if (threadIdx.x == 248) {
			value = to_store;
		} else if (threadIdx.x == 249) {
			value = to_store;
		} else if (threadIdx.x == 250) {
			value = to_store;
		} else if (threadIdx.x == 251) {
			value = to_store;
		} else if (threadIdx.x == 252) {
			value = to_store;
		} else if (threadIdx.x == 253) {
			value = to_store;
		} else if (threadIdx.x == 254) {
			value = to_store;
		} else if (threadIdx.x == 255) {
			value = to_store;
		} else if (threadIdx.x == 256) {
			value = to_store;
		} else if (threadIdx.x == 257) {
			value = to_store;
		} else if (threadIdx.x == 258) {
			value = to_store;
		} else if (threadIdx.x == 259) {
			value = to_store;
		} else if (threadIdx.x == 260) {
			value = to_store;
		} else if (threadIdx.x == 261) {
			value = to_store;
		} else if (threadIdx.x == 262) {
			value = to_store;
		} else if (threadIdx.x == 263) {
			value = to_store;
		} else if (threadIdx.x == 264) {
			value = to_store;
		} else if (threadIdx.x == 265) {
			value = to_store;
		} else if (threadIdx.x == 266) {
			value = to_store;
		} else if (threadIdx.x == 267) {
			value = to_store;
		} else if (threadIdx.x == 268) {
			value = to_store;
		} else if (threadIdx.x == 269) {
			value = to_store;
		} else if (threadIdx.x == 270) {
			value = to_store;
		} else if (threadIdx.x == 271) {
			value = to_store;
		} else if (threadIdx.x == 272) {
			value = to_store;
		} else if (threadIdx.x == 273) {
			value = to_store;
		} else if (threadIdx.x == 274) {
			value = to_store;
		} else if (threadIdx.x == 275) {
			value = to_store;
		} else if (threadIdx.x == 276) {
			value = to_store;
		} else if (threadIdx.x == 277) {
			value = to_store;
		} else if (threadIdx.x == 278) {
			value = to_store;
		} else if (threadIdx.x == 279) {
			value = to_store;
		} else if (threadIdx.x == 280) {
			value = to_store;
		} else if (threadIdx.x == 281) {
			value = to_store;
		} else if (threadIdx.x == 282) {
			value = to_store;
		} else if (threadIdx.x == 283) {
			value = to_store;
		} else if (threadIdx.x == 284) {
			value = to_store;
		} else if (threadIdx.x == 285) {
			value = to_store;
		} else if (threadIdx.x == 286) {
			value = to_store;
		} else if (threadIdx.x == 287) {
			value = to_store;
		} else if (threadIdx.x == 288) {
			value = to_store;
		} else if (threadIdx.x == 289) {
			value = to_store;
		} else if (threadIdx.x == 290) {
			value = to_store;
		} else if (threadIdx.x == 291) {
			value = to_store;
		} else if (threadIdx.x == 292) {
			value = to_store;
		} else if (threadIdx.x == 293) {
			value = to_store;
		} else if (threadIdx.x == 294) {
			value = to_store;
		} else if (threadIdx.x == 295) {
			value = to_store;
		} else if (threadIdx.x == 296) {
			value = to_store;
		} else if (threadIdx.x == 297) {
			value = to_store;
		} else if (threadIdx.x == 298) {
			value = to_store;
		} else if (threadIdx.x == 299) {
			value = to_store;
		} else if (threadIdx.x == 300) {
			value = to_store;
		} else if (threadIdx.x == 301) {
			value = to_store;
		} else if (threadIdx.x == 302) {
			value = to_store;
		} else if (threadIdx.x == 303) {
			value = to_store;
		} else if (threadIdx.x == 304) {
			value = to_store;
		} else if (threadIdx.x == 305) {
			value = to_store;
		} else if (threadIdx.x == 306) {
			value = to_store;
		} else if (threadIdx.x == 307) {
			value = to_store;
		} else if (threadIdx.x == 308) {
			value = to_store;
		} else if (threadIdx.x == 309) {
			value = to_store;
		} else if (threadIdx.x == 310) {
			value = to_store;
		} else if (threadIdx.x == 311) {
			value = to_store;
		} else if (threadIdx.x == 312) {
			value = to_store;
		} else if (threadIdx.x == 313) {
			value = to_store;
		} else if (threadIdx.x == 314) {
			value = to_store;
		} else if (threadIdx.x == 315) {
			value = to_store;
		} else if (threadIdx.x == 316) {
			value = to_store;
		} else if (threadIdx.x == 317) {
			value = to_store;
		} else if (threadIdx.x == 318) {
			value = to_store;
		} else if (threadIdx.x == 319) {
			value = to_store;
		} else if (threadIdx.x == 320) {
			value = to_store;
		} else if (threadIdx.x == 321) {
			value = to_store;
		} else if (threadIdx.x == 322) {
			value = to_store;
		} else if (threadIdx.x == 323) {
			value = to_store;
		} else if (threadIdx.x == 324) {
			value = to_store;
		} else if (threadIdx.x == 325) {
			value = to_store;
		} else if (threadIdx.x == 326) {
			value = to_store;
		} else if (threadIdx.x == 327) {
			value = to_store;
		} else if (threadIdx.x == 328) {
			value = to_store;
		} else if (threadIdx.x == 329) {
			value = to_store;
		} else if (threadIdx.x == 330) {
			value = to_store;
		} else if (threadIdx.x == 331) {
			value = to_store;
		} else if (threadIdx.x == 332) {
			value = to_store;
		} else if (threadIdx.x == 333) {
			value = to_store;
		} else if (threadIdx.x == 334) {
			value = to_store;
		} else if (threadIdx.x == 335) {
			value = to_store;
		} else if (threadIdx.x == 336) {
			value = to_store;
		} else if (threadIdx.x == 337) {
			value = to_store;
		} else if (threadIdx.x == 338) {
			value = to_store;
		} else if (threadIdx.x == 339) {
			value = to_store;
		} else if (threadIdx.x == 340) {
			value = to_store;
		} else if (threadIdx.x == 341) {
			value = to_store;
		} else if (threadIdx.x == 342) {
			value = to_store;
		} else if (threadIdx.x == 343) {
			value = to_store;
		} else if (threadIdx.x == 344) {
			value = to_store;
		} else if (threadIdx.x == 345) {
			value = to_store;
		} else if (threadIdx.x == 346) {
			value = to_store;
		} else if (threadIdx.x == 347) {
			value = to_store;
		} else if (threadIdx.x == 348) {
			value = to_store;
		} else if (threadIdx.x == 349) {
			value = to_store;
		} else if (threadIdx.x == 350) {
			value = to_store;
		} else if (threadIdx.x == 351) {
			value = to_store;
		} else if (threadIdx.x == 352) {
			value = to_store;
		} else if (threadIdx.x == 353) {
			value = to_store;
		} else if (threadIdx.x == 354) {
			value = to_store;
		} else if (threadIdx.x == 355) {
			value = to_store;
		} else if (threadIdx.x == 356) {
			value = to_store;
		} else if (threadIdx.x == 357) {
			value = to_store;
		} else if (threadIdx.x == 358) {
			value = to_store;
		} else if (threadIdx.x == 359) {
			value = to_store;
		} else if (threadIdx.x == 360) {
			value = to_store;
		} else if (threadIdx.x == 361) {
			value = to_store;
		} else if (threadIdx.x == 362) {
			value = to_store;
		} else if (threadIdx.x == 363) {
			value = to_store;
		} else if (threadIdx.x == 364) {
			value = to_store;
		} else if (threadIdx.x == 365) {
			value = to_store;
		} else if (threadIdx.x == 366) {
			value = to_store;
		} else if (threadIdx.x == 367) {
			value = to_store;
		} else if (threadIdx.x == 368) {
			value = to_store;
		} else if (threadIdx.x == 369) {
			value = to_store;
		} else if (threadIdx.x == 370) {
			value = to_store;
		} else if (threadIdx.x == 371) {
			value = to_store;
		} else if (threadIdx.x == 372) {
			value = to_store;
		} else if (threadIdx.x == 373) {
			value = to_store;
		} else if (threadIdx.x == 374) {
			value = to_store;
		} else if (threadIdx.x == 375) {
			value = to_store;
		} else if (threadIdx.x == 376) {
			value = to_store;
		} else if (threadIdx.x == 377) {
			value = to_store;
		} else if (threadIdx.x == 378) {
			value = to_store;
		} else if (threadIdx.x == 379) {
			value = to_store;
		} else if (threadIdx.x == 380) {
			value = to_store;
		} else if (threadIdx.x == 381) {
			value = to_store;
		} else if (threadIdx.x == 382) {
			value = to_store;
		} else if (threadIdx.x == 383) {
			value = to_store;
		} else if (threadIdx.x == 384) {
			value = to_store;
		} else if (threadIdx.x == 385) {
			value = to_store;
		} else if (threadIdx.x == 386) {
			value = to_store;
		} else if (threadIdx.x == 387) {
			value = to_store;
		} else if (threadIdx.x == 388) {
			value = to_store;
		} else if (threadIdx.x == 389) {
			value = to_store;
		} else if (threadIdx.x == 390) {
			value = to_store;
		} else if (threadIdx.x == 391) {
			value = to_store;
		} else if (threadIdx.x == 392) {
			value = to_store;
		} else if (threadIdx.x == 393) {
			value = to_store;
		} else if (threadIdx.x == 394) {
			value = to_store;
		} else if (threadIdx.x == 395) {
			value = to_store;
		} else if (threadIdx.x == 396) {
			value = to_store;
		} else if (threadIdx.x == 397) {
			value = to_store;
		} else if (threadIdx.x == 398) {
			value = to_store;
		} else if (threadIdx.x == 399) {
			value = to_store;
		} else if (threadIdx.x == 400) {
			value = to_store;
		} else if (threadIdx.x == 401) {
			value = to_store;
		} else if (threadIdx.x == 402) {
			value = to_store;
		} else if (threadIdx.x == 403) {
			value = to_store;
		} else if (threadIdx.x == 404) {
			value = to_store;
		} else if (threadIdx.x == 405) {
			value = to_store;
		} else if (threadIdx.x == 406) {
			value = to_store;
		} else if (threadIdx.x == 407) {
			value = to_store;
		} else if (threadIdx.x == 408) {
			value = to_store;
		} else if (threadIdx.x == 409) {
			value = to_store;
		} else if (threadIdx.x == 410) {
			value = to_store;
		} else if (threadIdx.x == 411) {
			value = to_store;
		} else if (threadIdx.x == 412) {
			value = to_store;
		} else if (threadIdx.x == 413) {
			value = to_store;
		} else if (threadIdx.x == 414) {
			value = to_store;
		} else if (threadIdx.x == 415) {
			value = to_store;
		} else if (threadIdx.x == 416) {
			value = to_store;
		} else if (threadIdx.x == 417) {
			value = to_store;
		} else if (threadIdx.x == 418) {
			value = to_store;
		} else if (threadIdx.x == 419) {
			value = to_store;
		} else if (threadIdx.x == 420) {
			value = to_store;
		} else if (threadIdx.x == 421) {
			value = to_store;
		} else if (threadIdx.x == 422) {
			value = to_store;
		} else if (threadIdx.x == 423) {
			value = to_store;
		} else if (threadIdx.x == 424) {
			value = to_store;
		} else if (threadIdx.x == 425) {
			value = to_store;
		} else if (threadIdx.x == 426) {
			value = to_store;
		} else if (threadIdx.x == 427) {
			value = to_store;
		} else if (threadIdx.x == 428) {
			value = to_store;
		} else if (threadIdx.x == 429) {
			value = to_store;
		} else if (threadIdx.x == 430) {
			value = to_store;
		} else if (threadIdx.x == 431) {
			value = to_store;
		} else if (threadIdx.x == 432) {
			value = to_store;
		} else if (threadIdx.x == 433) {
			value = to_store;
		} else if (threadIdx.x == 434) {
			value = to_store;
		} else if (threadIdx.x == 435) {
			value = to_store;
		} else if (threadIdx.x == 436) {
			value = to_store;
		} else if (threadIdx.x == 437) {
			value = to_store;
		} else if (threadIdx.x == 438) {
			value = to_store;
		} else if (threadIdx.x == 439) {
			value = to_store;
		} else if (threadIdx.x == 440) {
			value = to_store;
		} else if (threadIdx.x == 441) {
			value = to_store;
		} else if (threadIdx.x == 442) {
			value = to_store;
		} else if (threadIdx.x == 443) {
			value = to_store;
		} else if (threadIdx.x == 444) {
			value = to_store;
		} else if (threadIdx.x == 445) {
			value = to_store;
		} else if (threadIdx.x == 446) {
			value = to_store;
		} else if (threadIdx.x == 447) {
			value = to_store;
		} else if (threadIdx.x == 448) {
			value = to_store;
		} else if (threadIdx.x == 449) {
			value = to_store;
		} else if (threadIdx.x == 450) {
			value = to_store;
		} else if (threadIdx.x == 451) {
			value = to_store;
		} else if (threadIdx.x == 452) {
			value = to_store;
		} else if (threadIdx.x == 453) {
			value = to_store;
		} else if (threadIdx.x == 454) {
			value = to_store;
		} else if (threadIdx.x == 455) {
			value = to_store;
		} else if (threadIdx.x == 456) {
			value = to_store;
		} else if (threadIdx.x == 457) {
			value = to_store;
		} else if (threadIdx.x == 458) {
			value = to_store;
		} else if (threadIdx.x == 459) {
			value = to_store;
		} else if (threadIdx.x == 460) {
			value = to_store;
		} else if (threadIdx.x == 461) {
			value = to_store;
		} else if (threadIdx.x == 462) {
			value = to_store;
		} else if (threadIdx.x == 463) {
			value = to_store;
		} else if (threadIdx.x == 464) {
			value = to_store;
		} else if (threadIdx.x == 465) {
			value = to_store;
		} else if (threadIdx.x == 466) {
			value = to_store;
		} else if (threadIdx.x == 467) {
			value = to_store;
		} else if (threadIdx.x == 468) {
			value = to_store;
		} else if (threadIdx.x == 469) {
			value = to_store;
		} else if (threadIdx.x == 470) {
			value = to_store;
		} else if (threadIdx.x == 471) {
			value = to_store;
		} else if (threadIdx.x == 472) {
			value = to_store;
		} else if (threadIdx.x == 473) {
			value = to_store;
		} else if (threadIdx.x == 474) {
			value = to_store;
		} else if (threadIdx.x == 475) {
			value = to_store;
		} else if (threadIdx.x == 476) {
			value = to_store;
		} else if (threadIdx.x == 477) {
			value = to_store;
		} else if (threadIdx.x == 478) {
			value = to_store;
		} else if (threadIdx.x == 479) {
			value = to_store;
		} else if (threadIdx.x == 480) {
			value = to_store;
		} else if (threadIdx.x == 481) {
			value = to_store;
		} else if (threadIdx.x == 482) {
			value = to_store;
		} else if (threadIdx.x == 483) {
			value = to_store;
		} else if (threadIdx.x == 484) {
			value = to_store;
		} else if (threadIdx.x == 485) {
			value = to_store;
		} else if (threadIdx.x == 486) {
			value = to_store;
		} else if (threadIdx.x == 487) {
			value = to_store;
		} else if (threadIdx.x == 488) {
			value = to_store;
		} else if (threadIdx.x == 489) {
			value = to_store;
		} else if (threadIdx.x == 490) {
			value = to_store;
		} else if (threadIdx.x == 491) {
			value = to_store;
		} else if (threadIdx.x == 492) {
			value = to_store;
		} else if (threadIdx.x == 493) {
			value = to_store;
		} else if (threadIdx.x == 494) {
			value = to_store;
		} else if (threadIdx.x == 495) {
			value = to_store;
		} else if (threadIdx.x == 496) {
			value = to_store;
		} else if (threadIdx.x == 497) {
			value = to_store;
		} else if (threadIdx.x == 498) {
			value = to_store;
		} else if (threadIdx.x == 499) {
			value = to_store;
		} else if (threadIdx.x == 500) {
			value = to_store;
		} else if (threadIdx.x == 501) {
			value = to_store;
		} else if (threadIdx.x == 502) {
			value = to_store;
		} else if (threadIdx.x == 503) {
			value = to_store;
		} else if (threadIdx.x == 504) {
			value = to_store;
		} else if (threadIdx.x == 505) {
			value = to_store;
		} else if (threadIdx.x == 506) {
			value = to_store;
		} else if (threadIdx.x == 507) {
			value = to_store;
		} else if (threadIdx.x == 508) {
			value = to_store;
		} else if (threadIdx.x == 509) {
			value = to_store;
		} else if (threadIdx.x == 510) {
			value = to_store;
		} else if (threadIdx.x == 511) {
			value = to_store;
		} else if (threadIdx.x == 512) {
			value = to_store;
		} else if (threadIdx.x == 513) {
			value = to_store;
		} else if (threadIdx.x == 514) {
			value = to_store;
		} else if (threadIdx.x == 515) {
			value = to_store;
		} else if (threadIdx.x == 516) {
			value = to_store;
		} else if (threadIdx.x == 517) {
			value = to_store;
		} else if (threadIdx.x == 518) {
			value = to_store;
		} else if (threadIdx.x == 519) {
			value = to_store;
		} else if (threadIdx.x == 520) {
			value = to_store;
		} else if (threadIdx.x == 521) {
			value = to_store;
		} else if (threadIdx.x == 522) {
			value = to_store;
		} else if (threadIdx.x == 523) {
			value = to_store;
		} else if (threadIdx.x == 524) {
			value = to_store;
		} else if (threadIdx.x == 525) {
			value = to_store;
		} else if (threadIdx.x == 526) {
			value = to_store;
		} else if (threadIdx.x == 527) {
			value = to_store;
		} else if (threadIdx.x == 528) {
			value = to_store;
		} else if (threadIdx.x == 529) {
			value = to_store;
		} else if (threadIdx.x == 530) {
			value = to_store;
		} else if (threadIdx.x == 531) {
			value = to_store;
		} else if (threadIdx.x == 532) {
			value = to_store;
		} else if (threadIdx.x == 533) {
			value = to_store;
		} else if (threadIdx.x == 534) {
			value = to_store;
		} else if (threadIdx.x == 535) {
			value = to_store;
		} else if (threadIdx.x == 536) {
			value = to_store;
		} else if (threadIdx.x == 537) {
			value = to_store;
		} else if (threadIdx.x == 538) {
			value = to_store;
		} else if (threadIdx.x == 539) {
			value = to_store;
		} else if (threadIdx.x == 540) {
			value = to_store;
		} else if (threadIdx.x == 541) {
			value = to_store;
		} else if (threadIdx.x == 542) {
			value = to_store;
		} else if (threadIdx.x == 543) {
			value = to_store;
		} else if (threadIdx.x == 544) {
			value = to_store;
		} else if (threadIdx.x == 545) {
			value = to_store;
		} else if (threadIdx.x == 546) {
			value = to_store;
		} else if (threadIdx.x == 547) {
			value = to_store;
		} else if (threadIdx.x == 548) {
			value = to_store;
		} else if (threadIdx.x == 549) {
			value = to_store;
		} else if (threadIdx.x == 550) {
			value = to_store;
		} else if (threadIdx.x == 551) {
			value = to_store;
		} else if (threadIdx.x == 552) {
			value = to_store;
		} else if (threadIdx.x == 553) {
			value = to_store;
		} else if (threadIdx.x == 554) {
			value = to_store;
		} else if (threadIdx.x == 555) {
			value = to_store;
		} else if (threadIdx.x == 556) {
			value = to_store;
		} else if (threadIdx.x == 557) {
			value = to_store;
		} else if (threadIdx.x == 558) {
			value = to_store;
		} else if (threadIdx.x == 559) {
			value = to_store;
		} else if (threadIdx.x == 560) {
			value = to_store;
		} else if (threadIdx.x == 561) {
			value = to_store;
		} else if (threadIdx.x == 562) {
			value = to_store;
		} else if (threadIdx.x == 563) {
			value = to_store;
		} else if (threadIdx.x == 564) {
			value = to_store;
		} else if (threadIdx.x == 565) {
			value = to_store;
		} else if (threadIdx.x == 566) {
			value = to_store;
		} else if (threadIdx.x == 567) {
			value = to_store;
		} else if (threadIdx.x == 568) {
			value = to_store;
		} else if (threadIdx.x == 569) {
			value = to_store;
		} else if (threadIdx.x == 570) {
			value = to_store;
		} else if (threadIdx.x == 571) {
			value = to_store;
		} else if (threadIdx.x == 572) {
			value = to_store;
		} else if (threadIdx.x == 573) {
			value = to_store;
		} else if (threadIdx.x == 574) {
			value = to_store;
		} else if (threadIdx.x == 575) {
			value = to_store;
		} else if (threadIdx.x == 576) {
			value = to_store;
		} else if (threadIdx.x == 577) {
			value = to_store;
		} else if (threadIdx.x == 578) {
			value = to_store;
		} else if (threadIdx.x == 579) {
			value = to_store;
		} else if (threadIdx.x == 580) {
			value = to_store;
		} else if (threadIdx.x == 581) {
			value = to_store;
		} else if (threadIdx.x == 582) {
			value = to_store;
		} else if (threadIdx.x == 583) {
			value = to_store;
		} else if (threadIdx.x == 584) {
			value = to_store;
		} else if (threadIdx.x == 585) {
			value = to_store;
		} else if (threadIdx.x == 586) {
			value = to_store;
		} else if (threadIdx.x == 587) {
			value = to_store;
		} else if (threadIdx.x == 588) {
			value = to_store;
		} else if (threadIdx.x == 589) {
			value = to_store;
		} else if (threadIdx.x == 590) {
			value = to_store;
		} else if (threadIdx.x == 591) {
			value = to_store;
		} else if (threadIdx.x == 592) {
			value = to_store;
		} else if (threadIdx.x == 593) {
			value = to_store;
		} else if (threadIdx.x == 594) {
			value = to_store;
		} else if (threadIdx.x == 595) {
			value = to_store;
		} else if (threadIdx.x == 596) {
			value = to_store;
		} else if (threadIdx.x == 597) {
			value = to_store;
		} else if (threadIdx.x == 598) {
			value = to_store;
		} else if (threadIdx.x == 599) {
			value = to_store;
		} else if (threadIdx.x == 600) {
			value = to_store;
		} else if (threadIdx.x == 601) {
			value = to_store;
		} else if (threadIdx.x == 602) {
			value = to_store;
		} else if (threadIdx.x == 603) {
			value = to_store;
		} else if (threadIdx.x == 604) {
			value = to_store;
		} else if (threadIdx.x == 605) {
			value = to_store;
		} else if (threadIdx.x == 606) {
			value = to_store;
		} else if (threadIdx.x == 607) {
			value = to_store;
		} else if (threadIdx.x == 608) {
			value = to_store;
		} else if (threadIdx.x == 609) {
			value = to_store;
		} else if (threadIdx.x == 610) {
			value = to_store;
		} else if (threadIdx.x == 611) {
			value = to_store;
		} else if (threadIdx.x == 612) {
			value = to_store;
		} else if (threadIdx.x == 613) {
			value = to_store;
		} else if (threadIdx.x == 614) {
			value = to_store;
		} else if (threadIdx.x == 615) {
			value = to_store;
		} else if (threadIdx.x == 616) {
			value = to_store;
		} else if (threadIdx.x == 617) {
			value = to_store;
		} else if (threadIdx.x == 618) {
			value = to_store;
		} else if (threadIdx.x == 619) {
			value = to_store;
		} else if (threadIdx.x == 620) {
			value = to_store;
		} else if (threadIdx.x == 621) {
			value = to_store;
		} else if (threadIdx.x == 622) {
			value = to_store;
		} else if (threadIdx.x == 623) {
			value = to_store;
		} else if (threadIdx.x == 624) {
			value = to_store;
		} else if (threadIdx.x == 625) {
			value = to_store;
		} else if (threadIdx.x == 626) {
			value = to_store;
		} else if (threadIdx.x == 627) {
			value = to_store;
		} else if (threadIdx.x == 628) {
			value = to_store;
		} else if (threadIdx.x == 629) {
			value = to_store;
		} else if (threadIdx.x == 630) {
			value = to_store;
		} else if (threadIdx.x == 631) {
			value = to_store;
		} else if (threadIdx.x == 632) {
			value = to_store;
		} else if (threadIdx.x == 633) {
			value = to_store;
		} else if (threadIdx.x == 634) {
			value = to_store;
		} else if (threadIdx.x == 635) {
			value = to_store;
		} else if (threadIdx.x == 636) {
			value = to_store;
		} else if (threadIdx.x == 637) {
			value = to_store;
		} else if (threadIdx.x == 638) {
			value = to_store;
		} else if (threadIdx.x == 639) {
			value = to_store;
		} else if (threadIdx.x == 640) {
			value = to_store;
		} else if (threadIdx.x == 641) {
			value = to_store;
		} else if (threadIdx.x == 642) {
			value = to_store;
		} else if (threadIdx.x == 643) {
			value = to_store;
		} else if (threadIdx.x == 644) {
			value = to_store;
		} else if (threadIdx.x == 645) {
			value = to_store;
		} else if (threadIdx.x == 646) {
			value = to_store;
		} else if (threadIdx.x == 647) {
			value = to_store;
		} else if (threadIdx.x == 648) {
			value = to_store;
		} else if (threadIdx.x == 649) {
			value = to_store;
		} else if (threadIdx.x == 650) {
			value = to_store;
		} else if (threadIdx.x == 651) {
			value = to_store;
		} else if (threadIdx.x == 652) {
			value = to_store;
		} else if (threadIdx.x == 653) {
			value = to_store;
		} else if (threadIdx.x == 654) {
			value = to_store;
		} else if (threadIdx.x == 655) {
			value = to_store;
		} else if (threadIdx.x == 656) {
			value = to_store;
		} else if (threadIdx.x == 657) {
			value = to_store;
		} else if (threadIdx.x == 658) {
			value = to_store;
		} else if (threadIdx.x == 659) {
			value = to_store;
		} else if (threadIdx.x == 660) {
			value = to_store;
		} else if (threadIdx.x == 661) {
			value = to_store;
		} else if (threadIdx.x == 662) {
			value = to_store;
		} else if (threadIdx.x == 663) {
			value = to_store;
		} else if (threadIdx.x == 664) {
			value = to_store;
		} else if (threadIdx.x == 665) {
			value = to_store;
		} else if (threadIdx.x == 666) {
			value = to_store;
		} else if (threadIdx.x == 667) {
			value = to_store;
		} else if (threadIdx.x == 668) {
			value = to_store;
		} else if (threadIdx.x == 669) {
			value = to_store;
		} else if (threadIdx.x == 670) {
			value = to_store;
		} else if (threadIdx.x == 671) {
			value = to_store;
		} else if (threadIdx.x == 672) {
			value = to_store;
		} else if (threadIdx.x == 673) {
			value = to_store;
		} else if (threadIdx.x == 674) {
			value = to_store;
		} else if (threadIdx.x == 675) {
			value = to_store;
		} else if (threadIdx.x == 676) {
			value = to_store;
		} else if (threadIdx.x == 677) {
			value = to_store;
		} else if (threadIdx.x == 678) {
			value = to_store;
		} else if (threadIdx.x == 679) {
			value = to_store;
		} else if (threadIdx.x == 680) {
			value = to_store;
		} else if (threadIdx.x == 681) {
			value = to_store;
		} else if (threadIdx.x == 682) {
			value = to_store;
		} else if (threadIdx.x == 683) {
			value = to_store;
		} else if (threadIdx.x == 684) {
			value = to_store;
		} else if (threadIdx.x == 685) {
			value = to_store;
		} else if (threadIdx.x == 686) {
			value = to_store;
		} else if (threadIdx.x == 687) {
			value = to_store;
		} else if (threadIdx.x == 688) {
			value = to_store;
		} else if (threadIdx.x == 689) {
			value = to_store;
		} else if (threadIdx.x == 690) {
			value = to_store;
		} else if (threadIdx.x == 691) {
			value = to_store;
		} else if (threadIdx.x == 692) {
			value = to_store;
		} else if (threadIdx.x == 693) {
			value = to_store;
		} else if (threadIdx.x == 694) {
			value = to_store;
		} else if (threadIdx.x == 695) {
			value = to_store;
		} else if (threadIdx.x == 696) {
			value = to_store;
		} else if (threadIdx.x == 697) {
			value = to_store;
		} else if (threadIdx.x == 698) {
			value = to_store;
		} else if (threadIdx.x == 699) {
			value = to_store;
		} else if (threadIdx.x == 700) {
			value = to_store;
		} else if (threadIdx.x == 701) {
			value = to_store;
		} else if (threadIdx.x == 702) {
			value = to_store;
		} else if (threadIdx.x == 703) {
			value = to_store;
		} else if (threadIdx.x == 704) {
			value = to_store;
		} else if (threadIdx.x == 705) {
			value = to_store;
		} else if (threadIdx.x == 706) {
			value = to_store;
		} else if (threadIdx.x == 707) {
			value = to_store;
		} else if (threadIdx.x == 708) {
			value = to_store;
		} else if (threadIdx.x == 709) {
			value = to_store;
		} else if (threadIdx.x == 710) {
			value = to_store;
		} else if (threadIdx.x == 711) {
			value = to_store;
		} else if (threadIdx.x == 712) {
			value = to_store;
		} else if (threadIdx.x == 713) {
			value = to_store;
		} else if (threadIdx.x == 714) {
			value = to_store;
		} else if (threadIdx.x == 715) {
			value = to_store;
		} else if (threadIdx.x == 716) {
			value = to_store;
		} else if (threadIdx.x == 717) {
			value = to_store;
		} else if (threadIdx.x == 718) {
			value = to_store;
		} else if (threadIdx.x == 719) {
			value = to_store;
		} else if (threadIdx.x == 720) {
			value = to_store;
		} else if (threadIdx.x == 721) {
			value = to_store;
		} else if (threadIdx.x == 722) {
			value = to_store;
		} else if (threadIdx.x == 723) {
			value = to_store;
		} else if (threadIdx.x == 724) {
			value = to_store;
		} else if (threadIdx.x == 725) {
			value = to_store;
		} else if (threadIdx.x == 726) {
			value = to_store;
		} else if (threadIdx.x == 727) {
			value = to_store;
		} else if (threadIdx.x == 728) {
			value = to_store;
		} else if (threadIdx.x == 729) {
			value = to_store;
		} else if (threadIdx.x == 730) {
			value = to_store;
		} else if (threadIdx.x == 731) {
			value = to_store;
		} else if (threadIdx.x == 732) {
			value = to_store;
		} else if (threadIdx.x == 733) {
			value = to_store;
		} else if (threadIdx.x == 734) {
			value = to_store;
		} else if (threadIdx.x == 735) {
			value = to_store;
		} else if (threadIdx.x == 736) {
			value = to_store;
		} else if (threadIdx.x == 737) {
			value = to_store;
		} else if (threadIdx.x == 738) {
			value = to_store;
		} else if (threadIdx.x == 739) {
			value = to_store;
		} else if (threadIdx.x == 740) {
			value = to_store;
		} else if (threadIdx.x == 741) {
			value = to_store;
		} else if (threadIdx.x == 742) {
			value = to_store;
		} else if (threadIdx.x == 743) {
			value = to_store;
		} else if (threadIdx.x == 744) {
			value = to_store;
		} else if (threadIdx.x == 745) {
			value = to_store;
		} else if (threadIdx.x == 746) {
			value = to_store;
		} else if (threadIdx.x == 747) {
			value = to_store;
		} else if (threadIdx.x == 748) {
			value = to_store;
		} else if (threadIdx.x == 749) {
			value = to_store;
		} else if (threadIdx.x == 750) {
			value = to_store;
		} else if (threadIdx.x == 751) {
			value = to_store;
		} else if (threadIdx.x == 752) {
			value = to_store;
		} else if (threadIdx.x == 753) {
			value = to_store;
		} else if (threadIdx.x == 754) {
			value = to_store;
		} else if (threadIdx.x == 755) {
			value = to_store;
		} else if (threadIdx.x == 756) {
			value = to_store;
		} else if (threadIdx.x == 757) {
			value = to_store;
		} else if (threadIdx.x == 758) {
			value = to_store;
		} else if (threadIdx.x == 759) {
			value = to_store;
		} else if (threadIdx.x == 760) {
			value = to_store;
		} else if (threadIdx.x == 761) {
			value = to_store;
		} else if (threadIdx.x == 762) {
			value = to_store;
		} else if (threadIdx.x == 763) {
			value = to_store;
		} else if (threadIdx.x == 764) {
			value = to_store;
		} else if (threadIdx.x == 765) {
			value = to_store;
		} else if (threadIdx.x == 766) {
			value = to_store;
		} else if (threadIdx.x == 767) {
			value = to_store;
		} else if (threadIdx.x == 768) {
			value = to_store;
		} else if (threadIdx.x == 769) {
			value = to_store;
		} else if (threadIdx.x == 770) {
			value = to_store;
		} else if (threadIdx.x == 771) {
			value = to_store;
		} else if (threadIdx.x == 772) {
			value = to_store;
		} else if (threadIdx.x == 773) {
			value = to_store;
		} else if (threadIdx.x == 774) {
			value = to_store;
		} else if (threadIdx.x == 775) {
			value = to_store;
		} else if (threadIdx.x == 776) {
			value = to_store;
		} else if (threadIdx.x == 777) {
			value = to_store;
		} else if (threadIdx.x == 778) {
			value = to_store;
		} else if (threadIdx.x == 779) {
			value = to_store;
		} else if (threadIdx.x == 780) {
			value = to_store;
		} else if (threadIdx.x == 781) {
			value = to_store;
		} else if (threadIdx.x == 782) {
			value = to_store;
		} else if (threadIdx.x == 783) {
			value = to_store;
		} else if (threadIdx.x == 784) {
			value = to_store;
		} else if (threadIdx.x == 785) {
			value = to_store;
		} else if (threadIdx.x == 786) {
			value = to_store;
		} else if (threadIdx.x == 787) {
			value = to_store;
		} else if (threadIdx.x == 788) {
			value = to_store;
		} else if (threadIdx.x == 789) {
			value = to_store;
		} else if (threadIdx.x == 790) {
			value = to_store;
		} else if (threadIdx.x == 791) {
			value = to_store;
		} else if (threadIdx.x == 792) {
			value = to_store;
		} else if (threadIdx.x == 793) {
			value = to_store;
		} else if (threadIdx.x == 794) {
			value = to_store;
		} else if (threadIdx.x == 795) {
			value = to_store;
		} else if (threadIdx.x == 796) {
			value = to_store;
		} else if (threadIdx.x == 797) {
			value = to_store;
		} else if (threadIdx.x == 798) {
			value = to_store;
		} else if (threadIdx.x == 799) {
			value = to_store;
		} else if (threadIdx.x == 800) {
			value = to_store;
		} else if (threadIdx.x == 801) {
			value = to_store;
		} else if (threadIdx.x == 802) {
			value = to_store;
		} else if (threadIdx.x == 803) {
			value = to_store;
		} else if (threadIdx.x == 804) {
			value = to_store;
		} else if (threadIdx.x == 805) {
			value = to_store;
		} else if (threadIdx.x == 806) {
			value = to_store;
		} else if (threadIdx.x == 807) {
			value = to_store;
		} else if (threadIdx.x == 808) {
			value = to_store;
		} else if (threadIdx.x == 809) {
			value = to_store;
		} else if (threadIdx.x == 810) {
			value = to_store;
		} else if (threadIdx.x == 811) {
			value = to_store;
		} else if (threadIdx.x == 812) {
			value = to_store;
		} else if (threadIdx.x == 813) {
			value = to_store;
		} else if (threadIdx.x == 814) {
			value = to_store;
		} else if (threadIdx.x == 815) {
			value = to_store;
		} else if (threadIdx.x == 816) {
			value = to_store;
		} else if (threadIdx.x == 817) {
			value = to_store;
		} else if (threadIdx.x == 818) {
			value = to_store;
		} else if (threadIdx.x == 819) {
			value = to_store;
		} else if (threadIdx.x == 820) {
			value = to_store;
		} else if (threadIdx.x == 821) {
			value = to_store;
		} else if (threadIdx.x == 822) {
			value = to_store;
		} else if (threadIdx.x == 823) {
			value = to_store;
		} else if (threadIdx.x == 824) {
			value = to_store;
		} else if (threadIdx.x == 825) {
			value = to_store;
		} else if (threadIdx.x == 826) {
			value = to_store;
		} else if (threadIdx.x == 827) {
			value = to_store;
		} else if (threadIdx.x == 828) {
			value = to_store;
		} else if (threadIdx.x == 829) {
			value = to_store;
		} else if (threadIdx.x == 830) {
			value = to_store;
		} else if (threadIdx.x == 831) {
			value = to_store;
		} else if (threadIdx.x == 832) {
			value = to_store;
		} else if (threadIdx.x == 833) {
			value = to_store;
		} else if (threadIdx.x == 834) {
			value = to_store;
		} else if (threadIdx.x == 835) {
			value = to_store;
		} else if (threadIdx.x == 836) {
			value = to_store;
		} else if (threadIdx.x == 837) {
			value = to_store;
		} else if (threadIdx.x == 838) {
			value = to_store;
		} else if (threadIdx.x == 839) {
			value = to_store;
		} else if (threadIdx.x == 840) {
			value = to_store;
		} else if (threadIdx.x == 841) {
			value = to_store;
		} else if (threadIdx.x == 842) {
			value = to_store;
		} else if (threadIdx.x == 843) {
			value = to_store;
		} else if (threadIdx.x == 844) {
			value = to_store;
		} else if (threadIdx.x == 845) {
			value = to_store;
		} else if (threadIdx.x == 846) {
			value = to_store;
		} else if (threadIdx.x == 847) {
			value = to_store;
		} else if (threadIdx.x == 848) {
			value = to_store;
		} else if (threadIdx.x == 849) {
			value = to_store;
		} else if (threadIdx.x == 850) {
			value = to_store;
		} else if (threadIdx.x == 851) {
			value = to_store;
		} else if (threadIdx.x == 852) {
			value = to_store;
		} else if (threadIdx.x == 853) {
			value = to_store;
		} else if (threadIdx.x == 854) {
			value = to_store;
		} else if (threadIdx.x == 855) {
			value = to_store;
		} else if (threadIdx.x == 856) {
			value = to_store;
		} else if (threadIdx.x == 857) {
			value = to_store;
		} else if (threadIdx.x == 858) {
			value = to_store;
		} else if (threadIdx.x == 859) {
			value = to_store;
		} else if (threadIdx.x == 860) {
			value = to_store;
		} else if (threadIdx.x == 861) {
			value = to_store;
		} else if (threadIdx.x == 862) {
			value = to_store;
		} else if (threadIdx.x == 863) {
			value = to_store;
		} else if (threadIdx.x == 864) {
			value = to_store;
		} else if (threadIdx.x == 865) {
			value = to_store;
		} else if (threadIdx.x == 866) {
			value = to_store;
		} else if (threadIdx.x == 867) {
			value = to_store;
		} else if (threadIdx.x == 868) {
			value = to_store;
		} else if (threadIdx.x == 869) {
			value = to_store;
		} else if (threadIdx.x == 870) {
			value = to_store;
		} else if (threadIdx.x == 871) {
			value = to_store;
		} else if (threadIdx.x == 872) {
			value = to_store;
		} else if (threadIdx.x == 873) {
			value = to_store;
		} else if (threadIdx.x == 874) {
			value = to_store;
		} else if (threadIdx.x == 875) {
			value = to_store;
		} else if (threadIdx.x == 876) {
			value = to_store;
		} else if (threadIdx.x == 877) {
			value = to_store;
		} else if (threadIdx.x == 878) {
			value = to_store;
		} else if (threadIdx.x == 879) {
			value = to_store;
		} else if (threadIdx.x == 880) {
			value = to_store;
		} else if (threadIdx.x == 881) {
			value = to_store;
		} else if (threadIdx.x == 882) {
			value = to_store;
		} else if (threadIdx.x == 883) {
			value = to_store;
		} else if (threadIdx.x == 884) {
			value = to_store;
		} else if (threadIdx.x == 885) {
			value = to_store;
		} else if (threadIdx.x == 886) {
			value = to_store;
		} else if (threadIdx.x == 887) {
			value = to_store;
		} else if (threadIdx.x == 888) {
			value = to_store;
		} else if (threadIdx.x == 889) {
			value = to_store;
		} else if (threadIdx.x == 890) {
			value = to_store;
		} else if (threadIdx.x == 891) {
			value = to_store;
		} else if (threadIdx.x == 892) {
			value = to_store;
		} else if (threadIdx.x == 893) {
			value = to_store;
		} else if (threadIdx.x == 894) {
			value = to_store;
		} else if (threadIdx.x == 895) {
			value = to_store;
		} else if (threadIdx.x == 896) {
			value = to_store;
		} else if (threadIdx.x == 897) {
			value = to_store;
		} else if (threadIdx.x == 898) {
			value = to_store;
		} else if (threadIdx.x == 899) {
			value = to_store;
		} else if (threadIdx.x == 900) {
			value = to_store;
		} else if (threadIdx.x == 901) {
			value = to_store;
		} else if (threadIdx.x == 902) {
			value = to_store;
		} else if (threadIdx.x == 903) {
			value = to_store;
		} else if (threadIdx.x == 904) {
			value = to_store;
		} else if (threadIdx.x == 905) {
			value = to_store;
		} else if (threadIdx.x == 906) {
			value = to_store;
		} else if (threadIdx.x == 907) {
			value = to_store;
		} else if (threadIdx.x == 908) {
			value = to_store;
		} else if (threadIdx.x == 909) {
			value = to_store;
		} else if (threadIdx.x == 910) {
			value = to_store;
		} else if (threadIdx.x == 911) {
			value = to_store;
		} else if (threadIdx.x == 912) {
			value = to_store;
		} else if (threadIdx.x == 913) {
			value = to_store;
		} else if (threadIdx.x == 914) {
			value = to_store;
		} else if (threadIdx.x == 915) {
			value = to_store;
		} else if (threadIdx.x == 916) {
			value = to_store;
		} else if (threadIdx.x == 917) {
			value = to_store;
		} else if (threadIdx.x == 918) {
			value = to_store;
		} else if (threadIdx.x == 919) {
			value = to_store;
		} else if (threadIdx.x == 920) {
			value = to_store;
		} else if (threadIdx.x == 921) {
			value = to_store;
		} else if (threadIdx.x == 922) {
			value = to_store;
		} else if (threadIdx.x == 923) {
			value = to_store;
		} else if (threadIdx.x == 924) {
			value = to_store;
		} else if (threadIdx.x == 925) {
			value = to_store;
		} else if (threadIdx.x == 926) {
			value = to_store;
		} else if (threadIdx.x == 927) {
			value = to_store;
		} else if (threadIdx.x == 928) {
			value = to_store;
		} else if (threadIdx.x == 929) {
			value = to_store;
		} else if (threadIdx.x == 930) {
			value = to_store;
		} else if (threadIdx.x == 931) {
			value = to_store;
		} else if (threadIdx.x == 932) {
			value = to_store;
		} else if (threadIdx.x == 933) {
			value = to_store;
		} else if (threadIdx.x == 934) {
			value = to_store;
		} else if (threadIdx.x == 935) {
			value = to_store;
		} else if (threadIdx.x == 936) {
			value = to_store;
		} else if (threadIdx.x == 937) {
			value = to_store;
		} else if (threadIdx.x == 938) {
			value = to_store;
		} else if (threadIdx.x == 939) {
			value = to_store;
		} else if (threadIdx.x == 940) {
			value = to_store;
		} else if (threadIdx.x == 941) {
			value = to_store;
		} else if (threadIdx.x == 942) {
			value = to_store;
		} else if (threadIdx.x == 943) {
			value = to_store;
		} else if (threadIdx.x == 944) {
			value = to_store;
		} else if (threadIdx.x == 945) {
			value = to_store;
		} else if (threadIdx.x == 946) {
			value = to_store;
		} else if (threadIdx.x == 947) {
			value = to_store;
		} else if (threadIdx.x == 948) {
			value = to_store;
		} else if (threadIdx.x == 949) {
			value = to_store;
		} else if (threadIdx.x == 950) {
			value = to_store;
		} else if (threadIdx.x == 951) {
			value = to_store;
		} else if (threadIdx.x == 952) {
			value = to_store;
		} else if (threadIdx.x == 953) {
			value = to_store;
		} else if (threadIdx.x == 954) {
			value = to_store;
		} else if (threadIdx.x == 955) {
			value = to_store;
		} else if (threadIdx.x == 956) {
			value = to_store;
		} else if (threadIdx.x == 957) {
			value = to_store;
		} else if (threadIdx.x == 958) {
			value = to_store;
		} else if (threadIdx.x == 959) {
			value = to_store;
		} else if (threadIdx.x == 960) {
			value = to_store;
		} else if (threadIdx.x == 961) {
			value = to_store;
		} else if (threadIdx.x == 962) {
			value = to_store;
		} else if (threadIdx.x == 963) {
			value = to_store;
		} else if (threadIdx.x == 964) {
			value = to_store;
		} else if (threadIdx.x == 965) {
			value = to_store;
		} else if (threadIdx.x == 966) {
			value = to_store;
		} else if (threadIdx.x == 967) {
			value = to_store;
		} else if (threadIdx.x == 968) {
			value = to_store;
		} else if (threadIdx.x == 969) {
			value = to_store;
		} else if (threadIdx.x == 970) {
			value = to_store;
		} else if (threadIdx.x == 971) {
			value = to_store;
		} else if (threadIdx.x == 972) {
			value = to_store;
		} else if (threadIdx.x == 973) {
			value = to_store;
		} else if (threadIdx.x == 974) {
			value = to_store;
		} else if (threadIdx.x == 975) {
			value = to_store;
		} else if (threadIdx.x == 976) {
			value = to_store;
		} else if (threadIdx.x == 977) {
			value = to_store;
		} else if (threadIdx.x == 978) {
			value = to_store;
		} else if (threadIdx.x == 979) {
			value = to_store;
		} else if (threadIdx.x == 980) {
			value = to_store;
		} else if (threadIdx.x == 981) {
			value = to_store;
		} else if (threadIdx.x == 982) {
			value = to_store;
		} else if (threadIdx.x == 983) {
			value = to_store;
		} else if (threadIdx.x == 984) {
			value = to_store;
		} else if (threadIdx.x == 985) {
			value = to_store;
		} else if (threadIdx.x == 986) {
			value = to_store;
		} else if (threadIdx.x == 987) {
			value = to_store;
		} else if (threadIdx.x == 988) {
			value = to_store;
		} else if (threadIdx.x == 989) {
			value = to_store;
		} else if (threadIdx.x == 990) {
			value = to_store;
		} else if (threadIdx.x == 991) {
			value = to_store;
		} else if (threadIdx.x == 992) {
			value = to_store;
		} else if (threadIdx.x == 993) {
			value = to_store;
		} else if (threadIdx.x == 994) {
			value = to_store;
		} else if (threadIdx.x == 995) {
			value = to_store;
		} else if (threadIdx.x == 996) {
			value = to_store;
		} else if (threadIdx.x == 997) {
			value = to_store;
		} else if (threadIdx.x == 998) {
			value = to_store;
		} else if (threadIdx.x == 999) {
			value = to_store;
		} else if (threadIdx.x == 1000) {
			value = to_store;
		} else if (threadIdx.x == 1001) {
			value = to_store;
		} else if (threadIdx.x == 1002) {
			value = to_store;
		} else if (threadIdx.x == 1003) {
			value = to_store;
		} else if (threadIdx.x == 1004) {
			value = to_store;
		} else if (threadIdx.x == 1005) {
			value = to_store;
		} else if (threadIdx.x == 1006) {
			value = to_store;
		} else if (threadIdx.x == 1007) {
			value = to_store;
		} else if (threadIdx.x == 1008) {
			value = to_store;
		} else if (threadIdx.x == 1009) {
			value = to_store;
		} else if (threadIdx.x == 1010) {
			value = to_store;
		} else if (threadIdx.x == 1011) {
			value = to_store;
		} else if (threadIdx.x == 1012) {
			value = to_store;
		} else if (threadIdx.x == 1013) {
			value = to_store;
		} else if (threadIdx.x == 1014) {
			value = to_store;
		} else if (threadIdx.x == 1015) {
			value = to_store;
		} else if (threadIdx.x == 1016) {
			value = to_store;
		} else if (threadIdx.x == 1017) {
			value = to_store;
		} else if (threadIdx.x == 1018) {
			value = to_store;
		} else if (threadIdx.x == 1019) {
			value = to_store;
		} else if (threadIdx.x == 1020) {
			value = to_store;
		} else if (threadIdx.x == 1021) {
			value = to_store;
		} else if (threadIdx.x == 1022) {
			value = to_store;
		} else if (threadIdx.x == 1023) {
			value = to_store;
		}
	}
	dst_1[i] = value;

	dst_2[i] = value;

	dst_3[i] = value;

}

#endif /* BRANCH_KERNEL_H_ */
