#ifndef BRANCH_KERNEL_H_
#define BRANCH_KERNEL_H_

#include <cstdint>
#include "input_device.h"


template<typename int_t>
__global__ void int_branch_kernel(int_t* dst_1, int_t* dst_2, int_t* dst_3, uint32_t op) {
	const int_t i = (blockDim.x * blockIdx.x + threadIdx.x);
	int_t value = 0;
	for(int opi = 0; opi < op; opi++) {
		if (threadIdx.x == 0) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 2) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 3) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 4) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 5) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 6) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 7) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 8) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 9) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 10) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 11) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 12) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 13) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 14) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 15) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 16) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 17) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 18) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 19) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 20) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 21) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 22) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 23) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 24) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 25) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 26) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 27) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 28) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 29) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 30) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 31) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 32) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 33) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 34) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 35) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 36) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 37) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 38) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 39) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 40) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 41) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 42) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 43) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 44) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 45) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 46) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 47) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 48) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 49) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 50) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 51) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 52) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 53) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 54) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 55) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 56) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 57) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 58) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 59) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 60) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 61) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 62) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 63) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 64) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 65) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 66) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 67) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 68) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 69) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 70) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 71) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 72) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 73) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 74) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 75) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 76) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 77) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 78) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 79) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 80) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 81) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 82) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 83) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 84) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 85) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 86) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 87) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 88) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 89) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 90) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 91) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 92) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 93) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 94) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 95) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 96) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 97) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 98) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 99) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 100) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 101) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 102) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 103) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 104) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 105) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 106) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 107) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 108) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 109) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 110) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 111) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 112) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 113) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 114) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 115) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 116) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 117) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 118) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 119) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 120) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 121) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 122) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 123) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 124) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 125) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 126) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 127) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 128) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 129) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 130) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 131) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 132) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 133) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 134) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 135) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 136) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 137) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 138) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 139) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 140) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 141) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 142) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 143) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 144) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 145) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 146) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 147) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 148) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 149) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 150) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 151) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 152) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 153) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 154) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 155) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 156) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 157) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 158) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 159) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 160) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 161) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 162) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 163) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 164) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 165) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 166) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 167) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 168) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 169) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 170) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 171) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 172) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 173) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 174) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 175) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 176) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 177) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 178) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 179) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 180) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 181) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 182) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 183) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 184) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 185) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 186) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 187) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 188) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 189) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 190) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 191) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 192) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 193) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 194) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 195) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 196) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 197) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 198) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 199) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 200) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 201) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 202) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 203) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 204) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 205) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 206) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 207) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 208) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 209) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 210) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 211) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 212) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 213) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 214) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 215) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 216) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 217) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 218) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 219) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 220) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 221) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 222) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 223) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 224) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 225) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 226) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 227) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 228) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 229) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 230) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 231) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 232) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 233) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 234) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 235) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 236) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 237) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 238) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 239) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 240) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 241) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 242) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 243) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 244) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 245) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 246) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 247) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 248) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 249) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 250) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 251) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 252) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 253) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 254) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 255) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 256) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 257) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 258) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 259) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 260) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 261) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 262) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 263) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 264) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 265) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 266) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 267) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 268) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 269) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 270) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 271) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 272) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 273) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 274) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 275) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 276) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 277) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 278) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 279) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 280) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 281) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 282) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 283) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 284) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 285) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 286) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 287) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 288) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 289) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 290) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 291) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 292) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 293) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 294) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 295) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 296) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 297) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 298) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 299) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 300) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 301) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 302) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 303) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 304) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 305) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 306) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 307) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 308) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 309) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 310) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 311) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 312) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 313) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 314) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 315) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 316) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 317) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 318) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 319) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 320) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 321) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 322) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 323) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 324) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 325) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 326) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 327) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 328) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 329) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 330) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 331) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 332) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 333) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 334) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 335) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 336) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 337) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 338) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 339) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 340) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 341) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 342) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 343) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 344) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 345) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 346) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 347) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 348) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 349) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 350) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 351) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 352) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 353) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 354) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 355) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 356) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 357) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 358) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 359) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 360) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 361) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 362) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 363) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 364) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 365) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 366) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 367) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 368) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 369) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 370) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 371) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 372) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 373) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 374) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 375) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 376) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 377) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 378) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 379) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 380) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 381) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 382) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 383) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 384) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 385) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 386) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 387) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 388) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 389) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 390) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 391) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 392) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 393) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 394) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 395) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 396) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 397) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 398) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 399) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 400) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 401) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 402) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 403) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 404) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 405) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 406) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 407) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 408) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 409) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 410) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 411) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 412) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 413) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 414) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 415) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 416) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 417) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 418) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 419) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 420) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 421) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 422) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 423) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 424) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 425) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 426) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 427) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 428) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 429) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 430) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 431) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 432) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 433) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 434) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 435) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 436) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 437) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 438) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 439) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 440) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 441) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 442) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 443) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 444) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 445) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 446) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 447) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 448) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 449) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 450) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 451) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 452) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 453) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 454) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 455) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 456) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 457) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 458) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 459) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 460) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 461) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 462) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 463) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 464) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 465) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 466) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 467) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 468) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 469) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 470) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 471) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 472) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 473) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 474) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 475) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 476) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 477) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 478) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 479) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 480) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 481) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 482) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 483) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 484) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 485) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 486) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 487) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 488) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 489) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 490) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 491) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 492) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 493) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 494) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 495) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 496) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 497) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 498) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 499) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 500) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 501) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 502) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 503) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 504) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 505) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 506) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 507) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 508) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 509) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 510) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 511) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 512) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 513) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 514) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 515) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 516) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 517) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 518) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 519) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 520) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 521) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 522) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 523) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 524) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 525) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 526) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 527) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 528) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 529) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 530) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 531) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 532) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 533) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 534) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 535) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 536) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 537) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 538) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 539) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 540) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 541) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 542) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 543) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 544) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 545) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 546) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 547) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 548) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 549) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 550) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 551) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 552) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 553) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 554) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 555) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 556) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 557) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 558) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 559) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 560) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 561) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 562) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 563) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 564) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 565) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 566) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 567) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 568) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 569) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 570) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 571) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 572) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 573) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 574) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 575) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 576) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 577) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 578) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 579) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 580) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 581) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 582) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 583) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 584) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 585) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 586) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 587) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 588) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 589) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 590) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 591) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 592) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 593) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 594) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 595) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 596) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 597) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 598) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 599) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 600) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 601) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 602) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 603) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 604) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 605) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 606) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 607) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 608) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 609) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 610) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 611) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 612) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 613) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 614) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 615) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 616) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 617) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 618) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 619) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 620) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 621) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 622) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 623) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 624) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 625) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 626) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 627) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 628) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 629) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 630) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 631) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 632) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 633) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 634) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 635) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 636) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 637) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 638) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 639) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 640) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 641) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 642) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 643) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 644) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 645) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 646) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 647) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 648) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 649) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 650) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 651) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 652) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 653) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 654) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 655) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 656) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 657) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 658) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 659) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 660) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 661) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 662) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 663) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 664) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 665) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 666) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 667) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 668) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 669) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 670) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 671) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 672) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 673) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 674) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 675) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 676) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 677) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 678) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 679) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 680) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 681) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 682) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 683) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 684) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 685) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 686) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 687) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 688) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 689) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 690) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 691) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 692) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 693) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 694) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 695) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 696) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 697) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 698) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 699) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 700) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 701) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 702) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 703) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 704) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 705) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 706) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 707) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 708) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 709) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 710) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 711) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 712) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 713) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 714) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 715) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 716) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 717) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 718) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 719) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 720) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 721) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 722) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 723) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 724) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 725) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 726) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 727) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 728) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 729) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 730) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 731) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 732) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 733) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 734) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 735) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 736) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 737) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 738) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 739) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 740) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 741) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 742) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 743) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 744) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 745) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 746) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 747) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 748) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 749) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 750) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 751) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 752) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 753) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 754) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 755) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 756) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 757) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 758) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 759) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 760) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 761) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 762) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 763) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 764) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 765) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 766) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 767) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 768) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 769) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 770) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 771) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 772) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 773) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 774) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 775) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 776) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 777) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 778) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 779) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 780) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 781) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 782) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 783) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 784) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 785) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 786) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 787) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 788) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 789) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 790) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 791) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 792) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 793) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 794) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 795) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 796) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 797) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 798) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 799) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 800) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 801) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 802) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 803) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 804) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 805) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 806) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 807) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 808) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 809) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 810) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 811) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 812) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 813) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 814) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 815) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 816) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 817) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 818) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 819) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 820) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 821) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 822) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 823) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 824) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 825) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 826) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 827) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 828) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 829) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 830) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 831) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 832) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 833) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 834) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 835) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 836) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 837) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 838) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 839) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 840) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 841) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 842) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 843) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 844) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 845) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 846) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 847) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 848) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 849) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 850) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 851) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 852) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 853) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 854) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 855) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 856) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 857) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 858) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 859) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 860) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 861) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 862) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 863) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 864) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 865) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 866) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 867) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 868) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 869) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 870) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 871) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 872) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 873) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 874) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 875) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 876) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 877) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 878) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 879) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 880) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 881) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 882) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 883) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 884) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 885) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 886) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 887) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 888) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 889) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 890) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 891) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 892) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 893) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 894) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 895) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 896) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 897) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 898) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 899) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 900) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 901) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 902) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 903) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 904) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 905) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 906) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 907) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 908) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 909) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 910) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 911) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 912) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 913) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 914) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 915) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 916) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 917) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 918) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 919) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 920) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 921) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 922) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 923) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 924) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 925) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 926) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 927) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 928) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 929) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 930) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 931) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 932) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 933) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 934) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 935) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 936) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 937) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 938) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 939) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 940) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 941) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 942) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 943) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 944) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 945) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 946) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 947) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 948) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 949) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 950) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 951) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 952) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 953) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 954) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 955) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 956) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 957) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 958) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 959) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 960) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 961) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 962) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 963) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 964) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 965) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 966) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 967) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 968) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 969) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 970) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 971) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 972) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 973) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 974) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 975) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 976) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 977) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 978) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 979) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 980) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 981) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 982) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 983) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 984) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 985) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 986) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 987) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 988) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 989) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 990) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 991) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 992) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 993) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 994) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 995) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 996) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 997) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 998) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 999) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1000) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1001) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1002) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1003) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1004) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1005) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1006) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1007) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1008) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1009) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1010) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1011) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1012) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1013) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1014) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1015) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1016) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1017) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1018) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1019) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1020) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1021) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1022) {
			value = common_int_input[threadIdx.x] & opi;
		} else if (threadIdx.x == 1023) {
			value = common_int_input[threadIdx.x] & opi;
		}
	}
	dst_1[i] = value;

	dst_2[i] = value;

	dst_3[i] = value;

}

#endif /* BRANCH_KERNEL_H_ */
