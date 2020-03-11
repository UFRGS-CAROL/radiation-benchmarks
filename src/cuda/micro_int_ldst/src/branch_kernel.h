#ifndef BRANCH_KERNEL_H_
#define BRANCH_KERNEL_H_

template<uint32_t UNROLL_MAX, typename int_t>
__global__ void branch_int_kernel(int_t* src, int_t* dst, uint32_t op) {
	const uint32_t i =  (blockDim.x * blockIdx.x + threadIdx.x);

	if (threadIdx.x == 0) dst[i] = 0;
	else if (threadIdx.x == 1){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 2){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 3){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 4){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 5){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 6){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 7){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 8){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 9){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 10){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 11){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 12){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 13){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 14){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 15){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 16){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 17){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 18){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 19){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 20){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 21){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 22){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 23){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 24){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 25){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 26){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 27){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 28){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 29){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 30){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 31){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 32){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 33){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 34){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 35){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 36){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 37){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 38){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 39){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 40){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 41){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 42){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 43){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 44){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 45){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 46){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 47){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 48){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 49){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 50){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 51){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 52){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 53){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 54){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 55){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 56){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 57){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 58){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 59){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 60){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 61){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 62){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 63){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 64){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 65){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 66){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 67){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 68){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 69){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 70){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 71){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 72){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 73){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 74){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 75){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 76){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 77){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 78){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 79){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 80){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 81){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 82){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 83){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 84){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 85){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 86){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 87){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 88){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 89){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 90){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 91){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 92){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 93){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 94){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 95){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 96){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 97){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 98){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 99){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 100){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 101){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 102){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 103){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 104){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 105){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 106){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 107){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 108){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 109){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 110){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 111){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 112){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 113){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 114){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 115){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 116){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 117){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 118){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 119){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 120){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 121){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 122){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 123){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 124){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 125){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 126){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 127){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 128){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 129){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 130){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 131){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 132){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 133){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 134){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 135){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 136){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 137){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 138){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 139){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 140){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 141){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 142){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 143){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 144){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 145){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 146){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 147){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 148){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 149){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 150){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 151){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 152){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 153){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 154){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 155){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 156){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 157){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 158){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 159){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 160){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 161){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 162){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 163){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 164){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 165){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 166){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 167){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 168){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 169){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 170){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 171){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 172){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 173){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 174){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 175){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 176){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 177){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 178){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 179){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 180){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 181){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 182){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 183){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 184){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 185){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 186){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 187){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 188){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 189){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 190){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 191){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 192){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 193){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 194){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 195){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 196){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 197){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 198){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 199){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 200){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 201){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 202){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 203){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 204){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 205){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 206){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 207){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 208){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 209){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 210){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 211){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 212){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 213){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 214){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 215){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 216){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 217){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 218){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 219){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 220){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 221){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 222){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 223){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 224){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 225){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 226){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 227){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 228){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 229){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 230){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 231){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 232){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 233){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 234){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 235){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 236){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 237){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 238){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 239){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 240){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 241){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 242){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 243){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 244){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 245){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 246){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 247){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 248){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 249){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 250){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 251){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 252){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 253){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 254){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 255){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 256){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 257){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 258){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 259){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 260){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 261){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 262){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 263){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 264){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 265){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 266){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 267){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 268){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 269){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 270){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 271){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 272){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 273){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 274){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 275){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 276){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 277){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 278){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 279){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 280){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 281){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 282){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 283){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 284){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 285){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 286){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 287){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 288){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 289){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 290){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 291){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 292){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 293){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 294){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 295){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 296){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 297){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 298){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 299){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 300){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 301){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 302){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 303){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 304){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 305){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 306){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 307){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 308){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 309){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 310){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 311){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 312){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 313){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 314){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 315){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 316){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 317){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 318){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 319){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 320){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 321){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 322){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 323){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 324){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 325){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 326){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 327){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 328){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 329){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 330){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 331){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 332){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 333){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 334){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 335){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 336){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 337){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 338){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 339){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 340){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 341){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 342){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 343){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 344){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 345){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 346){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 347){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 348){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 349){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 350){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 351){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 352){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 353){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 354){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 355){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 356){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 357){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 358){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 359){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 360){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 361){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 362){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 363){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 364){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 365){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 366){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 367){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 368){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 369){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 370){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 371){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 372){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 373){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 374){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 375){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 376){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 377){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 378){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 379){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 380){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 381){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 382){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 383){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 384){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 385){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 386){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 387){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 388){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 389){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 390){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 391){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 392){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 393){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 394){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 395){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 396){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 397){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 398){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 399){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 400){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 401){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 402){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 403){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 404){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 405){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 406){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 407){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 408){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 409){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 410){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 411){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 412){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 413){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 414){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 415){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 416){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 417){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 418){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 419){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 420){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 421){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 422){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 423){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 424){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 425){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 426){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 427){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 428){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 429){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 430){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 431){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 432){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 433){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 434){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 435){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 436){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 437){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 438){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 439){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 440){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 441){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 442){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 443){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 444){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 445){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 446){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 447){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 448){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 449){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 450){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 451){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 452){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 453){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 454){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 455){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 456){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 457){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 458){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 459){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 460){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 461){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 462){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 463){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 464){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 465){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 466){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 467){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 468){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 469){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 470){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 471){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 472){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 473){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 474){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 475){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 476){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 477){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 478){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 479){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 480){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 481){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 482){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 483){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 484){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 485){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 486){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 487){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 488){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 489){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 490){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 491){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 492){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 493){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 494){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 495){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 496){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 497){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 498){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 499){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 500){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 501){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 502){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 503){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 504){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 505){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 506){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 507){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 508){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 509){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 510){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 511){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 512){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 513){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 514){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 515){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 516){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 517){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 518){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 519){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 520){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 521){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 522){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 523){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 524){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 525){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 526){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 527){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 528){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 529){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 530){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 531){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 532){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 533){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 534){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 535){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 536){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 537){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 538){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 539){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 540){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 541){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 542){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 543){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 544){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 545){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 546){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 547){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 548){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 549){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 550){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 551){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 552){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 553){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 554){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 555){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 556){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 557){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 558){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 559){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 560){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 561){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 562){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 563){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 564){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 565){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 566){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 567){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 568){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 569){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 570){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 571){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 572){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 573){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 574){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 575){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 576){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 577){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 578){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 579){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 580){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 581){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 582){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 583){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 584){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 585){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 586){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 587){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 588){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 589){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 590){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 591){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 592){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 593){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 594){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 595){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 596){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 597){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 598){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 599){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 600){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 601){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 602){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 603){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 604){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 605){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 606){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 607){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 608){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 609){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 610){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 611){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 612){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 613){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 614){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 615){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 616){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 617){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 618){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 619){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 620){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 621){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 622){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 623){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 624){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 625){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 626){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 627){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 628){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 629){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 630){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 631){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 632){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 633){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 634){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 635){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 636){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 637){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 638){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 639){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 640){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 641){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 642){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 643){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 644){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 645){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 646){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 647){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 648){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 649){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 650){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 651){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 652){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 653){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 654){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 655){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 656){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 657){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 658){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 659){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 660){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 661){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 662){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 663){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 664){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 665){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 666){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 667){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 668){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 669){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 670){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 671){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 672){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 673){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 674){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 675){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 676){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 677){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 678){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 679){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 680){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 681){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 682){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 683){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 684){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 685){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 686){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 687){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 688){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 689){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 690){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 691){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 692){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 693){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 694){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 695){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 696){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 697){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 698){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 699){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 700){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 701){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 702){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 703){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 704){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 705){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 706){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 707){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 708){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 709){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 710){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 711){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 712){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 713){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 714){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 715){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 716){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 717){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 718){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 719){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 720){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 721){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 722){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 723){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 724){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 725){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 726){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 727){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 728){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 729){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 730){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 731){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 732){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 733){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 734){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 735){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 736){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 737){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 738){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 739){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 740){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 741){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 742){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 743){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 744){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 745){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 746){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 747){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 748){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 749){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 750){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 751){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 752){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 753){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 754){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 755){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 756){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 757){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 758){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 759){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 760){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 761){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 762){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 763){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 764){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 765){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 766){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 767){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 768){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 769){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 770){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 771){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 772){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 773){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 774){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 775){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 776){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 777){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 778){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 779){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 780){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 781){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 782){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 783){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 784){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 785){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 786){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 787){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 788){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 789){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 790){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 791){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 792){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 793){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 794){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 795){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 796){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 797){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 798){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 799){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 800){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 801){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 802){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 803){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 804){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 805){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 806){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 807){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 808){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 809){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 810){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 811){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 812){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 813){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 814){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 815){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 816){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 817){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 818){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 819){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 820){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 821){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 822){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 823){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 824){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 825){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 826){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 827){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 828){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 829){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 830){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 831){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 832){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 833){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 834){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 835){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 836){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 837){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 838){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 839){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 840){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 841){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 842){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 843){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 844){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 845){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 846){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 847){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 848){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 849){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 850){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 851){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 852){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 853){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 854){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 855){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 856){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 857){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 858){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 859){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 860){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 861){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 862){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 863){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 864){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 865){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 866){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 867){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 868){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 869){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 870){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 871){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 872){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 873){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 874){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 875){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 876){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 877){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 878){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 879){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 880){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 881){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 882){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 883){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 884){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 885){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 886){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 887){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 888){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 889){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 890){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 891){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 892){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 893){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 894){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 895){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 896){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 897){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 898){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 899){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 900){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 901){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 902){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 903){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 904){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 905){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 906){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 907){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 908){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 909){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 910){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 911){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 912){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 913){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 914){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 915){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 916){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 917){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 918){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 919){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 920){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 921){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 922){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 923){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 924){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 925){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 926){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 927){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 928){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 929){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 930){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 931){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 932){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 933){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 934){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 935){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 936){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 937){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 938){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 939){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 940){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 941){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 942){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 943){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 944){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 945){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 946){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 947){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 948){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 949){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 950){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 951){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 952){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 953){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 954){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 955){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 956){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 957){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 958){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 959){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 960){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 961){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 962){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 963){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 964){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 965){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 966){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 967){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 968){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 969){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 970){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 971){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 972){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 973){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 974){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 975){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 976){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 977){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 978){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 979){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 980){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 981){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 982){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 983){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 984){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 985){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 986){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 987){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 988){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 989){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 990){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 991){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 992){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 993){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 994){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 995){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 996){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 997){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 998){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 999){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1000){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1001){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1002){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1003){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1004){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1005){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1006){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1007){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1008){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1009){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1010){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1011){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1012){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1013){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1014){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1015){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1016){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1017){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1018){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1019){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1020){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1021){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1022){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 1023){
		dst[i] = ((threadIdx.x % 2) ? threadIdx.x - UNROLL_MAX : threadIdx.x + UNROLL_MAX) + 1;
	}
}

#endif /* BRANCH_KERNEL_H_ */
