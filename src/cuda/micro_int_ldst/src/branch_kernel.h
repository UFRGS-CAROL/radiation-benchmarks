#ifndef BRANCH_KERNEL_H_
#define BRANCH_KERNEL_H_

template<uint32_t UNROLL_MAX, typename int_t>
__global__ void branch_int_kernel(int_t* src, int_t* dst, uint32_t op) {
	const uint32_t i =  (blockDim.x * blockIdx.x + threadIdx.x);

	if (threadIdx.x == 0) dst[i] = 0;
	else if (threadIdx.x == 1){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1;
	}
	else if (threadIdx.x == 2){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 2;
	}
	else if (threadIdx.x == 3){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 3;
	}
	else if (threadIdx.x == 4){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 4;
	}
	else if (threadIdx.x == 5){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 5;
	}
	else if (threadIdx.x == 6){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 6;
	}
	else if (threadIdx.x == 7){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 7;
	}
	else if (threadIdx.x == 8){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 8;
	}
	else if (threadIdx.x == 9){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 9;
	}
	else if (threadIdx.x == 10){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 10;
	}
	else if (threadIdx.x == 11){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 11;
	}
	else if (threadIdx.x == 12){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 12;
	}
	else if (threadIdx.x == 13){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 13;
	}
	else if (threadIdx.x == 14){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 14;
	}
	else if (threadIdx.x == 15){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 15;
	}
	else if (threadIdx.x == 16){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 16;
	}
	else if (threadIdx.x == 17){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 17;
	}
	else if (threadIdx.x == 18){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 18;
	}
	else if (threadIdx.x == 19){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 19;
	}
	else if (threadIdx.x == 20){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 20;
	}
	else if (threadIdx.x == 21){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 21;
	}
	else if (threadIdx.x == 22){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 22;
	}
	else if (threadIdx.x == 23){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 23;
	}
	else if (threadIdx.x == 24){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 24;
	}
	else if (threadIdx.x == 25){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 25;
	}
	else if (threadIdx.x == 26){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 26;
	}
	else if (threadIdx.x == 27){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 27;
	}
	else if (threadIdx.x == 28){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 28;
	}
	else if (threadIdx.x == 29){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 29;
	}
	else if (threadIdx.x == 30){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 30;
	}
	else if (threadIdx.x == 31){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 31;
	}
	else if (threadIdx.x == 32){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 32;
	}
	else if (threadIdx.x == 33){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 33;
	}
	else if (threadIdx.x == 34){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 34;
	}
	else if (threadIdx.x == 35){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 35;
	}
	else if (threadIdx.x == 36){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 36;
	}
	else if (threadIdx.x == 37){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 37;
	}
	else if (threadIdx.x == 38){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 38;
	}
	else if (threadIdx.x == 39){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 39;
	}
	else if (threadIdx.x == 40){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 40;
	}
	else if (threadIdx.x == 41){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 41;
	}
	else if (threadIdx.x == 42){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 42;
	}
	else if (threadIdx.x == 43){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 43;
	}
	else if (threadIdx.x == 44){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 44;
	}
	else if (threadIdx.x == 45){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 45;
	}
	else if (threadIdx.x == 46){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 46;
	}
	else if (threadIdx.x == 47){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 47;
	}
	else if (threadIdx.x == 48){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 48;
	}
	else if (threadIdx.x == 49){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 49;
	}
	else if (threadIdx.x == 50){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 50;
	}
	else if (threadIdx.x == 51){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 51;
	}
	else if (threadIdx.x == 52){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 52;
	}
	else if (threadIdx.x == 53){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 53;
	}
	else if (threadIdx.x == 54){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 54;
	}
	else if (threadIdx.x == 55){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 55;
	}
	else if (threadIdx.x == 56){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 56;
	}
	else if (threadIdx.x == 57){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 57;
	}
	else if (threadIdx.x == 58){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 58;
	}
	else if (threadIdx.x == 59){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 59;
	}
	else if (threadIdx.x == 60){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 60;
	}
	else if (threadIdx.x == 61){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 61;
	}
	else if (threadIdx.x == 62){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 62;
	}
	else if (threadIdx.x == 63){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 63;
	}
	else if (threadIdx.x == 64){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 64;
	}
	else if (threadIdx.x == 65){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 65;
	}
	else if (threadIdx.x == 66){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 66;
	}
	else if (threadIdx.x == 67){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 67;
	}
	else if (threadIdx.x == 68){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 68;
	}
	else if (threadIdx.x == 69){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 69;
	}
	else if (threadIdx.x == 70){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 70;
	}
	else if (threadIdx.x == 71){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 71;
	}
	else if (threadIdx.x == 72){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 72;
	}
	else if (threadIdx.x == 73){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 73;
	}
	else if (threadIdx.x == 74){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 74;
	}
	else if (threadIdx.x == 75){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 75;
	}
	else if (threadIdx.x == 76){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 76;
	}
	else if (threadIdx.x == 77){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 77;
	}
	else if (threadIdx.x == 78){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 78;
	}
	else if (threadIdx.x == 79){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 79;
	}
	else if (threadIdx.x == 80){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 80;
	}
	else if (threadIdx.x == 81){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 81;
	}
	else if (threadIdx.x == 82){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 82;
	}
	else if (threadIdx.x == 83){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 83;
	}
	else if (threadIdx.x == 84){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 84;
	}
	else if (threadIdx.x == 85){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 85;
	}
	else if (threadIdx.x == 86){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 86;
	}
	else if (threadIdx.x == 87){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 87;
	}
	else if (threadIdx.x == 88){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 88;
	}
	else if (threadIdx.x == 89){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 89;
	}
	else if (threadIdx.x == 90){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 90;
	}
	else if (threadIdx.x == 91){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 91;
	}
	else if (threadIdx.x == 92){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 92;
	}
	else if (threadIdx.x == 93){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 93;
	}
	else if (threadIdx.x == 94){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 94;
	}
	else if (threadIdx.x == 95){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 95;
	}
	else if (threadIdx.x == 96){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 96;
	}
	else if (threadIdx.x == 97){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 97;
	}
	else if (threadIdx.x == 98){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 98;
	}
	else if (threadIdx.x == 99){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 99;
	}
	else if (threadIdx.x == 100){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 100;
	}
	else if (threadIdx.x == 101){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 101;
	}
	else if (threadIdx.x == 102){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 102;
	}
	else if (threadIdx.x == 103){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 103;
	}
	else if (threadIdx.x == 104){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 104;
	}
	else if (threadIdx.x == 105){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 105;
	}
	else if (threadIdx.x == 106){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 106;
	}
	else if (threadIdx.x == 107){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 107;
	}
	else if (threadIdx.x == 108){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 108;
	}
	else if (threadIdx.x == 109){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 109;
	}
	else if (threadIdx.x == 110){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 110;
	}
	else if (threadIdx.x == 111){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 111;
	}
	else if (threadIdx.x == 112){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 112;
	}
	else if (threadIdx.x == 113){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 113;
	}
	else if (threadIdx.x == 114){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 114;
	}
	else if (threadIdx.x == 115){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 115;
	}
	else if (threadIdx.x == 116){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 116;
	}
	else if (threadIdx.x == 117){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 117;
	}
	else if (threadIdx.x == 118){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 118;
	}
	else if (threadIdx.x == 119){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 119;
	}
	else if (threadIdx.x == 120){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 120;
	}
	else if (threadIdx.x == 121){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 121;
	}
	else if (threadIdx.x == 122){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 122;
	}
	else if (threadIdx.x == 123){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 123;
	}
	else if (threadIdx.x == 124){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 124;
	}
	else if (threadIdx.x == 125){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 125;
	}
	else if (threadIdx.x == 126){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 126;
	}
	else if (threadIdx.x == 127){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 127;
	}
	else if (threadIdx.x == 128){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 128;
	}
	else if (threadIdx.x == 129){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 129;
	}
	else if (threadIdx.x == 130){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 130;
	}
	else if (threadIdx.x == 131){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 131;
	}
	else if (threadIdx.x == 132){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 132;
	}
	else if (threadIdx.x == 133){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 133;
	}
	else if (threadIdx.x == 134){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 134;
	}
	else if (threadIdx.x == 135){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 135;
	}
	else if (threadIdx.x == 136){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 136;
	}
	else if (threadIdx.x == 137){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 137;
	}
	else if (threadIdx.x == 138){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 138;
	}
	else if (threadIdx.x == 139){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 139;
	}
	else if (threadIdx.x == 140){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 140;
	}
	else if (threadIdx.x == 141){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 141;
	}
	else if (threadIdx.x == 142){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 142;
	}
	else if (threadIdx.x == 143){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 143;
	}
	else if (threadIdx.x == 144){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 144;
	}
	else if (threadIdx.x == 145){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 145;
	}
	else if (threadIdx.x == 146){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 146;
	}
	else if (threadIdx.x == 147){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 147;
	}
	else if (threadIdx.x == 148){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 148;
	}
	else if (threadIdx.x == 149){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 149;
	}
	else if (threadIdx.x == 150){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 150;
	}
	else if (threadIdx.x == 151){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 151;
	}
	else if (threadIdx.x == 152){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 152;
	}
	else if (threadIdx.x == 153){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 153;
	}
	else if (threadIdx.x == 154){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 154;
	}
	else if (threadIdx.x == 155){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 155;
	}
	else if (threadIdx.x == 156){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 156;
	}
	else if (threadIdx.x == 157){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 157;
	}
	else if (threadIdx.x == 158){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 158;
	}
	else if (threadIdx.x == 159){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 159;
	}
	else if (threadIdx.x == 160){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 160;
	}
	else if (threadIdx.x == 161){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 161;
	}
	else if (threadIdx.x == 162){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 162;
	}
	else if (threadIdx.x == 163){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 163;
	}
	else if (threadIdx.x == 164){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 164;
	}
	else if (threadIdx.x == 165){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 165;
	}
	else if (threadIdx.x == 166){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 166;
	}
	else if (threadIdx.x == 167){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 167;
	}
	else if (threadIdx.x == 168){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 168;
	}
	else if (threadIdx.x == 169){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 169;
	}
	else if (threadIdx.x == 170){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 170;
	}
	else if (threadIdx.x == 171){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 171;
	}
	else if (threadIdx.x == 172){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 172;
	}
	else if (threadIdx.x == 173){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 173;
	}
	else if (threadIdx.x == 174){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 174;
	}
	else if (threadIdx.x == 175){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 175;
	}
	else if (threadIdx.x == 176){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 176;
	}
	else if (threadIdx.x == 177){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 177;
	}
	else if (threadIdx.x == 178){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 178;
	}
	else if (threadIdx.x == 179){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 179;
	}
	else if (threadIdx.x == 180){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 180;
	}
	else if (threadIdx.x == 181){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 181;
	}
	else if (threadIdx.x == 182){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 182;
	}
	else if (threadIdx.x == 183){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 183;
	}
	else if (threadIdx.x == 184){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 184;
	}
	else if (threadIdx.x == 185){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 185;
	}
	else if (threadIdx.x == 186){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 186;
	}
	else if (threadIdx.x == 187){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 187;
	}
	else if (threadIdx.x == 188){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 188;
	}
	else if (threadIdx.x == 189){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 189;
	}
	else if (threadIdx.x == 190){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 190;
	}
	else if (threadIdx.x == 191){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 191;
	}
	else if (threadIdx.x == 192){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 192;
	}
	else if (threadIdx.x == 193){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 193;
	}
	else if (threadIdx.x == 194){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 194;
	}
	else if (threadIdx.x == 195){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 195;
	}
	else if (threadIdx.x == 196){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 196;
	}
	else if (threadIdx.x == 197){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 197;
	}
	else if (threadIdx.x == 198){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 198;
	}
	else if (threadIdx.x == 199){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 199;
	}
	else if (threadIdx.x == 200){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 200;
	}
	else if (threadIdx.x == 201){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 201;
	}
	else if (threadIdx.x == 202){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 202;
	}
	else if (threadIdx.x == 203){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 203;
	}
	else if (threadIdx.x == 204){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 204;
	}
	else if (threadIdx.x == 205){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 205;
	}
	else if (threadIdx.x == 206){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 206;
	}
	else if (threadIdx.x == 207){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 207;
	}
	else if (threadIdx.x == 208){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 208;
	}
	else if (threadIdx.x == 209){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 209;
	}
	else if (threadIdx.x == 210){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 210;
	}
	else if (threadIdx.x == 211){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 211;
	}
	else if (threadIdx.x == 212){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 212;
	}
	else if (threadIdx.x == 213){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 213;
	}
	else if (threadIdx.x == 214){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 214;
	}
	else if (threadIdx.x == 215){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 215;
	}
	else if (threadIdx.x == 216){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 216;
	}
	else if (threadIdx.x == 217){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 217;
	}
	else if (threadIdx.x == 218){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 218;
	}
	else if (threadIdx.x == 219){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 219;
	}
	else if (threadIdx.x == 220){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 220;
	}
	else if (threadIdx.x == 221){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 221;
	}
	else if (threadIdx.x == 222){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 222;
	}
	else if (threadIdx.x == 223){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 223;
	}
	else if (threadIdx.x == 224){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 224;
	}
	else if (threadIdx.x == 225){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 225;
	}
	else if (threadIdx.x == 226){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 226;
	}
	else if (threadIdx.x == 227){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 227;
	}
	else if (threadIdx.x == 228){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 228;
	}
	else if (threadIdx.x == 229){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 229;
	}
	else if (threadIdx.x == 230){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 230;
	}
	else if (threadIdx.x == 231){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 231;
	}
	else if (threadIdx.x == 232){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 232;
	}
	else if (threadIdx.x == 233){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 233;
	}
	else if (threadIdx.x == 234){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 234;
	}
	else if (threadIdx.x == 235){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 235;
	}
	else if (threadIdx.x == 236){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 236;
	}
	else if (threadIdx.x == 237){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 237;
	}
	else if (threadIdx.x == 238){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 238;
	}
	else if (threadIdx.x == 239){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 239;
	}
	else if (threadIdx.x == 240){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 240;
	}
	else if (threadIdx.x == 241){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 241;
	}
	else if (threadIdx.x == 242){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 242;
	}
	else if (threadIdx.x == 243){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 243;
	}
	else if (threadIdx.x == 244){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 244;
	}
	else if (threadIdx.x == 245){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 245;
	}
	else if (threadIdx.x == 246){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 246;
	}
	else if (threadIdx.x == 247){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 247;
	}
	else if (threadIdx.x == 248){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 248;
	}
	else if (threadIdx.x == 249){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 249;
	}
	else if (threadIdx.x == 250){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 250;
	}
	else if (threadIdx.x == 251){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 251;
	}
	else if (threadIdx.x == 252){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 252;
	}
	else if (threadIdx.x == 253){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 253;
	}
	else if (threadIdx.x == 254){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 254;
	}
	else if (threadIdx.x == 255){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 255;
	}
	else if (threadIdx.x == 256){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 256;
	}
	else if (threadIdx.x == 257){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 257;
	}
	else if (threadIdx.x == 258){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 258;
	}
	else if (threadIdx.x == 259){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 259;
	}
	else if (threadIdx.x == 260){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 260;
	}
	else if (threadIdx.x == 261){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 261;
	}
	else if (threadIdx.x == 262){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 262;
	}
	else if (threadIdx.x == 263){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 263;
	}
	else if (threadIdx.x == 264){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 264;
	}
	else if (threadIdx.x == 265){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 265;
	}
	else if (threadIdx.x == 266){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 266;
	}
	else if (threadIdx.x == 267){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 267;
	}
	else if (threadIdx.x == 268){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 268;
	}
	else if (threadIdx.x == 269){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 269;
	}
	else if (threadIdx.x == 270){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 270;
	}
	else if (threadIdx.x == 271){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 271;
	}
	else if (threadIdx.x == 272){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 272;
	}
	else if (threadIdx.x == 273){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 273;
	}
	else if (threadIdx.x == 274){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 274;
	}
	else if (threadIdx.x == 275){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 275;
	}
	else if (threadIdx.x == 276){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 276;
	}
	else if (threadIdx.x == 277){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 277;
	}
	else if (threadIdx.x == 278){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 278;
	}
	else if (threadIdx.x == 279){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 279;
	}
	else if (threadIdx.x == 280){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 280;
	}
	else if (threadIdx.x == 281){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 281;
	}
	else if (threadIdx.x == 282){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 282;
	}
	else if (threadIdx.x == 283){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 283;
	}
	else if (threadIdx.x == 284){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 284;
	}
	else if (threadIdx.x == 285){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 285;
	}
	else if (threadIdx.x == 286){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 286;
	}
	else if (threadIdx.x == 287){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 287;
	}
	else if (threadIdx.x == 288){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 288;
	}
	else if (threadIdx.x == 289){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 289;
	}
	else if (threadIdx.x == 290){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 290;
	}
	else if (threadIdx.x == 291){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 291;
	}
	else if (threadIdx.x == 292){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 292;
	}
	else if (threadIdx.x == 293){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 293;
	}
	else if (threadIdx.x == 294){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 294;
	}
	else if (threadIdx.x == 295){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 295;
	}
	else if (threadIdx.x == 296){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 296;
	}
	else if (threadIdx.x == 297){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 297;
	}
	else if (threadIdx.x == 298){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 298;
	}
	else if (threadIdx.x == 299){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 299;
	}
	else if (threadIdx.x == 300){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 300;
	}
	else if (threadIdx.x == 301){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 301;
	}
	else if (threadIdx.x == 302){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 302;
	}
	else if (threadIdx.x == 303){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 303;
	}
	else if (threadIdx.x == 304){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 304;
	}
	else if (threadIdx.x == 305){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 305;
	}
	else if (threadIdx.x == 306){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 306;
	}
	else if (threadIdx.x == 307){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 307;
	}
	else if (threadIdx.x == 308){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 308;
	}
	else if (threadIdx.x == 309){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 309;
	}
	else if (threadIdx.x == 310){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 310;
	}
	else if (threadIdx.x == 311){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 311;
	}
	else if (threadIdx.x == 312){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 312;
	}
	else if (threadIdx.x == 313){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 313;
	}
	else if (threadIdx.x == 314){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 314;
	}
	else if (threadIdx.x == 315){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 315;
	}
	else if (threadIdx.x == 316){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 316;
	}
	else if (threadIdx.x == 317){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 317;
	}
	else if (threadIdx.x == 318){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 318;
	}
	else if (threadIdx.x == 319){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 319;
	}
	else if (threadIdx.x == 320){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 320;
	}
	else if (threadIdx.x == 321){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 321;
	}
	else if (threadIdx.x == 322){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 322;
	}
	else if (threadIdx.x == 323){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 323;
	}
	else if (threadIdx.x == 324){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 324;
	}
	else if (threadIdx.x == 325){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 325;
	}
	else if (threadIdx.x == 326){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 326;
	}
	else if (threadIdx.x == 327){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 327;
	}
	else if (threadIdx.x == 328){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 328;
	}
	else if (threadIdx.x == 329){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 329;
	}
	else if (threadIdx.x == 330){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 330;
	}
	else if (threadIdx.x == 331){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 331;
	}
	else if (threadIdx.x == 332){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 332;
	}
	else if (threadIdx.x == 333){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 333;
	}
	else if (threadIdx.x == 334){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 334;
	}
	else if (threadIdx.x == 335){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 335;
	}
	else if (threadIdx.x == 336){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 336;
	}
	else if (threadIdx.x == 337){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 337;
	}
	else if (threadIdx.x == 338){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 338;
	}
	else if (threadIdx.x == 339){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 339;
	}
	else if (threadIdx.x == 340){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 340;
	}
	else if (threadIdx.x == 341){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 341;
	}
	else if (threadIdx.x == 342){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 342;
	}
	else if (threadIdx.x == 343){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 343;
	}
	else if (threadIdx.x == 344){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 344;
	}
	else if (threadIdx.x == 345){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 345;
	}
	else if (threadIdx.x == 346){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 346;
	}
	else if (threadIdx.x == 347){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 347;
	}
	else if (threadIdx.x == 348){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 348;
	}
	else if (threadIdx.x == 349){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 349;
	}
	else if (threadIdx.x == 350){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 350;
	}
	else if (threadIdx.x == 351){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 351;
	}
	else if (threadIdx.x == 352){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 352;
	}
	else if (threadIdx.x == 353){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 353;
	}
	else if (threadIdx.x == 354){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 354;
	}
	else if (threadIdx.x == 355){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 355;
	}
	else if (threadIdx.x == 356){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 356;
	}
	else if (threadIdx.x == 357){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 357;
	}
	else if (threadIdx.x == 358){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 358;
	}
	else if (threadIdx.x == 359){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 359;
	}
	else if (threadIdx.x == 360){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 360;
	}
	else if (threadIdx.x == 361){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 361;
	}
	else if (threadIdx.x == 362){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 362;
	}
	else if (threadIdx.x == 363){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 363;
	}
	else if (threadIdx.x == 364){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 364;
	}
	else if (threadIdx.x == 365){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 365;
	}
	else if (threadIdx.x == 366){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 366;
	}
	else if (threadIdx.x == 367){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 367;
	}
	else if (threadIdx.x == 368){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 368;
	}
	else if (threadIdx.x == 369){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 369;
	}
	else if (threadIdx.x == 370){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 370;
	}
	else if (threadIdx.x == 371){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 371;
	}
	else if (threadIdx.x == 372){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 372;
	}
	else if (threadIdx.x == 373){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 373;
	}
	else if (threadIdx.x == 374){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 374;
	}
	else if (threadIdx.x == 375){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 375;
	}
	else if (threadIdx.x == 376){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 376;
	}
	else if (threadIdx.x == 377){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 377;
	}
	else if (threadIdx.x == 378){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 378;
	}
	else if (threadIdx.x == 379){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 379;
	}
	else if (threadIdx.x == 380){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 380;
	}
	else if (threadIdx.x == 381){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 381;
	}
	else if (threadIdx.x == 382){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 382;
	}
	else if (threadIdx.x == 383){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 383;
	}
	else if (threadIdx.x == 384){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 384;
	}
	else if (threadIdx.x == 385){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 385;
	}
	else if (threadIdx.x == 386){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 386;
	}
	else if (threadIdx.x == 387){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 387;
	}
	else if (threadIdx.x == 388){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 388;
	}
	else if (threadIdx.x == 389){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 389;
	}
	else if (threadIdx.x == 390){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 390;
	}
	else if (threadIdx.x == 391){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 391;
	}
	else if (threadIdx.x == 392){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 392;
	}
	else if (threadIdx.x == 393){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 393;
	}
	else if (threadIdx.x == 394){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 394;
	}
	else if (threadIdx.x == 395){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 395;
	}
	else if (threadIdx.x == 396){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 396;
	}
	else if (threadIdx.x == 397){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 397;
	}
	else if (threadIdx.x == 398){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 398;
	}
	else if (threadIdx.x == 399){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 399;
	}
	else if (threadIdx.x == 400){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 400;
	}
	else if (threadIdx.x == 401){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 401;
	}
	else if (threadIdx.x == 402){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 402;
	}
	else if (threadIdx.x == 403){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 403;
	}
	else if (threadIdx.x == 404){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 404;
	}
	else if (threadIdx.x == 405){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 405;
	}
	else if (threadIdx.x == 406){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 406;
	}
	else if (threadIdx.x == 407){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 407;
	}
	else if (threadIdx.x == 408){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 408;
	}
	else if (threadIdx.x == 409){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 409;
	}
	else if (threadIdx.x == 410){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 410;
	}
	else if (threadIdx.x == 411){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 411;
	}
	else if (threadIdx.x == 412){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 412;
	}
	else if (threadIdx.x == 413){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 413;
	}
	else if (threadIdx.x == 414){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 414;
	}
	else if (threadIdx.x == 415){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 415;
	}
	else if (threadIdx.x == 416){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 416;
	}
	else if (threadIdx.x == 417){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 417;
	}
	else if (threadIdx.x == 418){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 418;
	}
	else if (threadIdx.x == 419){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 419;
	}
	else if (threadIdx.x == 420){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 420;
	}
	else if (threadIdx.x == 421){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 421;
	}
	else if (threadIdx.x == 422){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 422;
	}
	else if (threadIdx.x == 423){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 423;
	}
	else if (threadIdx.x == 424){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 424;
	}
	else if (threadIdx.x == 425){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 425;
	}
	else if (threadIdx.x == 426){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 426;
	}
	else if (threadIdx.x == 427){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 427;
	}
	else if (threadIdx.x == 428){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 428;
	}
	else if (threadIdx.x == 429){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 429;
	}
	else if (threadIdx.x == 430){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 430;
	}
	else if (threadIdx.x == 431){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 431;
	}
	else if (threadIdx.x == 432){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 432;
	}
	else if (threadIdx.x == 433){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 433;
	}
	else if (threadIdx.x == 434){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 434;
	}
	else if (threadIdx.x == 435){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 435;
	}
	else if (threadIdx.x == 436){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 436;
	}
	else if (threadIdx.x == 437){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 437;
	}
	else if (threadIdx.x == 438){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 438;
	}
	else if (threadIdx.x == 439){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 439;
	}
	else if (threadIdx.x == 440){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 440;
	}
	else if (threadIdx.x == 441){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 441;
	}
	else if (threadIdx.x == 442){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 442;
	}
	else if (threadIdx.x == 443){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 443;
	}
	else if (threadIdx.x == 444){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 444;
	}
	else if (threadIdx.x == 445){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 445;
	}
	else if (threadIdx.x == 446){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 446;
	}
	else if (threadIdx.x == 447){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 447;
	}
	else if (threadIdx.x == 448){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 448;
	}
	else if (threadIdx.x == 449){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 449;
	}
	else if (threadIdx.x == 450){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 450;
	}
	else if (threadIdx.x == 451){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 451;
	}
	else if (threadIdx.x == 452){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 452;
	}
	else if (threadIdx.x == 453){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 453;
	}
	else if (threadIdx.x == 454){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 454;
	}
	else if (threadIdx.x == 455){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 455;
	}
	else if (threadIdx.x == 456){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 456;
	}
	else if (threadIdx.x == 457){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 457;
	}
	else if (threadIdx.x == 458){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 458;
	}
	else if (threadIdx.x == 459){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 459;
	}
	else if (threadIdx.x == 460){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 460;
	}
	else if (threadIdx.x == 461){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 461;
	}
	else if (threadIdx.x == 462){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 462;
	}
	else if (threadIdx.x == 463){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 463;
	}
	else if (threadIdx.x == 464){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 464;
	}
	else if (threadIdx.x == 465){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 465;
	}
	else if (threadIdx.x == 466){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 466;
	}
	else if (threadIdx.x == 467){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 467;
	}
	else if (threadIdx.x == 468){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 468;
	}
	else if (threadIdx.x == 469){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 469;
	}
	else if (threadIdx.x == 470){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 470;
	}
	else if (threadIdx.x == 471){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 471;
	}
	else if (threadIdx.x == 472){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 472;
	}
	else if (threadIdx.x == 473){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 473;
	}
	else if (threadIdx.x == 474){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 474;
	}
	else if (threadIdx.x == 475){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 475;
	}
	else if (threadIdx.x == 476){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 476;
	}
	else if (threadIdx.x == 477){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 477;
	}
	else if (threadIdx.x == 478){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 478;
	}
	else if (threadIdx.x == 479){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 479;
	}
	else if (threadIdx.x == 480){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 480;
	}
	else if (threadIdx.x == 481){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 481;
	}
	else if (threadIdx.x == 482){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 482;
	}
	else if (threadIdx.x == 483){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 483;
	}
	else if (threadIdx.x == 484){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 484;
	}
	else if (threadIdx.x == 485){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 485;
	}
	else if (threadIdx.x == 486){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 486;
	}
	else if (threadIdx.x == 487){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 487;
	}
	else if (threadIdx.x == 488){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 488;
	}
	else if (threadIdx.x == 489){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 489;
	}
	else if (threadIdx.x == 490){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 490;
	}
	else if (threadIdx.x == 491){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 491;
	}
	else if (threadIdx.x == 492){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 492;
	}
	else if (threadIdx.x == 493){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 493;
	}
	else if (threadIdx.x == 494){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 494;
	}
	else if (threadIdx.x == 495){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 495;
	}
	else if (threadIdx.x == 496){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 496;
	}
	else if (threadIdx.x == 497){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 497;
	}
	else if (threadIdx.x == 498){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 498;
	}
	else if (threadIdx.x == 499){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 499;
	}
	else if (threadIdx.x == 500){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 500;
	}
	else if (threadIdx.x == 501){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 501;
	}
	else if (threadIdx.x == 502){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 502;
	}
	else if (threadIdx.x == 503){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 503;
	}
	else if (threadIdx.x == 504){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 504;
	}
	else if (threadIdx.x == 505){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 505;
	}
	else if (threadIdx.x == 506){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 506;
	}
	else if (threadIdx.x == 507){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 507;
	}
	else if (threadIdx.x == 508){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 508;
	}
	else if (threadIdx.x == 509){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 509;
	}
	else if (threadIdx.x == 510){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 510;
	}
	else if (threadIdx.x == 511){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 511;
	}
	else if (threadIdx.x == 512){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 512;
	}
	else if (threadIdx.x == 513){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 513;
	}
	else if (threadIdx.x == 514){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 514;
	}
	else if (threadIdx.x == 515){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 515;
	}
	else if (threadIdx.x == 516){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 516;
	}
	else if (threadIdx.x == 517){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 517;
	}
	else if (threadIdx.x == 518){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 518;
	}
	else if (threadIdx.x == 519){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 519;
	}
	else if (threadIdx.x == 520){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 520;
	}
	else if (threadIdx.x == 521){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 521;
	}
	else if (threadIdx.x == 522){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 522;
	}
	else if (threadIdx.x == 523){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 523;
	}
	else if (threadIdx.x == 524){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 524;
	}
	else if (threadIdx.x == 525){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 525;
	}
	else if (threadIdx.x == 526){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 526;
	}
	else if (threadIdx.x == 527){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 527;
	}
	else if (threadIdx.x == 528){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 528;
	}
	else if (threadIdx.x == 529){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 529;
	}
	else if (threadIdx.x == 530){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 530;
	}
	else if (threadIdx.x == 531){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 531;
	}
	else if (threadIdx.x == 532){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 532;
	}
	else if (threadIdx.x == 533){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 533;
	}
	else if (threadIdx.x == 534){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 534;
	}
	else if (threadIdx.x == 535){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 535;
	}
	else if (threadIdx.x == 536){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 536;
	}
	else if (threadIdx.x == 537){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 537;
	}
	else if (threadIdx.x == 538){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 538;
	}
	else if (threadIdx.x == 539){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 539;
	}
	else if (threadIdx.x == 540){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 540;
	}
	else if (threadIdx.x == 541){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 541;
	}
	else if (threadIdx.x == 542){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 542;
	}
	else if (threadIdx.x == 543){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 543;
	}
	else if (threadIdx.x == 544){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 544;
	}
	else if (threadIdx.x == 545){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 545;
	}
	else if (threadIdx.x == 546){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 546;
	}
	else if (threadIdx.x == 547){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 547;
	}
	else if (threadIdx.x == 548){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 548;
	}
	else if (threadIdx.x == 549){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 549;
	}
	else if (threadIdx.x == 550){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 550;
	}
	else if (threadIdx.x == 551){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 551;
	}
	else if (threadIdx.x == 552){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 552;
	}
	else if (threadIdx.x == 553){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 553;
	}
	else if (threadIdx.x == 554){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 554;
	}
	else if (threadIdx.x == 555){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 555;
	}
	else if (threadIdx.x == 556){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 556;
	}
	else if (threadIdx.x == 557){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 557;
	}
	else if (threadIdx.x == 558){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 558;
	}
	else if (threadIdx.x == 559){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 559;
	}
	else if (threadIdx.x == 560){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 560;
	}
	else if (threadIdx.x == 561){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 561;
	}
	else if (threadIdx.x == 562){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 562;
	}
	else if (threadIdx.x == 563){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 563;
	}
	else if (threadIdx.x == 564){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 564;
	}
	else if (threadIdx.x == 565){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 565;
	}
	else if (threadIdx.x == 566){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 566;
	}
	else if (threadIdx.x == 567){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 567;
	}
	else if (threadIdx.x == 568){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 568;
	}
	else if (threadIdx.x == 569){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 569;
	}
	else if (threadIdx.x == 570){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 570;
	}
	else if (threadIdx.x == 571){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 571;
	}
	else if (threadIdx.x == 572){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 572;
	}
	else if (threadIdx.x == 573){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 573;
	}
	else if (threadIdx.x == 574){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 574;
	}
	else if (threadIdx.x == 575){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 575;
	}
	else if (threadIdx.x == 576){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 576;
	}
	else if (threadIdx.x == 577){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 577;
	}
	else if (threadIdx.x == 578){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 578;
	}
	else if (threadIdx.x == 579){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 579;
	}
	else if (threadIdx.x == 580){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 580;
	}
	else if (threadIdx.x == 581){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 581;
	}
	else if (threadIdx.x == 582){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 582;
	}
	else if (threadIdx.x == 583){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 583;
	}
	else if (threadIdx.x == 584){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 584;
	}
	else if (threadIdx.x == 585){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 585;
	}
	else if (threadIdx.x == 586){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 586;
	}
	else if (threadIdx.x == 587){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 587;
	}
	else if (threadIdx.x == 588){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 588;
	}
	else if (threadIdx.x == 589){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 589;
	}
	else if (threadIdx.x == 590){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 590;
	}
	else if (threadIdx.x == 591){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 591;
	}
	else if (threadIdx.x == 592){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 592;
	}
	else if (threadIdx.x == 593){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 593;
	}
	else if (threadIdx.x == 594){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 594;
	}
	else if (threadIdx.x == 595){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 595;
	}
	else if (threadIdx.x == 596){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 596;
	}
	else if (threadIdx.x == 597){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 597;
	}
	else if (threadIdx.x == 598){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 598;
	}
	else if (threadIdx.x == 599){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 599;
	}
	else if (threadIdx.x == 600){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 600;
	}
	else if (threadIdx.x == 601){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 601;
	}
	else if (threadIdx.x == 602){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 602;
	}
	else if (threadIdx.x == 603){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 603;
	}
	else if (threadIdx.x == 604){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 604;
	}
	else if (threadIdx.x == 605){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 605;
	}
	else if (threadIdx.x == 606){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 606;
	}
	else if (threadIdx.x == 607){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 607;
	}
	else if (threadIdx.x == 608){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 608;
	}
	else if (threadIdx.x == 609){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 609;
	}
	else if (threadIdx.x == 610){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 610;
	}
	else if (threadIdx.x == 611){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 611;
	}
	else if (threadIdx.x == 612){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 612;
	}
	else if (threadIdx.x == 613){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 613;
	}
	else if (threadIdx.x == 614){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 614;
	}
	else if (threadIdx.x == 615){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 615;
	}
	else if (threadIdx.x == 616){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 616;
	}
	else if (threadIdx.x == 617){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 617;
	}
	else if (threadIdx.x == 618){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 618;
	}
	else if (threadIdx.x == 619){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 619;
	}
	else if (threadIdx.x == 620){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 620;
	}
	else if (threadIdx.x == 621){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 621;
	}
	else if (threadIdx.x == 622){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 622;
	}
	else if (threadIdx.x == 623){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 623;
	}
	else if (threadIdx.x == 624){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 624;
	}
	else if (threadIdx.x == 625){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 625;
	}
	else if (threadIdx.x == 626){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 626;
	}
	else if (threadIdx.x == 627){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 627;
	}
	else if (threadIdx.x == 628){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 628;
	}
	else if (threadIdx.x == 629){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 629;
	}
	else if (threadIdx.x == 630){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 630;
	}
	else if (threadIdx.x == 631){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 631;
	}
	else if (threadIdx.x == 632){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 632;
	}
	else if (threadIdx.x == 633){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 633;
	}
	else if (threadIdx.x == 634){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 634;
	}
	else if (threadIdx.x == 635){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 635;
	}
	else if (threadIdx.x == 636){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 636;
	}
	else if (threadIdx.x == 637){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 637;
	}
	else if (threadIdx.x == 638){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 638;
	}
	else if (threadIdx.x == 639){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 639;
	}
	else if (threadIdx.x == 640){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 640;
	}
	else if (threadIdx.x == 641){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 641;
	}
	else if (threadIdx.x == 642){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 642;
	}
	else if (threadIdx.x == 643){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 643;
	}
	else if (threadIdx.x == 644){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 644;
	}
	else if (threadIdx.x == 645){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 645;
	}
	else if (threadIdx.x == 646){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 646;
	}
	else if (threadIdx.x == 647){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 647;
	}
	else if (threadIdx.x == 648){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 648;
	}
	else if (threadIdx.x == 649){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 649;
	}
	else if (threadIdx.x == 650){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 650;
	}
	else if (threadIdx.x == 651){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 651;
	}
	else if (threadIdx.x == 652){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 652;
	}
	else if (threadIdx.x == 653){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 653;
	}
	else if (threadIdx.x == 654){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 654;
	}
	else if (threadIdx.x == 655){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 655;
	}
	else if (threadIdx.x == 656){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 656;
	}
	else if (threadIdx.x == 657){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 657;
	}
	else if (threadIdx.x == 658){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 658;
	}
	else if (threadIdx.x == 659){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 659;
	}
	else if (threadIdx.x == 660){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 660;
	}
	else if (threadIdx.x == 661){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 661;
	}
	else if (threadIdx.x == 662){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 662;
	}
	else if (threadIdx.x == 663){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 663;
	}
	else if (threadIdx.x == 664){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 664;
	}
	else if (threadIdx.x == 665){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 665;
	}
	else if (threadIdx.x == 666){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 666;
	}
	else if (threadIdx.x == 667){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 667;
	}
	else if (threadIdx.x == 668){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 668;
	}
	else if (threadIdx.x == 669){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 669;
	}
	else if (threadIdx.x == 670){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 670;
	}
	else if (threadIdx.x == 671){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 671;
	}
	else if (threadIdx.x == 672){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 672;
	}
	else if (threadIdx.x == 673){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 673;
	}
	else if (threadIdx.x == 674){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 674;
	}
	else if (threadIdx.x == 675){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 675;
	}
	else if (threadIdx.x == 676){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 676;
	}
	else if (threadIdx.x == 677){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 677;
	}
	else if (threadIdx.x == 678){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 678;
	}
	else if (threadIdx.x == 679){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 679;
	}
	else if (threadIdx.x == 680){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 680;
	}
	else if (threadIdx.x == 681){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 681;
	}
	else if (threadIdx.x == 682){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 682;
	}
	else if (threadIdx.x == 683){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 683;
	}
	else if (threadIdx.x == 684){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 684;
	}
	else if (threadIdx.x == 685){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 685;
	}
	else if (threadIdx.x == 686){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 686;
	}
	else if (threadIdx.x == 687){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 687;
	}
	else if (threadIdx.x == 688){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 688;
	}
	else if (threadIdx.x == 689){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 689;
	}
	else if (threadIdx.x == 690){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 690;
	}
	else if (threadIdx.x == 691){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 691;
	}
	else if (threadIdx.x == 692){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 692;
	}
	else if (threadIdx.x == 693){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 693;
	}
	else if (threadIdx.x == 694){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 694;
	}
	else if (threadIdx.x == 695){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 695;
	}
	else if (threadIdx.x == 696){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 696;
	}
	else if (threadIdx.x == 697){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 697;
	}
	else if (threadIdx.x == 698){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 698;
	}
	else if (threadIdx.x == 699){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 699;
	}
	else if (threadIdx.x == 700){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 700;
	}
	else if (threadIdx.x == 701){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 701;
	}
	else if (threadIdx.x == 702){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 702;
	}
	else if (threadIdx.x == 703){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 703;
	}
	else if (threadIdx.x == 704){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 704;
	}
	else if (threadIdx.x == 705){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 705;
	}
	else if (threadIdx.x == 706){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 706;
	}
	else if (threadIdx.x == 707){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 707;
	}
	else if (threadIdx.x == 708){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 708;
	}
	else if (threadIdx.x == 709){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 709;
	}
	else if (threadIdx.x == 710){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 710;
	}
	else if (threadIdx.x == 711){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 711;
	}
	else if (threadIdx.x == 712){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 712;
	}
	else if (threadIdx.x == 713){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 713;
	}
	else if (threadIdx.x == 714){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 714;
	}
	else if (threadIdx.x == 715){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 715;
	}
	else if (threadIdx.x == 716){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 716;
	}
	else if (threadIdx.x == 717){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 717;
	}
	else if (threadIdx.x == 718){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 718;
	}
	else if (threadIdx.x == 719){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 719;
	}
	else if (threadIdx.x == 720){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 720;
	}
	else if (threadIdx.x == 721){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 721;
	}
	else if (threadIdx.x == 722){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 722;
	}
	else if (threadIdx.x == 723){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 723;
	}
	else if (threadIdx.x == 724){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 724;
	}
	else if (threadIdx.x == 725){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 725;
	}
	else if (threadIdx.x == 726){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 726;
	}
	else if (threadIdx.x == 727){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 727;
	}
	else if (threadIdx.x == 728){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 728;
	}
	else if (threadIdx.x == 729){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 729;
	}
	else if (threadIdx.x == 730){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 730;
	}
	else if (threadIdx.x == 731){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 731;
	}
	else if (threadIdx.x == 732){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 732;
	}
	else if (threadIdx.x == 733){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 733;
	}
	else if (threadIdx.x == 734){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 734;
	}
	else if (threadIdx.x == 735){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 735;
	}
	else if (threadIdx.x == 736){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 736;
	}
	else if (threadIdx.x == 737){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 737;
	}
	else if (threadIdx.x == 738){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 738;
	}
	else if (threadIdx.x == 739){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 739;
	}
	else if (threadIdx.x == 740){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 740;
	}
	else if (threadIdx.x == 741){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 741;
	}
	else if (threadIdx.x == 742){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 742;
	}
	else if (threadIdx.x == 743){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 743;
	}
	else if (threadIdx.x == 744){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 744;
	}
	else if (threadIdx.x == 745){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 745;
	}
	else if (threadIdx.x == 746){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 746;
	}
	else if (threadIdx.x == 747){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 747;
	}
	else if (threadIdx.x == 748){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 748;
	}
	else if (threadIdx.x == 749){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 749;
	}
	else if (threadIdx.x == 750){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 750;
	}
	else if (threadIdx.x == 751){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 751;
	}
	else if (threadIdx.x == 752){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 752;
	}
	else if (threadIdx.x == 753){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 753;
	}
	else if (threadIdx.x == 754){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 754;
	}
	else if (threadIdx.x == 755){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 755;
	}
	else if (threadIdx.x == 756){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 756;
	}
	else if (threadIdx.x == 757){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 757;
	}
	else if (threadIdx.x == 758){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 758;
	}
	else if (threadIdx.x == 759){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 759;
	}
	else if (threadIdx.x == 760){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 760;
	}
	else if (threadIdx.x == 761){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 761;
	}
	else if (threadIdx.x == 762){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 762;
	}
	else if (threadIdx.x == 763){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 763;
	}
	else if (threadIdx.x == 764){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 764;
	}
	else if (threadIdx.x == 765){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 765;
	}
	else if (threadIdx.x == 766){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 766;
	}
	else if (threadIdx.x == 767){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 767;
	}
	else if (threadIdx.x == 768){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 768;
	}
	else if (threadIdx.x == 769){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 769;
	}
	else if (threadIdx.x == 770){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 770;
	}
	else if (threadIdx.x == 771){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 771;
	}
	else if (threadIdx.x == 772){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 772;
	}
	else if (threadIdx.x == 773){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 773;
	}
	else if (threadIdx.x == 774){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 774;
	}
	else if (threadIdx.x == 775){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 775;
	}
	else if (threadIdx.x == 776){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 776;
	}
	else if (threadIdx.x == 777){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 777;
	}
	else if (threadIdx.x == 778){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 778;
	}
	else if (threadIdx.x == 779){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 779;
	}
	else if (threadIdx.x == 780){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 780;
	}
	else if (threadIdx.x == 781){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 781;
	}
	else if (threadIdx.x == 782){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 782;
	}
	else if (threadIdx.x == 783){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 783;
	}
	else if (threadIdx.x == 784){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 784;
	}
	else if (threadIdx.x == 785){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 785;
	}
	else if (threadIdx.x == 786){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 786;
	}
	else if (threadIdx.x == 787){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 787;
	}
	else if (threadIdx.x == 788){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 788;
	}
	else if (threadIdx.x == 789){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 789;
	}
	else if (threadIdx.x == 790){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 790;
	}
	else if (threadIdx.x == 791){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 791;
	}
	else if (threadIdx.x == 792){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 792;
	}
	else if (threadIdx.x == 793){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 793;
	}
	else if (threadIdx.x == 794){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 794;
	}
	else if (threadIdx.x == 795){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 795;
	}
	else if (threadIdx.x == 796){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 796;
	}
	else if (threadIdx.x == 797){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 797;
	}
	else if (threadIdx.x == 798){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 798;
	}
	else if (threadIdx.x == 799){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 799;
	}
	else if (threadIdx.x == 800){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 800;
	}
	else if (threadIdx.x == 801){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 801;
	}
	else if (threadIdx.x == 802){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 802;
	}
	else if (threadIdx.x == 803){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 803;
	}
	else if (threadIdx.x == 804){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 804;
	}
	else if (threadIdx.x == 805){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 805;
	}
	else if (threadIdx.x == 806){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 806;
	}
	else if (threadIdx.x == 807){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 807;
	}
	else if (threadIdx.x == 808){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 808;
	}
	else if (threadIdx.x == 809){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 809;
	}
	else if (threadIdx.x == 810){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 810;
	}
	else if (threadIdx.x == 811){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 811;
	}
	else if (threadIdx.x == 812){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 812;
	}
	else if (threadIdx.x == 813){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 813;
	}
	else if (threadIdx.x == 814){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 814;
	}
	else if (threadIdx.x == 815){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 815;
	}
	else if (threadIdx.x == 816){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 816;
	}
	else if (threadIdx.x == 817){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 817;
	}
	else if (threadIdx.x == 818){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 818;
	}
	else if (threadIdx.x == 819){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 819;
	}
	else if (threadIdx.x == 820){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 820;
	}
	else if (threadIdx.x == 821){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 821;
	}
	else if (threadIdx.x == 822){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 822;
	}
	else if (threadIdx.x == 823){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 823;
	}
	else if (threadIdx.x == 824){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 824;
	}
	else if (threadIdx.x == 825){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 825;
	}
	else if (threadIdx.x == 826){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 826;
	}
	else if (threadIdx.x == 827){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 827;
	}
	else if (threadIdx.x == 828){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 828;
	}
	else if (threadIdx.x == 829){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 829;
	}
	else if (threadIdx.x == 830){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 830;
	}
	else if (threadIdx.x == 831){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 831;
	}
	else if (threadIdx.x == 832){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 832;
	}
	else if (threadIdx.x == 833){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 833;
	}
	else if (threadIdx.x == 834){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 834;
	}
	else if (threadIdx.x == 835){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 835;
	}
	else if (threadIdx.x == 836){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 836;
	}
	else if (threadIdx.x == 837){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 837;
	}
	else if (threadIdx.x == 838){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 838;
	}
	else if (threadIdx.x == 839){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 839;
	}
	else if (threadIdx.x == 840){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 840;
	}
	else if (threadIdx.x == 841){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 841;
	}
	else if (threadIdx.x == 842){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 842;
	}
	else if (threadIdx.x == 843){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 843;
	}
	else if (threadIdx.x == 844){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 844;
	}
	else if (threadIdx.x == 845){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 845;
	}
	else if (threadIdx.x == 846){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 846;
	}
	else if (threadIdx.x == 847){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 847;
	}
	else if (threadIdx.x == 848){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 848;
	}
	else if (threadIdx.x == 849){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 849;
	}
	else if (threadIdx.x == 850){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 850;
	}
	else if (threadIdx.x == 851){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 851;
	}
	else if (threadIdx.x == 852){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 852;
	}
	else if (threadIdx.x == 853){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 853;
	}
	else if (threadIdx.x == 854){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 854;
	}
	else if (threadIdx.x == 855){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 855;
	}
	else if (threadIdx.x == 856){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 856;
	}
	else if (threadIdx.x == 857){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 857;
	}
	else if (threadIdx.x == 858){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 858;
	}
	else if (threadIdx.x == 859){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 859;
	}
	else if (threadIdx.x == 860){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 860;
	}
	else if (threadIdx.x == 861){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 861;
	}
	else if (threadIdx.x == 862){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 862;
	}
	else if (threadIdx.x == 863){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 863;
	}
	else if (threadIdx.x == 864){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 864;
	}
	else if (threadIdx.x == 865){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 865;
	}
	else if (threadIdx.x == 866){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 866;
	}
	else if (threadIdx.x == 867){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 867;
	}
	else if (threadIdx.x == 868){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 868;
	}
	else if (threadIdx.x == 869){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 869;
	}
	else if (threadIdx.x == 870){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 870;
	}
	else if (threadIdx.x == 871){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 871;
	}
	else if (threadIdx.x == 872){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 872;
	}
	else if (threadIdx.x == 873){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 873;
	}
	else if (threadIdx.x == 874){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 874;
	}
	else if (threadIdx.x == 875){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 875;
	}
	else if (threadIdx.x == 876){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 876;
	}
	else if (threadIdx.x == 877){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 877;
	}
	else if (threadIdx.x == 878){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 878;
	}
	else if (threadIdx.x == 879){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 879;
	}
	else if (threadIdx.x == 880){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 880;
	}
	else if (threadIdx.x == 881){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 881;
	}
	else if (threadIdx.x == 882){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 882;
	}
	else if (threadIdx.x == 883){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 883;
	}
	else if (threadIdx.x == 884){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 884;
	}
	else if (threadIdx.x == 885){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 885;
	}
	else if (threadIdx.x == 886){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 886;
	}
	else if (threadIdx.x == 887){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 887;
	}
	else if (threadIdx.x == 888){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 888;
	}
	else if (threadIdx.x == 889){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 889;
	}
	else if (threadIdx.x == 890){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 890;
	}
	else if (threadIdx.x == 891){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 891;
	}
	else if (threadIdx.x == 892){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 892;
	}
	else if (threadIdx.x == 893){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 893;
	}
	else if (threadIdx.x == 894){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 894;
	}
	else if (threadIdx.x == 895){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 895;
	}
	else if (threadIdx.x == 896){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 896;
	}
	else if (threadIdx.x == 897){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 897;
	}
	else if (threadIdx.x == 898){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 898;
	}
	else if (threadIdx.x == 899){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 899;
	}
	else if (threadIdx.x == 900){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 900;
	}
	else if (threadIdx.x == 901){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 901;
	}
	else if (threadIdx.x == 902){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 902;
	}
	else if (threadIdx.x == 903){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 903;
	}
	else if (threadIdx.x == 904){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 904;
	}
	else if (threadIdx.x == 905){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 905;
	}
	else if (threadIdx.x == 906){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 906;
	}
	else if (threadIdx.x == 907){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 907;
	}
	else if (threadIdx.x == 908){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 908;
	}
	else if (threadIdx.x == 909){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 909;
	}
	else if (threadIdx.x == 910){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 910;
	}
	else if (threadIdx.x == 911){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 911;
	}
	else if (threadIdx.x == 912){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 912;
	}
	else if (threadIdx.x == 913){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 913;
	}
	else if (threadIdx.x == 914){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 914;
	}
	else if (threadIdx.x == 915){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 915;
	}
	else if (threadIdx.x == 916){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 916;
	}
	else if (threadIdx.x == 917){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 917;
	}
	else if (threadIdx.x == 918){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 918;
	}
	else if (threadIdx.x == 919){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 919;
	}
	else if (threadIdx.x == 920){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 920;
	}
	else if (threadIdx.x == 921){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 921;
	}
	else if (threadIdx.x == 922){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 922;
	}
	else if (threadIdx.x == 923){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 923;
	}
	else if (threadIdx.x == 924){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 924;
	}
	else if (threadIdx.x == 925){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 925;
	}
	else if (threadIdx.x == 926){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 926;
	}
	else if (threadIdx.x == 927){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 927;
	}
	else if (threadIdx.x == 928){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 928;
	}
	else if (threadIdx.x == 929){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 929;
	}
	else if (threadIdx.x == 930){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 930;
	}
	else if (threadIdx.x == 931){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 931;
	}
	else if (threadIdx.x == 932){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 932;
	}
	else if (threadIdx.x == 933){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 933;
	}
	else if (threadIdx.x == 934){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 934;
	}
	else if (threadIdx.x == 935){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 935;
	}
	else if (threadIdx.x == 936){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 936;
	}
	else if (threadIdx.x == 937){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 937;
	}
	else if (threadIdx.x == 938){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 938;
	}
	else if (threadIdx.x == 939){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 939;
	}
	else if (threadIdx.x == 940){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 940;
	}
	else if (threadIdx.x == 941){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 941;
	}
	else if (threadIdx.x == 942){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 942;
	}
	else if (threadIdx.x == 943){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 943;
	}
	else if (threadIdx.x == 944){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 944;
	}
	else if (threadIdx.x == 945){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 945;
	}
	else if (threadIdx.x == 946){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 946;
	}
	else if (threadIdx.x == 947){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 947;
	}
	else if (threadIdx.x == 948){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 948;
	}
	else if (threadIdx.x == 949){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 949;
	}
	else if (threadIdx.x == 950){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 950;
	}
	else if (threadIdx.x == 951){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 951;
	}
	else if (threadIdx.x == 952){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 952;
	}
	else if (threadIdx.x == 953){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 953;
	}
	else if (threadIdx.x == 954){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 954;
	}
	else if (threadIdx.x == 955){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 955;
	}
	else if (threadIdx.x == 956){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 956;
	}
	else if (threadIdx.x == 957){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 957;
	}
	else if (threadIdx.x == 958){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 958;
	}
	else if (threadIdx.x == 959){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 959;
	}
	else if (threadIdx.x == 960){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 960;
	}
	else if (threadIdx.x == 961){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 961;
	}
	else if (threadIdx.x == 962){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 962;
	}
	else if (threadIdx.x == 963){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 963;
	}
	else if (threadIdx.x == 964){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 964;
	}
	else if (threadIdx.x == 965){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 965;
	}
	else if (threadIdx.x == 966){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 966;
	}
	else if (threadIdx.x == 967){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 967;
	}
	else if (threadIdx.x == 968){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 968;
	}
	else if (threadIdx.x == 969){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 969;
	}
	else if (threadIdx.x == 970){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 970;
	}
	else if (threadIdx.x == 971){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 971;
	}
	else if (threadIdx.x == 972){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 972;
	}
	else if (threadIdx.x == 973){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 973;
	}
	else if (threadIdx.x == 974){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 974;
	}
	else if (threadIdx.x == 975){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 975;
	}
	else if (threadIdx.x == 976){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 976;
	}
	else if (threadIdx.x == 977){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 977;
	}
	else if (threadIdx.x == 978){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 978;
	}
	else if (threadIdx.x == 979){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 979;
	}
	else if (threadIdx.x == 980){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 980;
	}
	else if (threadIdx.x == 981){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 981;
	}
	else if (threadIdx.x == 982){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 982;
	}
	else if (threadIdx.x == 983){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 983;
	}
	else if (threadIdx.x == 984){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 984;
	}
	else if (threadIdx.x == 985){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 985;
	}
	else if (threadIdx.x == 986){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 986;
	}
	else if (threadIdx.x == 987){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 987;
	}
	else if (threadIdx.x == 988){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 988;
	}
	else if (threadIdx.x == 989){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 989;
	}
	else if (threadIdx.x == 990){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 990;
	}
	else if (threadIdx.x == 991){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 991;
	}
	else if (threadIdx.x == 992){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 992;
	}
	else if (threadIdx.x == 993){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 993;
	}
	else if (threadIdx.x == 994){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 994;
	}
	else if (threadIdx.x == 995){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 995;
	}
	else if (threadIdx.x == 996){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 996;
	}
	else if (threadIdx.x == 997){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 997;
	}
	else if (threadIdx.x == 998){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 998;
	}
	else if (threadIdx.x == 999){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 999;
	}
	else if (threadIdx.x == 1000){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1000;
	}
	else if (threadIdx.x == 1001){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1001;
	}
	else if (threadIdx.x == 1002){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1002;
	}
	else if (threadIdx.x == 1003){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1003;
	}
	else if (threadIdx.x == 1004){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1004;
	}
	else if (threadIdx.x == 1005){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1005;
	}
	else if (threadIdx.x == 1006){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1006;
	}
	else if (threadIdx.x == 1007){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1007;
	}
	else if (threadIdx.x == 1008){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1008;
	}
	else if (threadIdx.x == 1009){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1009;
	}
	else if (threadIdx.x == 1010){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1010;
	}
	else if (threadIdx.x == 1011){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1011;
	}
	else if (threadIdx.x == 1012){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1012;
	}
	else if (threadIdx.x == 1013){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1013;
	}
	else if (threadIdx.x == 1014){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1014;
	}
	else if (threadIdx.x == 1015){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1015;
	}
	else if (threadIdx.x == 1016){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1016;
	}
	else if (threadIdx.x == 1017){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1017;
	}
	else if (threadIdx.x == 1018){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1018;
	}
	else if (threadIdx.x == 1019){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1019;
	}
	else if (threadIdx.x == 1020){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1020;
	}
	else if (threadIdx.x == 1021){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1021;
	}
	else if (threadIdx.x == 1022){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1022;
	}
	else if (threadIdx.x == 1023){
		dst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + 1023;
	}
}

#endif /* BRANCH_KERNEL_H_ */
