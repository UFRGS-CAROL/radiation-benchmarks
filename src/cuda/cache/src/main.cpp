/*
 ============================================================================
 Name        : main.cpp
 Author      : Fernando
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include "NVMLWrapper.h"
#include "kernels.h"


void setup_for_kepler(){

}

void setup_for_volta(){

}

int main(void) {
	NVMLWrapper b(0);

	b.start_collecting_data();
	b.end_collecting_data();
	b.print_device_info();

	return 0;
}

