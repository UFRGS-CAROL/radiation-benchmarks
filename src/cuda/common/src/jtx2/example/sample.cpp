#include "JTX2Inst.h"
#include <unistd.h>
#include <iostream>

int main() {
	rad::JTX2Inst measure;

	measure.start_collecting_data();

	std::cout << "SLEEPING FOR 5 seconds" << std::endl;
	sleep(5);
	measure.end_collecting_data();

	auto test = measure.get_data_from_iteration();
	for (auto t : test) {
		std::cout << t << std::endl;
	}

	return 0;
}
