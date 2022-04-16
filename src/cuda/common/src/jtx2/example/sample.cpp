#include "JTX2Inst.h"
#include <unistd.h>
#include <iostream>

int main(int argc, char *argv[]) {
	std::string name="test.log";
	rad::JTX2Inst measure(0,name);
	
	measure.start_profile();

	std::cout << "SLEEPING FOR "<<argv[0] <<" seconds" << std::endl;
	sleep(atoi(argv[0]));
	measure.end_profile();

	//auto test = measure.get_data_from_iteration();
	//for (auto t : test) {
	//	std::cout << t << std::endl;
	//}

	return 0;
}
