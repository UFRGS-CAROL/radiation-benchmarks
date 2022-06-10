#include "JTX2Inst.h"
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <ctime>    
#include <string>
#include <algorithm>
int main(int argc, char *argv[]) {
	auto start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	
	char name[1000];
	snprintf(name, sizeof(name), "%ssample.log",std::ctime(&start));
	
	std::string s(name,sizeof(name));
	std::replace( s.begin(), s.end(),' ','_');
	std::replace( s.begin(), s.end(),'\n','_');

	rad::JTX2Inst measure(atoi(argv[2]),s);
	
	measure.start_profile();

	std::cout << "SLEEPING FOR "<<argv[1] <<" seconds" << std::endl;
	sleep(atoi(argv[1]));
	measure.end_profile();

	//auto test = measure.get_data_from_iteration();
	//for (auto t : test) {
	//	std::cout << t << std::endl;
	//}

	return 0;
}
