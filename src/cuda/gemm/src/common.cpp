#include <stdexcept>
#include <iostream>

#include "common.h"

void exception(std::string msg, std::string file, int line) {
	throw std::runtime_error(msg + " at " + file + ":" + std::to_string(line));
}

void show_iteration_status(int it, bool verbose, double copy_time,
		double comparing_time, double computation_time,
		std::pair<int, int> errors) {
	if (verbose) {
		auto wasted_time = copy_time + comparing_time;
		auto full_time = wasted_time + computation_time;
		std::cout << "Iteration: " << it << " DMR errors " << errors.first
				<< ". " << "Radiation errors: " << errors.second << ". "
				<< "Time spent on computation: " << computation_time << "s. "
				<< "Time spent on comparing: " << comparing_time << "s. "
				<< "Time spent on copying: " << copy_time << "s. " << std::endl;
		std::cout << "Wasted time " << wasted_time << " ("
				<< int((wasted_time / full_time) * 100.0f) << "%)" << std::endl;
	} else {
//				std::cout << "Iteration: " << it << " DMR errors "
//						<< errors.first << ". " << "Radiation errors: "
//						<< errors.second << ". " << std::endl;
	}

}
