/*
 * Parameters.h
 *
 *  Created on: 17/05/2019
 *      Author: fernando
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>

typedef enum {
	HALF, SINGLE, DOUBLE
} PRECISION;

typedef enum {
	NONE, DMR, TMR, DMRMIXED, TMRMIXED
} REDUNDANCY;

struct Parameters {
	int grid_cols, grid_rows;
	int nstreams;
	int sim_time;
	int size;
	int pyramid_height;

	long setup_loops;
	bool verbose;
	bool fault_injection;
	bool generate;

	std::string test_precision_description;
	std::string test_redundancy_description;

	std::string test_name;
	std::string tfile, pfile, ofile;

	PRECISION precision;
	REDUNDANCY redundancy;

	Parameters(int argc, char** argv);
	Parameters();
	virtual ~Parameters();

	void usage(int argc, char** argv);

	friend std::ostream& operator<<(std::ostream& os, const Parameters& p);
};

#endif /* PARAMETERS_H_ */

