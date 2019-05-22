/*
 * Parameters.h
 *
 *  Created on: 22/05/2019
 *      Author: fernando
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <string>

typedef enum {
	HALF, SINGLE, DOUBLE
} PRECISION;

typedef enum {
	NONE, DMR, TMR, DMRMIXED, TMRMIXED
} REDUNDANCY;

#define NUMBER_PAR_PER_BOX 192	 // keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

#define NUMBER_THREADS 192		 // this should be roughly equal to NUMBER_PAR_PER_BOX for best performance

class Parameters {
public:
	long iterations;
	bool verbose;
	bool fault_injection;
	bool generate;

	std::string test_precision_description;
	std::string test_redundancy_description;

	std::string test_name;
	PRECISION precision;
	REDUNDANCY redundancy;


	int boxes;
	std::string input_distances;
	std::string input_charges;
	std::string output_gold;
	int nstreams;
	bool gpu_check;



	Parameters(int argc, char** argv);
	Parameters();
	virtual ~Parameters();

	void usage(int argc, char** argv);

	friend std::ostream& operator<<(std::ostream& os, const Parameters& p);
};

#endif /* PARAMETERS_H_ */
