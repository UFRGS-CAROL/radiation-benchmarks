/*
 * Parameters.h
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <string>

struct Parameters {
	int boxes;
	std::string input_distances, input_charges, output_gold;
	int iterations, fault_injection, nstreams;
	bool verbose;
	bool gpu_check;
	bool generate;
	std::string test_precision_description;

	Parameters();

	Parameters(int argc, char** argv);

	virtual ~Parameters();

	void usage(int argc, char** argv);
	friend std::ostream& operator<<(std::ostream& os, const Parameters& p);

private:
	void del_arg(int argc, char **argv, int index);
	int find_int_arg(int argc, char **argv, std::string arg, int def);
	std::string find_char_arg(int argc, char **argv, std::string arg,
			std::string def);
	bool find_arg(int argc, char** argv, std::string arg);
};

#endif /* PARAMETERS_H_ */
