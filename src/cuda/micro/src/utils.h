/*
 * utils.h
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <random>

void __throw_line(std::string err, std::string line, std::string file);
bool file_exists(const std::string& name);

#define throw_line(err) __throw_line(std::string(err), std::to_string(__LINE__), std::string(__FILE__));

template<typename ... Types> struct Input {
};

template<>
struct Input<double> {
//#define OPS_PER_THREAD_OPERATION 1
	double INPUT_A = 1.1945305291614955E+103; // 0x5555555555555555
	double INPUT_B = 3.7206620809969885E-103; // 0x2AAAAAAAAAAAAAAA
	double OUTPUT_R = 4.444444444444444; //0x4011C71C71C71C71
};

template<>
struct Input<float> {
//#define OPS_PER_THREAD_OPERATION 1
	float INPUT_A = 1.4660155E+13; // 0x55555555
	float INPUT_B = 3.0316488E-13; // 0x2AAAAAAA
	float OUTPUT_R = 4.444444; //0x408E38E3
};

//template<>
//struct Input<half> {
////#define OPS_PER_THREAD_OPERATION 2
//	half INPUT_A = 1.066E+2; // 0x56AA
//	half INPUT_B = 4.166E-2; // 0x2955
//	half OUTPUT_R = 4.44; // 0x4471
//};

template<typename type_t>
void generate_new_array(std::vector<type_t>& raw_array) {
	Input<type_t> input_limits;
	// First create an instance of an engine.
	std::random_device rnd_device;
	// Specify the engine and distribution.
	std::mt19937 mersenne_engine { rnd_device() }; // Generates random integers
	std::uniform_real_distribution<type_t> dist { input_limits.OUTPUT_R,
			input_limits.OUTPUT_R };

	for (auto& i : raw_array)
		i = dist(mersenne_engine) + type_t(0.001); //never zero
}



#endif /* UTILS_H_ */
