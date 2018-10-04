#include "GEMM.cuh"
#include "GEMMWMMA.cuh"

#include <iostream>
#include <cuda_fp16.h>
#include <string>

void del_arg(int argc, char **argv, int index) {
	int i;
	for (i = index; i < argc - 1; ++i)
		argv[i] = argv[i + 1];
	argv[i] = 0;
}

int find_int_arg(int argc, char **argv, std::string arg, int def) {
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			def = atoi(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

std::string find_char_arg(int argc, char **argv, std::string arg, std::string def) {
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			def = std::string(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}



int find_arg(int argc, char* argv[], std::string arg) {
	int i;
	for (i = 0; i < argc; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			del_arg(argc, argv, i);
			return 1;
		}
	}
	return 0;
}

float find_float_arg(int argc, char **argv, char *arg, float def) {
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (0 == strcmp(argv[i], arg)) {
			def = atof(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}

int main(int argc, char** argv) {

	return 0;
}
