

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
//#include <sys/time.h>

#include <ctime>
#include <vector>

#include <climits>


#include "common.h"

extern long long get_time();
extern void videoSequence(std::vector<unsigned char>& I, int IszX, int IszY, int Nfr, std::vector<int>& seed) ;
extern void particleFilter(std::vector<unsigned char>& I, int IszX, int IszY, int Nfr,
		std::vector<int>& seed, int Nparticles) ;
extern float_t elapsed_time(long long start_time, long long end_time);

int main(int argc, char * argv[]) {

	char* usage = "float_t.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
	//check number of arguments
	if (argc != 9) {
		printf("%s\n", usage);
		return 0;
	}
	//check args deliminators
	if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z")
			|| strcmp(argv[7], "-np")) {
		printf("%s\n", usage);
		return 0;
	}

	int IszX, IszY, Nfr, Nparticles;

	//converting a string to a integer
	if (sscanf(argv[2], "%d", &IszX) == EOF) {
		printf("ERROR: dimX input is incorrect");
		return 0;
	}

	if (IszX <= 0) {
		printf("dimX must be > 0\n");
		return 0;
	}

	//converting a string to a integer
	if (sscanf(argv[4], "%d", &IszY) == EOF) {
		printf("ERROR: dimY input is incorrect");
		return 0;
	}

	if (IszY <= 0) {
		printf("dimY must be > 0\n");
		return 0;
	}

	//converting a string to a integer
	if (sscanf(argv[6], "%d", &Nfr) == EOF) {
		printf("ERROR: Number of frames input is incorrect");
		return 0;
	}

	if (Nfr <= 0) {
		printf("number of frames must be > 0\n");
		return 0;
	}

	//converting a string to a integer
	if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
		printf("ERROR: Number of particles input is incorrect");
		return 0;
	}

	if (Nparticles <= 0) {
		printf("Number of particles must be > 0\n");
		return 0;
	}
	//establish seed
//	int * seed = (int *) malloc(sizeof(int) * Nparticles);
	std::vector<int> seed(Nparticles);

	int i;
	for (i = 0; i < Nparticles; i++)
		seed[i] = time(0) * i;
	//malloc matrix
//	unsigned char * I = (unsigned char *) malloc(
//			sizeof(unsigned char) * IszX * IszY * Nfr);
	std::vector<unsigned char> I(IszX * IszY * Nfr);

	long long start = get_time();
	//call video sequence
	videoSequence(I, IszX, IszY, Nfr, seed);
	long long endVideoSequence = get_time();
	printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));
	//call particle filter
	particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);
	long long endParticleFilter = get_time();
	printf("PARTICLE FILTER TOOK %f\n",
			elapsed_time(endVideoSequence, endParticleFilter));
	printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));

//	free(seed);
//	free(I);
	return 0;
}
