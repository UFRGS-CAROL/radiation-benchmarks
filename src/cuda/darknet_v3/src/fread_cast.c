#include "type.h"



/**
 * Read a file for all precisions
 */
int fread_float_to_real_t(real_t* dst, size_t siz, size_t times, FILE* fp) {
	float* temp = (float*) calloc(times, sizeof(float));
	if (temp == NULL) {
		return -1;
	}
	size_t fread_result = fread(temp, sizeof(float), times, fp);
	if (fread_result != times) {
		free(temp);
		return -1;
	}

	for (size_t i = 0; i < times; i++) {
		//TODO: make ready for half
		dst[i] = (real_t)(temp[i]);
	}
	free(temp);
	return fread_result;

}
