#include <stdlib.h>
#include <sys/time.h>

 // Returns the current system time in microseconds
double get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double(tv.tv_sec * 1000000) + double(tv.tv_usec)) /1000000;
}
