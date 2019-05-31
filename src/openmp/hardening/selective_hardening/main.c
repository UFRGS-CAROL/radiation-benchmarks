#include "pi.h"

double pi_montecarlo_sequential(int niter)
{
	double x,y;
	unsigned int i,count=0; /* # of points in the 1st quadrant of unit circle */
    	double pi;
	
   	srand(time(NULL));

   	for ( i=0; i<niter; i++) 
	{
		x = (double)rand()/RAND_MAX;
		y = (double)rand()/RAND_MAX;
	      	
		if (x*x+y*y<=1) 
		{
			count++;
		}
       	}

	//printf("Count:%d\n",count);
    	pi=(double)count/niter*4;
	
	return pi;
}

int main(int argc, char** argv)
{
    	struct timeval time_start, time_end;
    	double elapsed;
	long unsigned int n;
    	int nthreads;
	double pi;
   
    	if(argc != 3)
	{
		printf("Usage: ./pi[_hardened] 2^num_iterations num_threads\n\n");
		exit(1);
	}
   
	n = pow(2, atoi(argv[1]));
    	nthreads = atoi(argv[2]);

	printf("Number of iterations: %lu\n", n);
	printf("Number of threads: %d\n\n", nthreads);

	omp_set_num_threads(nthreads); 

	gettimeofday(&time_start, NULL);
	pi = pi_montecarlo_parallel(n);
	gettimeofday(&time_end, NULL);
	elapsed = time_end.tv_sec - time_start.tv_sec + (time_end.tv_usec - time_start.tv_usec) / 1000000.0;
	
	printf("Time elapsed: %lf\n", elapsed);
	printf("Pi estimative: %lf\n\n", pi);
	//printf("%.6f;%g;",elapsed,pi);
	
	return 0;
}
