#include <stdio.h> 
#include <stdlib.h> 
#include <omp.h> 

#define MIC_NUM_THREADS 56000
  
#define SIZE 16
// 56 cores * 4 threads = 224
// ITER multiple of 224 to equally schedule threads among cores
#define ITER 22400000
  
extern double elapsedTime (void); 
  
int main() { 
	double startTime,  duration; 
	int i; 

	omp_set_num_threads(MIC_NUM_THREADS);

	int count = 0;

	#pragma offload target(mic) reduction(+:count)
	{

		__declspec(aligned(64)) double a[SIZE],b[SIZE],c[SIZE]; 


		for (i=0; i<SIZE;i++) 
		{ 
			c[i]=b[i]=a[i]=(double)rand(); 
		} 

		#pragma omp parallel for   
		for(i=0; i<ITER;i++) { 
			#pragma vector aligned (a,b,c) 
			a[0:SIZE]=b[0:SIZE]*c[0:SIZE]+a[0:SIZE]; 
		} 

		// need the gold output to chech errors
		// count++;???
	}

	printf("error count = %d\n",count);

	return 0; 

} 
