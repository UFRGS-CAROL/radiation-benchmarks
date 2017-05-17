/***********************************************************************/ 
//					INF-01191 - Arquiteturas Avançadas 
//
//								Trabalho 1
//
//			Calculo de Pi utilizando método Montecarlo e OpenMP
//
//
//				Gabriel Piscoya Dávila 		00246031
//				Rubens Ideron dos Santos	00243658	
//
/**********************************************************************/

#include "pi.h"

//#define HARDENING_DEBUG
#define READ_HARDENED_VAR(VAR_NAME_1, VAR_NAME_2, VAR_TYPE, VAR_SIZE) (*((VAR_TYPE*)hardened_compare_and_return((void*)(&VAR_NAME_1), (void*)(&VAR_NAME_2), VAR_SIZE)))
#define READ_HARDENED_ARRAY(ARRAY_NAME_1, ARRAY_NAME_2, ARRAY_TYPE, ARRAY_SIZE) ((ARRAY_TYPE)((void*)hardened_compare_and_return_array((void*)(&ARRAY_NAME_1), (void*)(&ARRAY_NAME_2), ARRAY_SIZE)))

inline void* hardened_compare_and_return(void* var_a, void* var_b, long long size)
{
        if(memcmp(var_a, var_b, size) != 0)
        {
                printf("\nHardening error: at file \"%s\"\n\n", __FILE__);
                exit(1);
        }

        return var_a;
}

inline void* hardened_compare_and_return_array(void* array_ptr_a, void* array_ptr_b, long long size)
{
	char* bytes_array_a = (char*)((char**)array_ptr_a);
	char* bytes_array_b = (char*)((char**)array_ptr_b);

#ifdef HARDENING_DEBUG
	printf("hardening_array: array_ptr_1 = %p, array_ptr_2 = %p, array_size = %d\n", bytes_array_a, bytes_array_b, size);
#endif

        if(memcmp(bytes_array_a, bytes_array_b, size) != 0)
        {
                printf("\nHardening error: at file \"%s\"\n\n", __FILE__);
                exit(1);
        }

        return array_ptr_a;
}

// Compute a pseudorandom double 
// from a random integer between 0 and 32767
// Output value in range [0, 1]

double pi_montecarlo_parallel(int niter)
{
   	double x_hardened_1, x_hardened_2, y_hardened_1, y_hardened_2;
   	unsigned int i, count; /* # of points in the 1st quadrant of unit circle */
   	double pi;
   	unsigned int g_seed;

   	count = 0;
	#pragma omp parallel private(g_seed)
	{
   		/* initialize seeds - one for each thread */
   		g_seed = time(NULL)*(omp_get_thread_num()+1);

		#pragma omp for private(x_hardened_1, y_hardened_1) schedule(static) reduction(+:count)
   		for ( i=0; i<niter; i++) 
		{
      			g_seed = (214013*g_seed+2531011);
      			x_hardened_1 = (double)((g_seed>>16)&0x7FFF)/32767;
			//x_hardened_2 = (double)((g_seed>>16)&0x7FFF)/32767;

      			g_seed = (214013*g_seed+2531011);
      			y_hardened_1 = (double)((g_seed>>16)&0x7FFF)/32767;
			//y_hardened_2 = (double)((g_seed>>16)&0x7FFF)/32767;

			//printf("x=%f,y=%f, thread=%d\n",x,y, omp_get_thread_num());
			double x_temp, y_temp;
			x_temp = x_hardened_1;//READ_HARDENED_VAR(x_hardened_1, x_hardened_2, double, sizeof(double));
			y_temp = y_hardened_1;//READ_HARDENED_VAR(y_hardened_1, y_hardened_2, double, sizeof(double));

      			if (x_temp*x_temp+y_temp*y_temp<=1)
			{
				count++;
			}
      		}
	}
   
	pi=(double)count/niter*4;
	
	return pi;
}
