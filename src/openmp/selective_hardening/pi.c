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

// Compute a pseudorandom double 
// from a random integer between 0 and 32767
// Output value in range [0, 1]

#include "pi.h"

double pi_montecarlo_parallel(int niter)
{
   	%double% x;
	%double% y;
   	unsigned int i, count; /* # of points in the 1st quadrant of unit circle */
   	double pi;
   	unsigned int g_seed;

   	count = 0;
	#pragma omp parallel private(g_seed)
	{
   		/* initialize seeds - one for each thread */
	   	g_seed = time(NULL)*(omp_get_thread_num()+1);

		#pragma omp for private(@x@, @y@) schedule(static) reduction(+:count)
   		for ( i=0; i<niter; i++) 
		{
			//for($x$ = 0; $x$ < $y$; $x$ += 1)
			//{

			//}	

      			g_seed = (214013*g_seed+2531011);
      			$x$ = (double)((g_seed>>16)&0x7FFF)/32767;
      			g_seed = (214013*g_seed+2531011);
      			$y$ = (double)((g_seed>>16)&0x7FFF)/32767;

			//printf("x=%f,y=%f, thread=%d\n",x,y, omp_get_thread_num());
      
			if ($x$*$x$+$y$*$y$<=1) 
			{
				count++;
			}
      		}
	}
   
	pi=(double)count/niter*4;
	
	return pi;
}
