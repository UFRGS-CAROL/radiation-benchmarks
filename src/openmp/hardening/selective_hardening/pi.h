#ifndef _PI_H_
#define _PI_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>

double pi_montecarlo_parallel(int niter);

#endif
