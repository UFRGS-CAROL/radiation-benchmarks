#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "common.h"

void stopwatch_start(stopwatch *sw) {
    if (sw == NULL)
        return;

    bzero(&sw->begin, sizeof(struct timeval));
    bzero(&sw->end  , sizeof(struct timeval));

    gettimeofday(&sw->begin, NULL);
}

void stopwatch_stop(stopwatch *sw) {
    if (sw == NULL)
        return;

    gettimeofday(&sw->end, NULL);
}

double
get_interval_by_sec(stopwatch *sw) {
    if (sw == NULL)
        return 0;
    return ((double)(sw->end.tv_sec-sw->begin.tv_sec)+(double)(sw->end.tv_usec-sw->begin.tv_usec)/1000000);
}


func_ret_t
create_matrix_from_file(FP **mp, const char* filename, int *size_p) {
    int i, j, size;
    FP *m;
    FILE *fp = NULL;

    fp = fopen(filename, "rb");
    if ( fp == NULL) {
        return RET_FAILURE;
    }

    fscanf(fp, "%d\n", &size);

    m = (FP*) malloc(sizeof(FP)*size*size);
    if ( m == NULL) {
        fclose(fp);
        return RET_FAILURE;
    }

    for (i=0; i < size; i++) {
        for (j=0; j < size; j++) {
            fscanf(fp, "%f ", m+i*size+j);
        }
    }

    fclose(fp);

    *size_p = size;
    *mp = m;

    return RET_SUCCESS;
}




func_ret_t
lud_verify(FP *m, FP *lu, int matrix_dim) {
    int i,j,k;
    FP *tmp = (FP*)malloc(matrix_dim*matrix_dim*sizeof(FP));

    for (i=0; i < matrix_dim; i ++)
        for (j=0; j< matrix_dim; j++) {
            FP sum = 0;
            FP l,u;
            for (k=0; k <= MIN(i,j); k++) {
                if ( i==k)
                    l=1;
                else
                    l=lu[i*matrix_dim+k];
                u=lu[k*matrix_dim+j];
                sum+=l*u;
            }
            tmp[i*matrix_dim+j] = sum;
        }

    for (i=0; i<matrix_dim; i++) {
        for (j=0; j<matrix_dim; j++) {
            if ( fabs(m[i*matrix_dim+j]-tmp[i*matrix_dim+j]) > 0.0001)
                printf("dismatch at (%d, %d): (o)%f (n)%f\n", i, j, m[i*matrix_dim+j], tmp[i*matrix_dim+j]);
        }
    }
    free(tmp);
}

void
matrix_duplicate(FP *src, FP **dst, int matrix_dim) {
    int s = matrix_dim*matrix_dim*sizeof(FP);
    FP *p = (FP *) malloc (s);
    memcpy(p, src, s);
    *dst = p;
}


// Generate well-conditioned matrix internally  by Ke Wang 2013/08/07 22:20:06

func_ret_t
create_matrix(FP **mp, int size) {
    FP *m;
    int i,j;
    FP lamda = -0.001;
    FP coe[2*size-1];
    FP coe_i =0.0;

    for (i=0; i < size; i++)
    {
        coe_i = 10*exp(lamda*i);
        j=size-1+i;
        coe[j]=coe_i;
        j=size-1-i;
        coe[j]=coe_i;
    }

    m = (FP*) malloc(sizeof(FP)*size*size);
    if ( m == NULL) {
        return RET_FAILURE;
    }

    for (i=0; i < size; i++) {
        for (j=0; j < size; j++) {
            m[i*size+j]=coe[size-1-i+j];
        }
    }

    *mp = m;

    return RET_SUCCESS;
}
