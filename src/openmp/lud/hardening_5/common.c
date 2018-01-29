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
create_matrix_from_file(float **mp, const char* filename, int *size_p) {
    int i, j, size;
    float *m;
    FILE *fp = NULL;

    fp = fopen(filename, "rb");
    if ( fp == NULL) {
        return RET_FAILURE;
    }

    fscanf(fp, "%d\n", &size);

    m = (float*) malloc(sizeof(float)*size*size);
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
lud_verify(float *m, float *lu, int matrix_dim) {
    int i,j,k;
    float *tmp = (float*)malloc(matrix_dim*matrix_dim*sizeof(float));

    for (i=0; i < matrix_dim; i ++)
        for (j=0; j< matrix_dim; j++) {
            float sum = 0;
            float l,u;
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
matrix_duplicate(float *src, float **dst, int matrix_dim) {
    int s = matrix_dim*matrix_dim*sizeof(float);
    float *p = (float *) malloc (s);
    memcpy(p, src, s);
    *dst = p;
}


// Generate well-conditioned matrix internally  by Ke Wang 2013/08/07 22:20:06

func_ret_t
create_matrix(float **mp, int size) {
    float *m;
    int i,j;
    float lamda = -0.001;
    float coe[2*size-1];
    float coe_i =0.0;

    for (i=0; i < size; i++)
    {
        coe_i = 10*exp(lamda*i);
        j=size-1+i;
        coe[j]=coe_i;
        j=size-1-i;
        coe[j]=coe_i;
    }

    m = (float*) malloc(sizeof(float)*size*size);
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
