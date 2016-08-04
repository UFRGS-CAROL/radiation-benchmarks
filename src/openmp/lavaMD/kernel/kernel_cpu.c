#ifdef __cplusplus
extern "C" {
#endif

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "./../main.h"
#include "kernel_cpu.h"

void  kernel_cpu( par_str par, dim_str dim, box_str* box, FOUR_VECTOR* rv, fp* qv, FOUR_VECTOR* fv)
{


    // parameters
    fp alpha;
    fp a2;

    // counters
    int i, j, k, l;

    // home box
    long first_i;
    FOUR_VECTOR* rA;
    FOUR_VECTOR* fA;

    // neighbor box
    int pointer;
    long first_j;
    FOUR_VECTOR* rB;
    fp* qB;

    // common
    fp r2;
    fp u2;
    fp fs;
    fp vij;
    fp fxij,fyij,fzij;
    THREE_VECTOR d;

    omp_set_num_threads(dim.cores_arg);

    alpha = par.alpha;
    a2 = 2.0*alpha*alpha;


    #pragma omp	parallel for \
    private(i, j, k) \
    private(first_i, rA, fA) \
    private(pointer, first_j, rB, qB) \
    private(r2, u2, fs, vij, fxij, fyij, fzij, d)
    for(l=0; l<dim.number_boxes; l=l+1) {

        first_i = box[l].offset;


        rA = &rv[first_i];
        fA = &fv[first_i];


        for (k=0; k<(1+box[l].nn); k++) {

            if(k==0) {
                pointer = l;
            } else {
                pointer = box[l].nei[k-1].number;
            }


            first_j = box[pointer].offset;

            rB = &rv[first_j];
            qB = &qv[first_j];


            for (i=0; i<NUMBER_PAR_PER_BOX; i=i+1) {

                // do for the # of particles in current (home or neighbor) box
                for (j=0; j<NUMBER_PAR_PER_BOX; j=j+1) {

                    // // coefficients
                    r2 = rA[i].v + rB[j].v - DOT(rA[i],rB[j]);
                    u2 = a2*r2;
                    vij= exp(-u2);
                    fs = 2.*vij;
                    d.x = rA[i].x  - rB[j].x;
                    d.y = rA[i].y  - rB[j].y;
                    d.z = rA[i].z  - rB[j].z;
                    fxij=fs*d.x;
                    fyij=fs*d.y;
                    fzij=fs*d.z;

                    // forces
                    fA[i].v +=  qB[j]*vij;
                    fA[i].x +=  qB[j]*fxij;
                    fA[i].y +=  qB[j]*fyij;
                    fA[i].z +=  qB[j]*fzij;

                }

            }

        }

    }

}

#ifdef __cplusplus
}
#endif
