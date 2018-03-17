#include "dgemm.h"
#include "../../selective_hardening/hardening.h"

void dgemm(double *A, double *B, double *C, long order, int block) {

    int     i,ii,j,jj,k,kk,jg,kg;
    int ig_hardened_1, ig_hardened_2;

    #pragma omp parallel private (i,j,k,ii,jj,kk,ig_hardened_1,ig_hardened_2,jg,kg)
    {
        double *AA=NULL, *BB=NULL, *CC=NULL;

        /* matrix blocks for local temporary copies*/
        AA = (double *) prk_malloc(block*(block+BOFFSET)*3*sizeof(double));
        if (!AA) {
            printf("Could not allocate space for matrix tiles on thread %d\n",
                   omp_get_thread_num());
            exit(1);
        }
        BB = AA + block*(block+BOFFSET);
        CC = BB + block*(block+BOFFSET);


        #pragma omp for
        for(jj = 0; jj < order; jj+=block) {
            for(kk = 0; kk < order; kk+=block) {

                for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)
                    for (kg=kk,k=0; kg<MIN(kk+block,order); k++,kg++)
                        BB_arr(j,k) =  B_arr(kg,jg);

                for(ii = 0; ii < order; ii+=block) {

                    for (kg=kk,k=0; kg<MIN(kk+block,order); k++,kg++)
                        for (ig_hardened_1=ii,ig_hardened_2=ii,i=0; READ_HARDENED_VAR_INT(ig_hardened_1, ig_hardened_2, "ig")<MIN(ii+block,order); i++,ig_hardened_1++,ig_hardened_2++)
                            AA_arr(i,k) = A_arr(READ_HARDENED_VAR_INT(ig_hardened_1, ig_hardened_2, "ig"),kg);

                    for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)
                        for (ig_hardened_1=ii,ig_hardened_2=ii,i=0; READ_HARDENED_VAR_INT(ig_hardened_1, ig_hardened_2, "ig")<MIN(ii+block,order); i++,ig_hardened_1++,ig_hardened_2++)
                            CC_arr(i,j) = 0.0;

                    for (kg=kk,k=0; kg<MIN(kk+block,order); k++,kg++)
                        for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)
                            for (ig_hardened_1=ii,ig_hardened_2=ii,i=0; READ_HARDENED_VAR_INT(ig_hardened_1, ig_hardened_2, "ig")<MIN(ii+block,order); i++,ig_hardened_1++,ig_hardened_2++)
                                CC_arr(i,j) += AA_arr(i,k)*BB_arr(j,k);

                    for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)
                        for (ig_hardened_1=ii,ig_hardened_2=ii,i=0; READ_HARDENED_VAR_INT(ig_hardened_1, ig_hardened_2, "ig")<MIN(ii+block,order); i++,ig_hardened_1++,ig_hardened_2++)
                            C_arr(READ_HARDENED_VAR_INT(ig_hardened_1, ig_hardened_2, "ig"),jg) += CC_arr(i,j);

                }
            }
        }
        prk_free(AA);
    }

}
