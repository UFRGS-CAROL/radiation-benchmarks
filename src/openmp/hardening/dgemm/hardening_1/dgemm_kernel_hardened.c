#include "dgemm.h"
#include "../../selective_hardening/hardening.h"

void dgemm(double *A, double *B, double *C, long order, int block) {

    int     i,ii,j,jj,k,kk,ig,jg;
    int kg_hardened_1, kg_hardened_2;

    #pragma omp parallel private (i,j,k,ii,jj,kk,ig,jg,kg_hardened_1, kg_hardened_2)
    {
        double *AA=NULL, *BB=NULL, *CC=NULL;

        /* matrix blocks for local temporary copies*/
        AA = (double *) prk_malloc(block*(block+BOFFSET)*3*sizeof(double));
        if (!AA) {
            /*printf("Could not allocate space for matrix tiles on thread %d\n",
                   omp_get_thread_num());*/
            exit(1);
        }
        BB = AA + block*(block+BOFFSET);
        CC = BB + block*(block+BOFFSET);


        #pragma omp for
        for(jj = 0; jj < order; jj+=block) {
            for(kk = 0; kk < order; kk+=block) {

                for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)
                    for (kg_hardened_1=kk,kg_hardened_2=kk,k=0; READ_HARDENED_VAR_INT(kg_hardened_1, kg_hardened_2, "kg")<MIN(kk+block,order); k++,kg_hardened_1++,kg_hardened_2++)
                        BB_arr(j,k) =  B_arr(READ_HARDENED_VAR_INT(kg_hardened_1, kg_hardened_2, "kg"),jg);

                for(ii = 0; ii < order; ii+=block) {

                    for (kg_hardened_1=kk,kg_hardened_2=kk,k=0; READ_HARDENED_VAR_INT(kg_hardened_1, kg_hardened_2, "kg")<MIN(kk+block,order); k++,kg_hardened_1++,kg_hardened_2++)
                        for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
                            AA_arr(i,k) = A_arr(ig,READ_HARDENED_VAR_INT(kg_hardened_1, kg_hardened_2, "kg"));

                    for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)
                        for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
                            CC_arr(i,j) = 0.0;

                    for (kg_hardened_1=kk,kg_hardened_2=kk,k=0; READ_HARDENED_VAR_INT(kg_hardened_1, kg_hardened_2, "kg")<MIN(kk+block,order); k++,kg_hardened_1++,kg_hardened_2++)
                        for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)
                            for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
                                CC_arr(i,j) += AA_arr(i,k)*BB_arr(j,k);

                    for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)
                        for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
                            C_arr(ig,jg) += CC_arr(i,j);

                }
            }
        }
        prk_free(AA);
    }

}
