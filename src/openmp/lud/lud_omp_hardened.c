#include <stdio.h>
#include <omp.h>

extern int omp_num_threads;

#define BS 16

#define AA(_i,_j) a[offset*size+_i*size+_j+offset]
#define BB(_i,_j) a[_i*size+_j]

#ifdef OMP_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif

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

void lud_diagonal_omp (float* a, int size, int offset)
{
    int i, j, k;
    for (i = 0; i < BS; i++) {

        for (j = i; j < BS; j++) {
            for (k = 0; k < i ; k++) {
                AA(i,j) = AA(i,j) - AA(i,k) * AA(k,j);
            }
        }

        float temp = 1.f/AA(i,i);
        for (j = i+1; j < BS; j++) {
            for (k = 0; k < i ; k++) {
                AA(j,i) = AA(j,i) - AA(j,k) * AA(k,i);
            }
            AA(j,i) = AA(j,i)*temp;
        }
    }

}

#ifdef OMP_OFFLOAD
#pragma offload_attribute(pop)
#endif


// implements block LU factorization
void lud_omp(float *a, int size)
{
    int offset, chunk_idx, size_inter, chunks_in_inter_row, chunks_per_inter;
	int chunk_idx_hardened_1, chunk_idx_hardened_2;

#ifdef OMP_OFFLOAD
    #pragma omp target map(to: size) map(a[0:size*size])
#endif

#ifdef OMP_OFFLOAD
    {
        omp_set_num_threads(224);
#else
    //printf("running OMP on host\n");
    omp_set_num_threads(omp_num_threads);
#endif
        for (offset = 0; offset < size - BS ; offset += BS)
        {
            // lu factorization of left-top corner block diagonal matrix
            //
            lud_diagonal_omp(a, size, offset);

            size_inter = size - offset -  BS;
            chunks_in_inter_row  = size_inter/BS;
	
            // calculate perimeter block matrices
            //
            #pragma omp parallel for default(none) \
            private(chunk_idx, chunk_idx_hardened_1, chunk_idx_hardened_2) shared(size, chunks_per_inter, chunks_in_inter_row, offset, a)
            for ( chunk_idx = 0; chunk_idx < chunks_in_inter_row; chunk_idx++)
            {	
				chunk_idx_hardened_1 = chunk_idx;
				chunk_idx_hardened_2 = chunk_idx;

	        	int i, j, k, i_global, j_global, i_here, j_here;
                float sum_hardened_1, sum_hardened_2;
		
                float temp[BS*BS] __attribute__ ((aligned (64)));

                for (i = 0; i < BS; i++) {
                    #pragma omp simd
                    for (j =0; j < BS; j++) {
                        temp[i*BS + j] = a[size*(i + offset) + offset + j ];
                    }
                }
                i_global = offset;
                j_global = offset;

                // processing top perimeter
                //
    
#ifdef HARDENING_DEBUG
		printf("hardening: original_val = %d, hardened_val_1 = %d, hardened_val_2 = %d, read_val = %d\n", chunk_idx, chunk_idx_hardened_1, chunk_idx_hardened_2, READ_HARDENED_VAR(chunk_idx_hardened_1, chunk_idx_hardened_2, int, sizeof(int)));
#endif

	    	j_global += BS * (READ_HARDENED_VAR(chunk_idx_hardened_1, chunk_idx_hardened_2, int, sizeof(int)) + 1);
                for (j = 0; j < BS; j++) {
                    for (i = 0; i < BS; i++) {
                        sum_hardened_1 = 0.f;
						sum_hardened_2 = 0.f;
                        for (k=0; k < i; k++) {
                            sum_hardened_1 += temp[BS*i +k] * BB((i_global+k),(j_global+j));
				 			sum_hardened_2 += temp[BS*i +k] * BB((i_global+k),(j_global+j));
                        }
                        i_here = i_global + i;
                        j_here = j_global + j;
                        BB(i_here, j_here) = BB(i_here,j_here) - READ_HARDENED_VAR(sum_hardened_1, sum_hardened_2, float, sizeof(float));
                    }
                }

                // processing left perimeter
                //
                j_global = offset;
                i_global += BS * (READ_HARDENED_VAR(chunk_idx_hardened_1, chunk_idx_hardened_2, int, sizeof(int)) + 1);
                for (i = 0; i < BS; i++) {
                    for (j = 0; j < BS; j++) {
                        sum_hardened_1 = 0.f;
			sum_hardened_2 = 0.f;
                        for (k=0; k < j; k++) {
                            sum_hardened_1 += BB((i_global+i),(j_global+k)) * temp[BS*k + j];
                            sum_hardened_2 += BB((i_global+i),(j_global+k)) * temp[BS*k + j];
                        }
                        i_here = i_global + i;
                        j_here = j_global + j;
                        a[size*i_here + j_here] = ( a[size*i_here+j_here] - READ_HARDENED_VAR(sum_hardened_1, sum_hardened_2, float, sizeof(float))) / a[size*(offset+j) + offset+j];
                    }
                }

		chunk_idx_hardened_1++;
		chunk_idx_hardened_2++;
            }

            // update interior block matrices
            //
            chunks_per_inter = chunks_in_inter_row*chunks_in_inter_row;
	
            #pragma omp parallel for schedule(auto) default(none) \
            private(chunk_idx, chunk_idx_hardened_1, chunk_idx_hardened_2) shared(size, chunks_per_inter, chunks_in_inter_row, offset, a)
            for  (chunk_idx = 0; chunk_idx < chunks_per_inter; chunk_idx++)
            {
        		chunk_idx_hardened_1 = chunk_idx;
				chunk_idx_hardened_2 = chunk_idx;

                int i, j, k, i_global, j_global;
                float temp_top_hardened_1[BS*BS] __attribute__ ((aligned (64)));
		float temp_top_hardened_2[BS*BS] __attribute__ ((aligned (64)));
                float temp_left[BS*BS] __attribute__ ((aligned (64)));
                float sum_hardened_1[BS] __attribute__ ((aligned (64))) = {0.f};
		float sum_hardened_2[BS] __attribute__ ((aligned (64))) = {0.f};

#ifdef HARDENING_DEBUG
		printf("hardening: original_val = %d, hardened_val_1 = %d, hardened_val_2 = %d, read_val = %d\n", chunk_idx, chunk_idx_hardened_1, chunk_idx_hardened_2, READ_HARDENED_VAR(chunk_idx_hardened_1, chunk_idx_hardened_2, int, sizeof(int)));
#endif

                i_global = offset + BS * (1 + READ_HARDENED_VAR(chunk_idx_hardened_1, chunk_idx_hardened_2, int, sizeof(int))/chunks_in_inter_row);
                j_global = offset + BS * (1 + READ_HARDENED_VAR(chunk_idx_hardened_1, chunk_idx_hardened_2, int, sizeof(int))%chunks_in_inter_row);

                for (i = 0; i < BS; i++) {
                    #pragma omp simd
                    for (j =0; j < BS; j++) {
                        temp_top_hardened_1[i*BS + j]  = a[size*(i + offset) + j + j_global ];
			temp_top_hardened_2[i*BS + j]  = a[size*(i + offset) + j + j_global ];
                        temp_left[i*BS + j] = a[size*(i + i_global) + offset + j];
                    }
                }

                for (i = 0; i < BS; i++)
                {
                    for (k=0; k < BS; k++) {
                        #pragma omp simd
                        for (j = 0; j < BS; j++) {
        	            sum_hardened_1[j] += temp_left[BS*i + k] * READ_HARDENED_ARRAY(temp_top_hardened_1, temp_top_hardened_2, float*, BS*BS*sizeof(float))[BS*k + j];
			    sum_hardened_2[j] += temp_left[BS*i + k] * READ_HARDENED_ARRAY(temp_top_hardened_1, temp_top_hardened_2, float*, BS*BS*sizeof(float))[BS*k + j];
                        }
                    }
                    #pragma omp simd	
                    for (j = 0; j < BS; j++) {
#ifdef HARDENING_DEBUG
                    	printf("hardening: array_ptr_1 = %p, array_ptr_2 = %p, size_array = %d, read_array_ptr = %p\n", sum_hardened_1, sum_hardened_2, BS*sizeof(float), READ_HARDENED_ARRAY(sum_hardened_1, sum_hardened_2, float*, BS*sizeof(float)));
#endif
			BB((i+i_global),(j+j_global)) -= READ_HARDENED_ARRAY(sum_hardened_1, sum_hardened_2, float*, BS*sizeof(float))[j];
                        sum_hardened_1[j] = 0.f;
			sum_hardened_2[j] = 0.f;
                    }
                }

				chunk_idx_hardened_1++;
				chunk_idx_hardened_2++;
            }
        }

        lud_diagonal_omp(a, size, offset);
#ifdef OMP_OFFLOAD
    }
#endif

}
