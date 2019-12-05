#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "./main.h"
#include "./kernel/kernel_cpu.h"
#ifdef ERR_INJ
#include <unistd.h>
#endif

#ifdef LOGS
#include "../../include/log_helper.h"
#endif /* LOGS */

#define MAX_ERR_ITER_LOG 500

#ifdef TIMING
long long timing_get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

long long setup_start, setup_end;
long long loop_start, loop_end;
long long kernel_start, kernel_end;
long long check_start, check_end;
#endif

void usage()
{
    printf("Usage: lavamd <# cores> <# boxes 1d> <input_distances> <input_charges> <gold_output> <#iterations>\n");
    printf("  # cores is the number of threads that OpenMP will create\n");
    printf("  # boxes 1d is the input size, 15 is reasonable\n");
}


int iteractions = 100000;

int main( int argc, char *argv [])
{

#ifdef TIMING
    setup_start = timing_get_time();
#endif
    char * input_distance;
    char * input_charges;
    char * output_gold;

    int i, j, k, l, m, n;

    par_str par_cpu;
    dim_str dim_cpu;
    box_str* box_cpu;
    FOUR_VECTOR* rv_cpu;
    fp* qv_cpu;
    FOUR_VECTOR* fv_cpu;
    FOUR_VECTOR* fv_cpu_GOLD;
    int nh;

    dim_cpu.cores_arg = 1;
    dim_cpu.boxes1d_arg = 1;

    if(argc == 7) {
        dim_cpu.cores_arg  = atoi(argv[1]);
        dim_cpu.boxes1d_arg = atoi(argv[2]);
        input_distance = argv[3];
        input_charges = argv[4];
        output_gold = argv[5];
        iteractions = atoi(argv[6]);
    } else {
        usage();
        exit(1);
    }


    printf("Configuration used: cores = %d, boxes1d = %d\n", dim_cpu.cores_arg, dim_cpu.boxes1d_arg);

    par_cpu.alpha = 0.5;

    dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

    dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
    dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
    dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

    dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

    box_cpu = (box_str*)malloc(dim_cpu.box_mem);

    nh = 0;

    for(i=0; i<dim_cpu.boxes1d_arg; i++) {

        for(j=0; j<dim_cpu.boxes1d_arg; j++) {

            for(k=0; k<dim_cpu.boxes1d_arg; k++) {

                box_cpu[nh].x = k;
                box_cpu[nh].y = j;
                box_cpu[nh].z = i;
                box_cpu[nh].number = nh;
                box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

                box_cpu[nh].nn = 0;

                for(l=-1; l<2; l++) {

                    for(m=-1; m<2; m++) {

                        for(n=-1; n<2; n++) {

                            if((((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true) && (l==0 && m==0 && n==0)==false) {

                                box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
                                box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
                                box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
                                box_cpu[nh].nei[box_cpu[nh].nn].number = (box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) + (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + box_cpu[nh].nei[box_cpu[nh].nn].x;
                                box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

                                box_cpu[nh].nn = box_cpu[nh].nn + 1;

                            }
                        }
                    }
                }

                nh = nh + 1;
            }
        }
    }


    srand(time(NULL));

    FILE *file;

    if( (file = fopen(input_distance, "rb" )) == 0 ) {
        printf( "The file 'input_distances' was not opened\n" );
        exit(1);
    }

    rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        fread(&(rv_cpu[i].v), 1, sizeof(double), file);
        fread(&(rv_cpu[i].x), 1, sizeof(double), file);
        fread(&(rv_cpu[i].y), 1, sizeof(double), file);
        fread(&(rv_cpu[i].z), 1, sizeof(double), file);
    }

    fclose(file);

    if( (file = fopen(input_charges, "rb" )) == 0 ) {
        printf( "The file 'input_charges' was not opened\n" );
        exit(1);
    }

    qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        fread(&(qv_cpu[i]), 1, sizeof(double), file);
    }
    fclose(file);

    fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
    fv_cpu_GOLD = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
    if( (file = fopen(output_gold, "rb" )) == 0 ) {
        printf( "The file 'output_forces' was not opened\n" );
        exit(1);
    }
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        fv_cpu[i].v = 0;
        fv_cpu[i].x = 0;
        fv_cpu[i].y = 0;
        fv_cpu[i].z = 0;

        fread(&(fv_cpu_GOLD[i].v), 1, sizeof(double), file);
        fread(&(fv_cpu_GOLD[i].x), 1, sizeof(double), file);
        fread(&(fv_cpu_GOLD[i].y), 1, sizeof(double), file);
        fread(&(fv_cpu_GOLD[i].z), 1, sizeof(double), file);
    }

    fclose(file);

#ifdef LOGS
    char test_info[100];
    snprintf(test_info, 100, "box:%d spaceElem:%ld cores:%d", dim_cpu.boxes1d_arg,dim_cpu.space_elem,dim_cpu.cores_arg);
    start_log_file((char *)"openmpLavaMD", test_info);
    set_max_errors_iter(MAX_ERR_ITER_LOG);
    set_iter_interval_print(5);
#endif /* LOGS */

#ifdef TIMING
    setup_end = timing_get_time();
#endif
    int loop;
    for(loop=0; loop<iteractions; loop++) {
#ifdef TIMING
        loop_start = timing_get_time();
#endif

        for(i=0; i<dim_cpu.space_elem; i=i+1) {
            fv_cpu[i].v = 0;
            fv_cpu[i].x = 0;
            fv_cpu[i].y = 0;
            fv_cpu[i].z = 0;
        }

#ifdef ERR_INJ
        if(loop == 2) {
            printf("injecting error, changing input!\n");
            rv_cpu[0].v = rv_cpu[0].v*2;
            rv_cpu[0].x = rv_cpu[0].x*-1;
            rv_cpu[0].y = rv_cpu[0].y*6;
            rv_cpu[0].z = rv_cpu[0].z*-3;
            qv_cpu[0] = qv_cpu[0]*-2;
        } else if(loop == 3) {
            printf("injecting error, restoring input!\n");
            rv_cpu[0].v = rv_cpu[0].v/2;
            rv_cpu[0].x = rv_cpu[0].x/-1;
            rv_cpu[0].y = rv_cpu[0].y/6;
            rv_cpu[0].z = rv_cpu[0].z/-3;
            qv_cpu[0] = qv_cpu[0]/-2;
        } else if (loop == 4) {
            printf("get ready, infinite loop...\n");
            fflush(stdout);
            while(1) {
                sleep(100);
            }
        }
#endif
#ifdef TIMING
        kernel_start = timing_get_time();
#endif
#ifdef LOGS
        start_iteration();
#endif /* LOGS */
        kernel_cpu(	par_cpu,
                    dim_cpu,
                    box_cpu,
                    rv_cpu,
                    qv_cpu,
                    fv_cpu);

#ifdef LOGS
        end_iteration();
#endif /* LOGS */
#ifdef TIMING
        kernel_end = timing_get_time();
#endif

#ifdef TIMING
        check_start = timing_get_time();
#endif
        int part_error=0;
        #pragma omp parallel for  reduction(+:part_error)
        for(i=0; i<dim_cpu.space_elem; i++) {
            int thread_error=0;
            if ((fabs((fv_cpu[i].v - fv_cpu_GOLD[i].v) / fv_cpu[i].v) > 0.0000000001) || (fabs((fv_cpu[i].v - fv_cpu_GOLD[i].v) / fv_cpu_GOLD[i].v) > 0.0000000001)) {
                //if(fv_cpu_GOLD[i].v != fv_cpu[i].v) {
                thread_error++;
            }
            if ((fabs((fv_cpu[i].x - fv_cpu_GOLD[i].x) / fv_cpu[i].x) > 0.0000000001) || (fabs((fv_cpu[i].x - fv_cpu_GOLD[i].x) / fv_cpu_GOLD[i].x) > 0.0000000001)) {
                //if(fv_cpu_GOLD[i].x != fv_cpu[i].x) {
                thread_error++;
            }
            if ((fabs((fv_cpu[i].y - fv_cpu_GOLD[i].y) / fv_cpu[i].y) > 0.0000000001) || (fabs((fv_cpu[i].y - fv_cpu_GOLD[i].y) / fv_cpu_GOLD[i].y) > 0.0000000001)) {
                //if(fv_cpu_GOLD[i].y != fv_cpu[i].y) {
                thread_error++;
            }
            if ((fabs((fv_cpu[i].z - fv_cpu_GOLD[i].z) / fv_cpu[i].z) > 0.0000000001) || (fabs((fv_cpu[i].z - fv_cpu_GOLD[i].z) / fv_cpu_GOLD[i].z) > 0.0000000001)) {
                //if(fv_cpu_GOLD[i].z != fv_cpu[i].z) {
                thread_error++;
            }
            if (thread_error  > 0) {
                // #pragma omp critical
                {
                    part_error++;
                    char error_detail[300];

                    snprintf(error_detail, 300, "p: [%d], ea: %d, v_r: %1.16e, v_e: %1.16e, x_r: %1.16e, x_e: %1.16e, y_r: %1.16e, y_e: %1.16e, z_r: %1.16e, z_e: %1.16e\n", i, thread_error, fv_cpu[i].v, fv_cpu_GOLD[i].v, fv_cpu[i].x, fv_cpu_GOLD[i].x, fv_cpu[i].y, fv_cpu_GOLD[i].y, fv_cpu[i].z, fv_cpu_GOLD[i].z);
                    printf("error: %s\n",error_detail);
                    thread_error = 0;
                }
            }


        }
        #pragma omp parallel for  reduction(+:part_error)
        for(i=0; i<dim_cpu.space_elem; i++) {
            int thread_error=0;
            if ((fabs((fv_cpu[i].v - fv_cpu_GOLD[i].v) / fv_cpu[i].v) > 0.0000000001) || (fabs((fv_cpu[i].v - fv_cpu_GOLD[i].v) / fv_cpu_GOLD[i].v) > 0.0000000001)) {
                thread_error++;
            }
            if ((fabs((fv_cpu[i].x - fv_cpu_GOLD[i].x) / fv_cpu[i].x) > 0.0000000001) || (fabs((fv_cpu[i].x - fv_cpu_GOLD[i].x) / fv_cpu_GOLD[i].x) > 0.0000000001)) {
                thread_error++;
            }
            if ((fabs((fv_cpu[i].y - fv_cpu_GOLD[i].y) / fv_cpu[i].y) > 0.0000000001) || (fabs((fv_cpu[i].y - fv_cpu_GOLD[i].y) / fv_cpu_GOLD[i].y) > 0.0000000001)) {
                thread_error++;
            }
            if ((fabs((fv_cpu[i].z - fv_cpu_GOLD[i].z) / fv_cpu[i].z) > 0.0000000001) || (fabs((fv_cpu[i].z - fv_cpu_GOLD[i].z) / fv_cpu_GOLD[i].z) > 0.0000000001)) {
                thread_error++;
            }
            if (thread_error  > 0) {
                // #pragma omp critical
                {
                    part_error++;
                    char error_detail[300];

                    snprintf(error_detail, 300, "p: [%d], ea: %d, v_r: %1.16e, v_e: %1.16e, x_r: %1.16e, x_e: %1.16e, y_r: %1.16e, y_e: %1.16e, z_r: %1.16e, z_e: %1.16e\n", i, thread_error, fv_cpu[i].v, fv_cpu_GOLD[i].v, fv_cpu[i].x, fv_cpu_GOLD[i].x, fv_cpu[i].y, fv_cpu_GOLD[i].y, fv_cpu[i].z, fv_cpu_GOLD[i].z);
                    printf("error: %s\n",error_detail);
#ifdef LOGS
                    log_error_detail(error_detail);
#endif
                    thread_error = 0;
                }
            }
        }
        if(loop%5==0||part_error>0) {
            printf("errors:%d\n",part_error);
        }
#ifdef LOGS
        log_error_count(part_error);
#endif /* LOGS */

#ifdef TIMING
        check_end = timing_get_time();
        loop_end = timing_get_time();
        double setup_timing = (double) (setup_end - setup_start) / 1000000;
        double loop_timing = (double) (loop_end - loop_start) / 1000000;
        double kernel_timing = (double) (kernel_end - kernel_start) / 1000000;
        double check_timing = (double) (check_end - check_start) / 1000000;
        printf("\n\tTIMING:\n");
        printf("setup: %f\n",setup_timing);
        printf("loop: %f\n",loop_timing);
        printf("kernel: %f\n",kernel_timing);
        printf("check: %f\n",check_timing);
#endif

    }
#ifdef LOGS
    end_log_file();
#endif /* LOGS */

    free(rv_cpu);
    free(qv_cpu);
    free(fv_cpu);
    free(box_cpu);

    return 0;
}
