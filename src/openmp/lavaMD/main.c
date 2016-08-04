#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include "./main.h"
#include "./kernel/kernel_cpu.h"

void usage()
{
    printf("Usage: lavamd_gen <# cores> <# boxes 1d>\n");
    printf("  # cores is the number of threads that OpenMP will create\n");
    printf("  # boxes 1d is the input size, 15 is reasonable\n");
}


int main( int argc, char *argv [])
{

    char input_distance[150];
    char input_charges[150];
    char output_gold[150];

    int i, j, k, l, m, n;

    par_str par_cpu;
    dim_str dim_cpu;
    box_str* box_cpu;
    FOUR_VECTOR* rv_cpu;
    fp* qv_cpu;
    FOUR_VECTOR* fv_cpu;
    int nh;

    dim_cpu.cores_arg = 1;
    dim_cpu.boxes1d_arg = 1;

    if(argc == 3) {
        dim_cpu.cores_arg  = atoi(argv[1]);
        dim_cpu.boxes1d_arg = atoi(argv[2]);
    } else {
        usage();
        exit(1);
    }

    printf("Configuration used: cores = %d, boxes1d = %d\n", dim_cpu.cores_arg, dim_cpu.boxes1d_arg);
    snprintf(input_distance, 150, "input_distance_%d_%d",dim_cpu.cores_arg, dim_cpu.boxes1d_arg);
    snprintf(input_charges, 150, "input_charges_%d_%d",dim_cpu.cores_arg, dim_cpu.boxes1d_arg);
    snprintf(output_gold, 150, "output_gold_%d_%d",dim_cpu.cores_arg, dim_cpu.boxes1d_arg);

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

    if( (file = fopen(input_distance, "wb" )) == 0 )
        printf( "The file 'input_distances' was not opened\n" );

    rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        rv_cpu[i].v = (rand()%10 + 1) / 10.0;
        rv_cpu[i].x = (rand()%10 + 1) / 10.0;
        rv_cpu[i].y = (rand()%10 + 1) / 10.0;
        rv_cpu[i].z = (rand()%10 + 1) / 10.0;
        fwrite(&(rv_cpu[i].v), 1, sizeof(double), file);
        fwrite(&(rv_cpu[i].x), 1, sizeof(double), file);
        fwrite(&(rv_cpu[i].y), 1, sizeof(double), file);
        fwrite(&(rv_cpu[i].z), 1, sizeof(double), file);
    }

    fclose(file);

    if( (file = fopen(input_charges, "wb" )) == 0 )
        printf( "The file 'input_charges' was not opened\n" );

    qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        qv_cpu[i] = (rand()%10 + 1) / 10.0;
        fwrite(&(qv_cpu[i]), 1, sizeof(double), file);
    }
    fclose(file);

    fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        fv_cpu[i].v = 0;
        fv_cpu[i].x = 0;
        fv_cpu[i].y = 0;
        fv_cpu[i].z = 0;
    }

    printf("executin kernel\n");
    kernel_cpu(	par_cpu,
                dim_cpu,
                box_cpu,
                rv_cpu,
                qv_cpu,
                fv_cpu);


    printf("saving results\n");
    // dump results
    if( (file = fopen(output_gold, "wb" )) == 0 )
        printf( "The file 'output_forces' was not opened\n" );
    int number_zeros = 0;
    int higher_zero = 0;
    int lower_zero = 0;
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        if(fv_cpu[i].v == 0.0)
            number_zeros++;
        if(fv_cpu[i].v > 0.0)
            higher_zero++;
        if(fv_cpu[i].v < 0.0)
            lower_zero++;

        if(fv_cpu[i].x == 0.0)
            number_zeros++;
        if(fv_cpu[i].x > 0.0)
            higher_zero++;
        if(fv_cpu[i].x < 0.0)
            lower_zero++;

        if(fv_cpu[i].y == 0.0)
            number_zeros++;
        if(fv_cpu[i].y > 0.0)
            higher_zero++;
        if(fv_cpu[i].y < 0.0)
            lower_zero++;

        if(fv_cpu[i].z == 0.0)
            number_zeros++;
        if(fv_cpu[i].z > 0.0)
            higher_zero++;
        if(fv_cpu[i].z < 0.0)
            lower_zero++;

        fwrite(&(fv_cpu[i].v), 1, sizeof(double), file);
        fwrite(&(fv_cpu[i].x), 1, sizeof(double), file);
        fwrite(&(fv_cpu[i].y), 1, sizeof(double), file);
        fwrite(&(fv_cpu[i].z), 1, sizeof(double), file);
    }
    fclose(file);

    printf("Total Number of zeros in the output is %d, from %ld numbers\n",number_zeros, (dim_cpu.space_elem*4));

    free(rv_cpu);
    free(qv_cpu);
    free(fv_cpu);
    free(box_cpu);

    return 0;
}
