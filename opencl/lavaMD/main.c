#ifdef __cplusplus
extern "C" {
#endif


#include <stdio.h>					// (in path known to compiler)			needed by printf
#include <stdlib.h>					// (in path known to compiler)			needed by malloc
#include <stdbool.h>				// (in path known to compiler)			needed by true/false


#include "./util/timer/timer.h"			// (in path specified here)
#include "./util/num/num.h"				// (in path specified here)


#include "./main.h"						// (in the current directory)

#include "./kernel/kernel_gpu_opencl_wrapper.h"	// (in library path specified here)

#define INPUT_DISTANCE "input_distances"
#define INPUT_CHARGES "input_charges"
#define OUTPUT_GOLD "output_forces"

// input size
#define BOXES 15

int
main(	int argc,
        char *argv [])
{

    //====================================================================
    //	CPU/MCPU VARIABLES
    //=====================================================================

    // timer
    long long time0;

    time0 = get_time();

    // timer
    long long time1;
    long long time2;
    long long time3;
    long long time4;
    long long time5;
    long long time6;
    long long time7;

    // counters
    int i, j, k, l, m, n;

    // system memory
    par_str par_cpu;
    dim_str dim_cpu;
    box_str* box_cpu;
    FOUR_VECTOR* rv_cpu;
    fp* qv_cpu;
    FOUR_VECTOR* fv_cpu;
    int nh;


    printf("WG size of kernel = %d \n", NUMBER_THREADS);

    time1 = get_time();

    //=====================================================================
    //	CHECK INPUT ARGUMENTS
    //=====================================================================

    // assing default values
    dim_cpu.arch_arg = 0;
    dim_cpu.cores_arg = 1;
    dim_cpu.boxes1d_arg = BOXES;

    time2 = get_time();

    //=====================================================================
    //	INPUTS
    //=====================================================================

    par_cpu.alpha = 0.5;

    time3 = get_time();

    //=====================================================================
    //	DIMENSIONS
    //=====================================================================

    // total number of boxes
    dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg; // 8*8*8=512

    // how many particles space has in each direction
    dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;							//512*100=51,200
    dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
    dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

    // box array
    dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

    time4 = get_time();

    //=====================================================================
    //	SYSTEM MEMORY
    //=====================================================================

    //=====================================================================
    //	BOX
    //=====================================================================

    // allocate boxes
    box_cpu = (box_str*)malloc(dim_cpu.box_mem);

    // initialize number of home boxes
    nh = 0;

    // home boxes in z direction
    for(i=0; i<dim_cpu.boxes1d_arg; i++) {
        // home boxes in y direction
        for(j=0; j<dim_cpu.boxes1d_arg; j++) {
            // home boxes in x direction
            for(k=0; k<dim_cpu.boxes1d_arg; k++) {

                // current home box
                box_cpu[nh].x = k;
                box_cpu[nh].y = j;
                box_cpu[nh].z = i;
                box_cpu[nh].number = nh;
                box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

                // initialize number of neighbor boxes
                box_cpu[nh].nn = 0;

                // neighbor boxes in z direction
                for(l=-1; l<2; l++) {
                    // neighbor boxes in y direction
                    for(m=-1; m<2; m++) {
                        // neighbor boxes in x direction
                        for(n=-1; n<2; n++) {

                            // check if (this neighbor exists) and (it is not the same as home box)
                            if(		(((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)	&&
                                    (l==0 && m==0 && n==0)==false	) {

                                // current neighbor box
                                box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
                                box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
                                box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
                                box_cpu[nh].nei[box_cpu[nh].nn].number =	(box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) +
                                        (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) +
                                        box_cpu[nh].nei[box_cpu[nh].nn].x;
                                box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

                                // increment neighbor box
                                box_cpu[nh].nn = box_cpu[nh].nn + 1;

                            }

                        } // neighbor boxes in x direction
                    } // neighbor boxes in y direction
                } // neighbor boxes in z direction

                // increment home box
                nh = nh + 1;

            } // home boxes in x direction
        } // home boxes in y direction
    } // home boxes in z direction

    //=====================================================================
    //	PARAMETERS, DISTANCE, CHARGE AND FORCE
    //=====================================================================

    // random generator seed set to random value - time in this case
    srand(time(NULL));

    FILE *file;

    // input (distances)
    if( (file = fopen(INPUT_DISTANCE, "wb" )) == 0 )
        printf( "The file 'input_distances' was not opened\n" );

    // input (distances)
    rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        rv_cpu[i].v = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
        // rv_cpu[i].v = 0.1;			// get a number in the range 0.1 - 1.0
        rv_cpu[i].x = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
        // rv_cpu[i].x = 0.2;			// get a number in the range 0.1 - 1.0
        rv_cpu[i].y = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
        // rv_cpu[i].y = 0.3;			// get a number in the range 0.1 - 1.0
        rv_cpu[i].z = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
        // rv_cpu[i].z = 0.4;			// get a number in the range 0.1 - 1.0
        fwrite(&(rv_cpu[i].v), 1, sizeof(double), file);
        fwrite(&(rv_cpu[i].x), 1, sizeof(double), file);
        fwrite(&(rv_cpu[i].y), 1, sizeof(double), file);
        fwrite(&(rv_cpu[i].z), 1, sizeof(double), file);
    }

    fclose(file);

    // input (charge)
    if( (file = fopen(INPUT_CHARGES, "wb" )) == 0 )
        printf( "The file 'input_charges' was not opened\n" );
    qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        qv_cpu[i] = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
        // qv_cpu[i] = 0.5;			// get a number in the range 0.1 - 1.0
        fwrite(&(qv_cpu[i]), 1, sizeof(double), file);
    }
    fclose(file);

    // output (forces)
    fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        fv_cpu[i].v = 0;								// set to 0, because kernels keeps adding to initial value
        fv_cpu[i].x = 0;								// set to 0, because kernels keeps adding to initial value
        fv_cpu[i].y = 0;								// set to 0, because kernels keeps adding to initial value
        fv_cpu[i].z = 0;								// set to 0, because kernels keeps adding to initial value
    }

    time5 = get_time();

    //=====================================================================
    //	KERNEL
    //=====================================================================

    //=====================================================================
    //	GPU_OPENCL
    //=====================================================================

    kernel_gpu_opencl_wrapper(	par_cpu,
                                dim_cpu,
                                box_cpu,
                                rv_cpu,
                                qv_cpu,
                                fv_cpu);

    time6 = get_time();

    //=====================================================================
    //	SYSTEM MEMORY DEALLOCATION
    //=====================================================================

    // dump results
    if( (file = fopen(OUTPUT_GOLD, "wb" )) == 0 )
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

    time7 = get_time();

    //=====================================================================
    //	DISPLAY TIMING
    //=====================================================================

    // printf("Time spent in different stages of the application:\n");

    // printf("%15.12f s, %15.12f % : VARIABLES\n",						(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time7-time0) * 100);
    // printf("%15.12f s, %15.12f % : INPUT ARGUMENTS\n", 					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time7-time0) * 100);
    // printf("%15.12f s, %15.12f % : INPUTS\n",							(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time7-time0) * 100);
    // printf("%15.12f s, %15.12f % : dim_cpu\n", 							(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time7-time0) * 100);
    // printf("%15.12f s, %15.12f % : SYS MEM: ALO\n",						(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time7-time0) * 100);

    // printf("%15.12f s, %15.12f % : KERNEL: COMPUTE\n",					(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time7-time0) * 100);

    // printf("%15.12f s, %15.12f % : SYS MEM: FRE\n", 					(float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time7-time0) * 100);

    // printf("Total time:\n");
    // printf("%.12f s\n", 												(float) (time7-time0) / 1000000);

    //=====================================================================
    //	RETURN
    //=====================================================================

    return 0.0;																					// always returns 0.0

}

//=============================================================================
//	END
//=============================================================================

#ifdef __cplusplus
}
#endif
