#include <omp.h>
#include <time.h>
#include <unistd.h>

#include "md.h"

#define OMP_NUM_THREADS 16

using namespace std;


bool checkGold(double4* force, double4 *gold, int nAtom){
    long int error_count = 0;
    omp_set_num_threads(OMP_NUM_THREADS);
    #pragma omp parallel for reduction(+: error_count)
    for (int i = 0; i < nAtom; i++) {
        double diffx = (force[i].x - gold[i].x) / force[i].x;
        double diffy = (force[i].y - gold[i].y) / force[i].y;
        double diffz = (force[i].z - gold[i].z) / force[i].z;
        double err = sqrt(diffx*diffx) + sqrt(diffy*diffy) + sqrt(diffz*diffz);
        if (err > (3.0 * EPSILON)) {
            error_count++;
        }
    }
    return error_count;
}

void omp_kernel(double4 *force, double4 *position, int *neighList, int nAton_ini, int nAtom) {

    omp_set_num_threads(OMP_NUM_THREADS);
    int i;
    #pragma omp parallel for
    for (i = nAton_ini; i < nAtom; i++) {
        double4 ipos = position[i];
        double4 f = {0.0f, 0.0f, 0.0f};
        int j = 0;
        while (j < maxNeighbors)
        {
            int jidx = neighList[j*nAtom + i];
            double4 jpos = position[jidx];
            // Calculate distance
            double delx = ipos.x - jpos.x;
            double dely = ipos.y - jpos.y;
            double delz = ipos.z - jpos.z;
            double r2inv = delx*delx + dely*dely + delz*delz;

            // If distance is less than cutoff, calculate force
            if (r2inv < cutsq) {

                r2inv = 1.0f/r2inv;
                double r6inv = r2inv * r2inv * r2inv;
                double forceL = r2inv*r6inv*(lj1*r6inv - lj2);

                f.x += delx * forceL;
                f.y += dely * forceL;
                f.z += delz * forceL;
            }
            j++;
        }
        force[i] = f;
    }
}


extern const char *cl_source_md;


int main(int argc, char** argv) {


	// gpu_work_percentage from 0% to 100%.
	// gpu_work_percentage default value is 100%
	int gpu_workload = 0;
	if(argc > 1) {
        gpu_workload = atoi(argv[1]);
    }
    if(gpu_workload > 100 || gpu_workload < 0)
        gpu_workload = 100;
    
    // Problem Parameters
    const int probSizes[6] = { 12288, 24576, 36864, 73728, 147456 , 294912};
    int sizeClass = 6;
    int nAtom = probSizes[sizeClass - 1];

    int ocl_nAtom = nAtom*((float)gpu_workload/100);

	cout << "#Atons = " << nAtom << "\n";
	cout << "#Atons to ocl = " << ocl_nAtom << "\n";
	cout << "#Atons to omp = " << nAtom - ocl_nAtom << "\n";
	
    double4 *position;
    double4 *force, *gold;
    int *neighborList;

    position = (double4*)malloc(sizeof(double4)*nAtom);
    force = (double4*)malloc(sizeof(double4)*nAtom);
    gold = (double4*)malloc(sizeof(double4)*nAtom);
    neighborList = (int*)malloc(sizeof(int) * nAtom * maxNeighbors);

    cout << "Initializing test problem (this can take several "
            "minutes for large problems).\n                   ";

    // Seed random number generator
    srand48(8650341L);

    FILE *fp = fopen("position", "rb" );
    if(fp == NULL){
        printf("Cant open position file\n");
        exit(1);
    }
    FILE *fp_gold = fopen("gold", "rb" );
    if(fp_gold == NULL){
        printf("Cant open position file\n");
        exit(1);
    }
    int return_value, return_value2, return_value3;
    // Initialize positions -- random distribution in cubic domain
    for (int i = 0; i < nAtom; i++)
    {
        return_value = fread(&position[i].x, 1, sizeof(double), fp);
        return_value2 = fread(&position[i].y, 1, sizeof(double), fp);
        return_value3 = fread(&position[i].z, 1, sizeof(double), fp);
        if(return_value == 0 || return_value2 == 0 || return_value3 == 0 ) {
            printf("error reading position files\n");
            exit(1);
        }
        return_value = fread(&gold[i].x, 1, sizeof(double), fp_gold);
        return_value2 = fread(&gold[i].y, 1, sizeof(double), fp_gold);
        return_value3 = fread(&gold[i].z, 1, sizeof(double), fp_gold);
        if(return_value == 0 || return_value2 == 0 || return_value3 == 0 ) {
            printf("error reading gold files\n");
            exit(1);
        }
    }
    fclose(fp);
    fclose(fp_gold);
	
    fp = fopen("neighborList", "rb" );
    if(fp == NULL){
        printf("Cant open neighborList file\n");
        exit(1);
    }
    
    for(int i = 0; i < maxNeighbors * nAtom; i++) {
        return_value = fread(&neighborList[i], 1, sizeof(int), fp);
        if(return_value == 0) {
            printf("error reading position files\n");
            exit(1);
        }
    }
	double total_time = 0;
	double gpu_kernel_time = 0, cpu_kernel_time = 0;
	long long time0, time1, total_time0, total_time1;
	
	total_time0 = get_time();

    omp_set_num_threads(OMP_NUM_THREADS);
	#pragma omp parallel sections
    {
	    #pragma omp section
        {
            if(ocl_nAtom > 0) {
	            initOpenCL();

	            ocl_alloc_buffers(ocl_nAtom, maxNeighbors);
	
                ocl_write_position_buffer(ocl_nAtom, position);
	                
	            ocl_write_neighborList_buffer(maxNeighbors, ocl_nAtom, neighborList);

	            ocl_set_kernel_args(maxNeighbors, ocl_nAtom);
                
                int localSize  = 128;
                int globalSize = ocl_nAtom;
                
                time0 = get_time();
	            ocl_exec_kernel(globalSize, localSize);
                time1 = get_time();
                gpu_kernel_time = (double) (time1-time0) / 1000000;
                
	            ocl_read_force_buffer(ocl_nAtom, force);

                ocl_release_buffers();
                deinitOpenCL();
            }
        }
        
	    #pragma omp section
        {
            if(ocl_nAtom < nAtom){
                time0 = get_time();
                omp_kernel(force, position, neighborList, ocl_nAtom, nAtom);
                time1 = get_time();
                cpu_kernel_time = (double) (time1-time0) / 1000000;
            }
        }
    }
    
    total_time1 = get_time();
    
	total_time = (double) (total_time1-total_time0) / 1000000;
	printf("\ntotal GPU time: %.12f", gpu_kernel_time);
	printf("\ntotal CPU time: %.12f", cpu_kernel_time);
	printf("\ntotal kernels time: %.12f", gpu_kernel_time+cpu_kernel_time);
	printf("\ntotal fft time: %.12f\n", total_time);
	
    long int error_count = checkGold(force, gold, nAtom);

	cout << "Errors: " << error_count << endl;

}


