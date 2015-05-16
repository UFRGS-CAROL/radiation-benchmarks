//============================================================================
//	UPDATE
//============================================================================

//	14 APR 2011 Lukasz G. Szafaryn

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>			 // (in path known to compiler) needed by true/false

//=============================================================================
//	DEFINE / INCLUDE
//=============================================================================
#define INPUT_DISTANCE "./input_distances_192_13"
#define INPUT_CHARGES "./input_charges_192_13"
#define OUTPUT_GOLD "./output_forces_192_13"


#define NUMBER_PAR_PER_BOX 192	 // keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

#define NUMBER_THREADS 192		 // this should be roughly equal to NUMBER_PAR_PER_BOX for best performance

								 // STABLE
#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))

//=============================================================================
//	STRUCTURES
//=============================================================================

typedef struct
{
	double x, y, z;
} THREE_VECTOR;

typedef struct
{
	double v, x, y, z;
} FOUR_VECTOR;

typedef struct nei_str
{
	// neighbor box
	int x, y, z;
	int number;
	long offset;
} nei_str;

typedef struct box_str
{
	// home box
	int x, y, z;
	int number;
	long offset;
	// neighbor boxes
	int nn;
	nei_str nei[26];
} box_str;

typedef struct par_str
{
	double alpha;
} par_str;

typedef struct dim_str
{
	// input arguments
	int cur_arg;
	int arch_arg;
	int cores_arg;
	int boxes1d_arg;
	// system memory
	long number_boxes;
	long box_mem;
	long space_elem;
	long space_mem;
	long space_mem2;
} dim_str;

// Returns the current system time in microseconds
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}


void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		printf("Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
		fflush(NULL);
		exit(EXIT_FAILURE);
	}
}


//-----------------------------------------------------------------------------
//	plasmaKernel_gpu_2
//-----------------------------------------------------------------------------

__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu, box_str* d_box_gpu, FOUR_VECTOR* d_rv_gpu, double* d_qv_gpu, FOUR_VECTOR* d_fv_gpu) {

	//---------------------------------------------------------------------
	//	THREAD PARAMETERS
	//---------------------------------------------------------------------

	int bx = blockIdx.x;		 // get current horizontal block index (0-n)
	int tx = threadIdx.x;		 // get current horizontal thread index (0-n)
	int wtx = tx;

	//---------------------------------------------------------------------
	//	DO FOR THE NUMBER OF BOXES
	//---------------------------------------------------------------------

	if(bx<d_dim_gpu.number_boxes) {

		//-------------------------------------------------------------
		//	Extract input parameters
		//-------------------------------------------------------------

		// parameters
		double a2 = 2.0*d_par_gpu.alpha*d_par_gpu.alpha;

		// home box
		int first_i;
		FOUR_VECTOR* rA;
		FOUR_VECTOR* fA;
		__shared__ FOUR_VECTOR rA_shared[200];

		// nei box
		int pointer;
		int k = 0;
		int first_j;
		FOUR_VECTOR* rB;
		double* qB;
		int j = 0;
		__shared__ FOUR_VECTOR rB_shared[200];
		__shared__ double qB_shared[200];

		// common
		double r2;
		double u2;
		double vij;
		double fs;
		double fxij;
		double fyij;
		double fzij;
		THREE_VECTOR d;

		//-------------------------------------------------------------
		//	Home box
		//-------------------------------------------------------------

		//-------------------------------------------------------------
		//	Setup parameters
		//-------------------------------------------------------------

		// home box - box parameters
		first_i = d_box_gpu[bx].offset;

		// home box - distance, force, charge and type parameters
		rA = &d_rv_gpu[first_i];
		fA = &d_fv_gpu[first_i];

		//-------------------------------------------------------------
		//	Copy to shared memory
		//-------------------------------------------------------------

		// home box - shared memory
		while(wtx<NUMBER_PAR_PER_BOX) {
			rA_shared[wtx] = rA[wtx];
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
		__syncthreads();

		//-------------------------------------------------------------
		//	nei box loop
		//-------------------------------------------------------------

		// loop over neiing boxes of home box
		for (k=0; k<(1+d_box_gpu[bx].nn); k++) {

			//---------------------------------------------
			//	nei box - get pointer to the right box
			//---------------------------------------------

			if(k==0) {
				pointer = bx;	 // set first box to be processed to home box
			}
			else {
								 // remaining boxes are nei boxes
				pointer = d_box_gpu[bx].nei[k-1].number;
			}

			//-----------------------------------------------------
			//	Setup parameters
			//-----------------------------------------------------

			// nei box - box parameters
			first_j = d_box_gpu[pointer].offset;

			// nei box - distance, (force), charge and (type) parameters
			rB = &d_rv_gpu[first_j];
			qB = &d_qv_gpu[first_j];

			//-----------------------------------------------------
			//	Setup parameters
			//-----------------------------------------------------

			// nei box - shared memory
			while(wtx<NUMBER_PAR_PER_BOX) {
				rB_shared[wtx] = rB[wtx];
				qB_shared[wtx] = qB[wtx];
				wtx = wtx + NUMBER_THREADS;
			}
			wtx = tx;

			// synchronize threads because in next section each thread accesses data brought in by different threads here
			__syncthreads();

			//-----------------------------------------------------
			//	Calculation
			//-----------------------------------------------------

			// loop for the number of particles in the home box
			// for (int i=0; i<nTotal_i; i++){
			while(wtx<NUMBER_PAR_PER_BOX) {

				// loop for the number of particles in the current nei box
				for (j=0; j<NUMBER_PAR_PER_BOX; j++) {
					r2 = (double)rA_shared[wtx].v + (double)rB_shared[j].v - DOT((double)rA_shared[wtx],(double)rB_shared[j]);
					u2 = a2*r2;
					vij= exp(-u2);
					fs = 2*vij;

					d.x = (double)rA_shared[wtx].x  - (double)rB_shared[j].x;
					fxij=fs*d.x;
					d.y = (double)rA_shared[wtx].y  - (double)rB_shared[j].y;
					fyij=fs*d.y;
					d.z = (double)rA_shared[wtx].z  - (double)rB_shared[j].z;
					fzij=fs*d.z;

					fA[wtx].v +=  (double)((double)qB_shared[j]*vij);
					fA[wtx].x +=  (double)((double)qB_shared[j]*fxij);
					fA[wtx].y +=  (double)((double)qB_shared[j]*fyij);
					fA[wtx].z +=  (double)((double)qB_shared[j]*fzij);
				}
				// increment work thread index
				wtx = wtx + NUMBER_THREADS;
			}

			// reset work index
			wtx = tx;

			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
			__syncthreads();

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation END
			//----------------------------------------------------------------------------------------------------------------------------------140
		}
		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop END
		//------------------------------------------------------------------------------------------------------------------------------------------------------160
	}
}


//=============================================================================
//	MAIN FUNCTION
//=============================================================================

int main(int argc, char *argv []) {

	//=====================================================================
	//	CPU/MCPU VARIABLES
	//=====================================================================

	// timer
	long long time0;
	long long time1;

	// counters
	int i, j, k, l, m, n;

	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	FOUR_VECTOR* rv_cpu;
	double* qv_cpu;
	FOUR_VECTOR* fv_cpu;
	int nh;

	//=====================================================================
	//	INPUT ARGUMENTS
	//=====================================================================

	dim_cpu.boxes1d_arg = 13;
	printf("Configuration used: boxes1d = %d\n", dim_cpu.boxes1d_arg);

	//=====================================================================
	//	INPUTS
	//=====================================================================

	par_cpu.alpha = 0.5;

	//=====================================================================
	//	DIMENSIONS
	//=====================================================================

	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(double);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

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
							if(     (((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)   &&
							(l==0 && m==0 && n==0)==false   ) {

								// current neighbor box
								box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
								box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
								box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
								box_cpu[nh].nei[box_cpu[nh].nn].number = (box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) +
									(box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + box_cpu[nh].nei[box_cpu[nh].nn].x;
								box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

								// increment neighbor box
								box_cpu[nh].nn = box_cpu[nh].nn + 1;

							}

						}		 // neighbor boxes in x direction
					}			 // neighbor boxes in y direction
				}				 // neighbor boxes in z direction

				// increment home box
				nh = nh + 1;

			}					 // home boxes in x direction
		}						 // home boxes in y direction
	}							 // home boxes in z direction

	//=====================================================================
	//	PARAMETERS, DISTANCE, CHARGE AND FORCE
	//=====================================================================

	// random generator seed set to random value - time in this case
	srand(time(NULL));

	FILE *fp;

	// input (distances)
	if( (fp = fopen(INPUT_DISTANCE, "wb" )) == 0 )
		printf( "The file 'input_distances' was not opened\n" );
	rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
								 // get a number in the range 0.1 - 1.0
		rv_cpu[i].v = (double)(rand()%10 + 1) / 10.0;
		fwrite(&(rv_cpu[i].v), 1, sizeof(double), fp);
								 // get a number in the range 0.1 - 1.0
		rv_cpu[i].x = (double)(rand()%10 + 1) / 10.0;
		fwrite(&(rv_cpu[i].x), 1, sizeof(double), fp);
								 // get a number in the range 0.1 - 1.0
		rv_cpu[i].y = (double)(rand()%10 + 1) / 10.0;
		fwrite(&(rv_cpu[i].y), 1, sizeof(double), fp);
								 // get a number in the range 0.1 - 1.0
		rv_cpu[i].z = (double)(rand()%10 + 1) / 10.0;
		fwrite(&(rv_cpu[i].z), 1, sizeof(double), fp);
	}
	fclose(fp);

	// input (charge)
	if( (fp = fopen(INPUT_CHARGES, "wb" )) == 0 )
		printf( "The file 'input_charges' was not opened\n" );
	qv_cpu = (double*)malloc(dim_cpu.space_mem2);
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
								 // get a number in the range 0.1 - 1.0
		qv_cpu[i] = (double)(rand()%10 + 1) / 10.0;
		fwrite(&(qv_cpu[i]), 1, sizeof(double), fp);
	}
	fclose(fp);

	// output (forces)
	fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
		// set to 0, because kernels keeps adding to initial value
		fv_cpu[i].v = 0;
		fv_cpu[i].x = 0;
		fv_cpu[i].y = 0;
		fv_cpu[i].z = 0;
	}

	//=====================================================================
	//	GPU_CUDA
	//=====================================================================


	//=====================================================================
	//	GPU SETUP
	//=====================================================================

	//=====================================================================
	//	INITIAL DRIVER OVERHEAD
	//=====================================================================

	cudaThreadSynchronize();

	//=====================================================================
	//	VARIABLES
	//=====================================================================

	box_str* d_box_gpu;
	FOUR_VECTOR* d_rv_gpu;
	double* d_qv_gpu;
	FOUR_VECTOR* d_fv_gpu;

	dim3 threads;
	dim3 blocks;

	//=====================================================================
	//	EXECUTION PARAMETERS
	//=====================================================================

	blocks.x = dim_cpu.number_boxes;
	blocks.y = 1;
	threads.x = NUMBER_THREADS;	 // define the number of threads in the block
	threads.y = 1;


	//=====================================================================
	//	GPU MEMORY				(MALLOC)
	//=====================================================================

	//==================================================
	//	boxes
	//==================================================

	cudaMalloc( (void **)&d_box_gpu,
		dim_cpu.box_mem);

	//==================================================
	//	rv
	//==================================================

	cudaMalloc( (void **)&d_rv_gpu,
		dim_cpu.space_mem);

	//==================================================
	//	qv
	//==================================================

	cudaMalloc( (void **)&d_qv_gpu,
		dim_cpu.space_mem2);

	//==================================================
	//	fv
	//==================================================

	cudaMalloc( (void **)&d_fv_gpu,
		dim_cpu.space_mem);


	//=====================================================================
	//	GPU MEMORY			COPY
	//=====================================================================

	//==================================================
	//	boxes
	//==================================================

	cudaMemcpy(d_box_gpu, box_cpu, dim_cpu.box_mem, cudaMemcpyHostToDevice);

	//==================================================
	//	rv
	//==================================================

	cudaMemcpy( d_rv_gpu, rv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);

	//==================================================
	//	qv
	//==================================================

	cudaMemcpy( d_qv_gpu, qv_cpu, dim_cpu.space_mem2, cudaMemcpyHostToDevice);

	//==================================================
	//	fv
	//==================================================

	cudaMemcpy( d_fv_gpu, fv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);

	time0 = get_time();

	//=====================================================================
	//	KERNEL
	//=====================================================================

	// launch kernel - all boxes
	kernel_gpu_cuda<<<blocks, threads>>>( par_cpu, dim_cpu, d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu);

	checkCUDAError("Start");
	cudaThreadSynchronize();

	time1 = get_time();

	//=====================================================================
	//	GPU MEMORY			COPY BACK
	//=====================================================================

	cudaMemcpy( fv_cpu, d_fv_gpu, dim_cpu.space_mem, cudaMemcpyDeviceToHost);

	if( (fp = fopen(OUTPUT_GOLD, "wb" )) == 0 )
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

		fwrite(&(fv_cpu[i].v), 1, sizeof(double), fp);
		fwrite(&(fv_cpu[i].x), 1, sizeof(double), fp);
		fwrite(&(fv_cpu[i].y), 1, sizeof(double), fp);
		fwrite(&(fv_cpu[i].z), 1, sizeof(double), fp);
	}
	fclose(fp);

	printf("Total Number of zeros in the output is %d, from %ld numbers\n",number_zeros, (dim_cpu.space_elem*4));

	//=====================================================================
	//	GPU MEMORY DEALLOCATION
	//=====================================================================

	cudaFree(d_rv_gpu);
	cudaFree(d_qv_gpu);
	cudaFree(d_fv_gpu);
	cudaFree(d_box_gpu);


	//=====================================================================
	//	DISPLAY TIMING
	//=====================================================================


	printf("%.12f s: GPU: KERNEL\n", (double) (time1-time0) / 1000000);
	//=====================================================================
	//	SYSTEM MEMORY DEALLOCATION
	//=====================================================================

	free(rv_cpu);
	free(qv_cpu);
	free(fv_cpu);
	free(box_cpu);

	//=====================================================================
	//	RETURN
	//=====================================================================

	return 0.0;					 // always returns 0.0

}
