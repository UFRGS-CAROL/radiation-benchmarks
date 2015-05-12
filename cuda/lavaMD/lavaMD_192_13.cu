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
#define INPUT_DISTANCE "/home/carol/TestGPU/lavaMD/input_distances_192_13"
#define INPUT_CHARGES "/home/carol/TestGPU/lavaMD/input_charges_192_13"
#define OUTPUT_GOLD "/home/carol/TestGPU/lavaMD/output_forces_192_13"

#define ITERACTIONS 10000		 //first loop, killed when there is a cuda malloc error, cuda thread sync error, too many output error

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

int last_part_errors = 0;

// Returns the current system time in microseconds
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}


void UpdateTimestamp() {
	time_t timestamp = time(NULL);
	char time_s[50];
	sprintf(time_s, "%d", int(timestamp));

	char string[100] = "echo ";
	strcat(string, time_s);
	strcat(string, " > /home/carol/TestGPU/timestamp.txt");
	system(string);
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
	int t_ea = 0;
	int old_ea = 0;

	double total_kernel_time = 0;
	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	FOUR_VECTOR* rv_cpu;
	double* qv_cpu;
	FOUR_VECTOR* fv_cpu;
	FOUR_VECTOR* fv_cpu_GOLD;
	int nh;

	FILE *log_file;
	FILE *fp;
	FILE *file;

	cudaError_t cuda_error;
	const char *error_string;

	///////////////////////////////////////////////////////
	////////////////FILE NAME//////////////////////////////
	///////////////////////////////////////////////////////
	// infos : dd_mm_yyyy_hh_mm_ss_LavaMD_192_13

	// to be modified if you have better ideas

	time_t file_time;
	struct tm *ptm;
	char day[2], month[2], year[4], hour[2], second[2], minute[2];
	char file_name[60];
	char file_name_log[60];

	file_time = time(NULL);
	ptm = gmtime(&file_time);

	snprintf(day, sizeof(day + 1), "%d", ptm->tm_mday);
	snprintf(month, sizeof(month + 1), "%d", ptm->tm_mon+1);
	snprintf(year, sizeof(year + 1), "%d", ptm->tm_year+1900);
	snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
	snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
	snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);
	strcpy(file_name,day);strcat(file_name,"_");
	strcat(file_name,month);strcat(file_name,"_");
	strcat(file_name,year);strcat(file_name,"_");
	strcat(file_name,hour);strcat(file_name,"_");
	strcat(file_name,minute);strcat(file_name,"_");
	strcat(file_name,second);strcat(file_name,"_");
	strcat(file_name,"_LavaMD_192_13");
	strcpy(file_name_log, file_name);

	strcat(file_name,".txt");
	strcat(file_name_log,"_log.txt");

	//LOOP START
	int loop;
	for(loop=0; loop<ITERACTIONS; loop++) {

		file = fopen(file_name, "a");
		if(file == NULL) {
			printf("error to open file '%s'\n",file_name);
			break;
		}

		//=====================================================================
		//	CHECK INPUT ARGUMENTS
		//=====================================================================
		// assing default values
		dim_cpu.boxes1d_arg = 13;
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
		if(box_cpu == NULL) {
			printf("error box_cpu malloc\n");
			fprintf(file, "error box_cpu malloc\n");
			break;
		}
		fclose(file);

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

							}	 // neighbor boxes in x direction
						}		 // neighbor boxes in y direction
					}			 // neighbor boxes in z direction

					// increment home box
					nh = nh + 1;

				}				 // home boxes in x direction
			}					 // home boxes in y direction
		}						 // home boxes in z direction

		//=====================================================================
		//	PARAMETERS, DISTANCE, CHARGE AND FORCE
		//=====================================================================

		file = fopen(file_name, "a");
		if(file == NULL) {
			printf("error to open file '%s'\n",file_name);
			break;
		}

		size_t return_value[4];
		// input (distances)
		if( (fp = fopen(INPUT_DISTANCE, "rb" )) == 0 )
			printf( "The file 'input_distances' was not opened\n" );
		rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
		if(rv_cpu == NULL) {
			printf("error rv_cpu malloc\n");
			fprintf(file, "error rv_cpu malloc\n");
			break;
		}
		for(i=0; i<dim_cpu.space_elem; i=i+1) {
			return_value[0] = fread(&(rv_cpu[i].v), 1, sizeof(double), fp);
			return_value[1] = fread(&(rv_cpu[i].x), 1, sizeof(double), fp);
			return_value[2] = fread(&(rv_cpu[i].y), 1, sizeof(double), fp);
			return_value[3] = fread(&(rv_cpu[i].z), 1, sizeof(double), fp);
			if (return_value[0] == 0 || return_value[1] == 0 || return_value[2] == 0 || return_value[3] == 0) {
				printf("error reading rv_cpu from file\n");
				fprintf(file, "error reading rv_cpu from file\n");
				break;
			}
		}
		fclose(fp);

		// input (charge)
		if( (fp = fopen(INPUT_CHARGES, "rb" )) == 0 )
			printf( "The file 'input_charges' was not opened\n" );
		qv_cpu = (double*)malloc(dim_cpu.space_mem2);
		if(qv_cpu == NULL) {
			printf("error qv_cpu malloc\n");
			fprintf(file, "error qv_cpu malloc\n");
			break;
		}
		for(i=0; i<dim_cpu.space_elem; i=i+1) {
			return_value[0] = fread(&(qv_cpu[i]), 1, sizeof(double), fp);
			if (return_value[0] == 0) {
				printf("error reading qv_cpu from file\n");
				fprintf(file, "error reading qv_cpu from file\n");
				break;
			}
		}
		fclose(fp);

		// GOLD output (forces)
		if( (fp = fopen(OUTPUT_GOLD, "rb" )) == 0 )
			printf( "The file 'output_forces' was not opened\n" );
		fv_cpu_GOLD = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
		if(fv_cpu_GOLD == NULL) {
			printf("error fv_cpu_GOLD malloc\n");
			fprintf(file, "error fv_cpu_GOLD malloc\n");
			break;
		}
		for(i=0; i<dim_cpu.space_elem; i=i+1) {
			return_value[0] = fread(&(fv_cpu_GOLD[i].v), 1, sizeof(double), fp);
			return_value[1] = fread(&(fv_cpu_GOLD[i].x), 1, sizeof(double), fp);
			return_value[2] = fread(&(fv_cpu_GOLD[i].y), 1, sizeof(double), fp);
			return_value[3] = fread(&(fv_cpu_GOLD[i].z), 1, sizeof(double), fp);
			if (return_value[0] == 0 || return_value[1] == 0 || return_value[2] == 0 || return_value[3] == 0) {
				printf("error reading rv_cpu from file\n");
				fprintf(file, "error reading rv_cpu from file\n");
				break;
			}
		}
		fclose(fp);

		// output (forces)
		fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
		if(fv_cpu == NULL) {
			printf("error fv_cpu malloc\n");
			fprintf(file, "error fv_cpu malloc\n");
			break;
		}
		for(i=0; i<dim_cpu.space_elem; i=i+1) {
			// set to 0, because kernels keeps adding to initial value
			fv_cpu[i].v = 0;
			fv_cpu[i].x = 0;
			fv_cpu[i].y = 0;
			fv_cpu[i].z = 0;
		}

		fclose(file);

		//=====================================================================
		//	GPU_CUDA
		//=====================================================================

		//=====================================================================
		//	GPU SETUP
		//=====================================================================

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
								 // define the number of threads in the block
		threads.x = NUMBER_THREADS;
		threads.y = 1;

		//=====================================================================
		//	GPU MEMORY				(MALLOC)
		//=====================================================================

		//==================================================
		//	boxes
		//==================================================

		file = fopen(file_name, "a");
		if(file == NULL) {
			printf("error to open file '%s'\n",file_name);
			break;
		}

		cuda_error = cudaMalloc( (void **)&d_box_gpu, dim_cpu.box_mem);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error d_box_gpu cudaMalloc\n");
			fprintf(file, "error d_box_gpu cudaMalloc\n"); break;
		}
		//==================================================
		//	rv
		//==================================================

		cuda_error = cudaMalloc( (void **)&d_rv_gpu, dim_cpu.space_mem);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error d_rv_gpu cudaMalloc\n");
			fprintf(file, "error d_box_gpu cudaMalloc\n"); break;
		}
		//==================================================
		//	qv
		//==================================================

		cuda_error = cudaMalloc( (void **)&d_qv_gpu, dim_cpu.space_mem2);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error d_qv_gpu cudaMalloc\n");
			fprintf(file, "error d_box_gpu cudaMalloc\n"); break;
		}
		//==================================================
		//	fv
		//==================================================

		cuda_error = cudaMalloc( (void **)&d_fv_gpu, dim_cpu.space_mem);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error d_fv_gpu cudaMalloc\n");
			fprintf(file, "error d_box_gpu cudaMalloc\n"); break;
		}

		//=====================================================================
		//	GPU MEMORY			COPY
		//=====================================================================

		//==================================================
		//	boxes
		//==================================================

		cuda_error = cudaMemcpy(d_box_gpu, box_cpu, dim_cpu.box_mem, cudaMemcpyHostToDevice);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error load d_boc_gpu\n");
			fprintf(file, "error load d_box_gpu\n"); break;
		}
		//==================================================
		//	rv
		//==================================================

		cuda_error = cudaMemcpy( d_rv_gpu, rv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error load d_rv_gpu\n");
			fprintf(file, "error load d_box_gpu\n"); break;
		}
		//==================================================
		//	qv
		//==================================================

		cuda_error = cudaMemcpy( d_qv_gpu, qv_cpu, dim_cpu.space_mem2, cudaMemcpyHostToDevice);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error load d_qv_gpu\n");
			fprintf(file, "error load d_box_gpu\n"); break;
		}
		//==================================================
		//	fv
		//==================================================

		cuda_error = cudaMemcpy( d_fv_gpu, fv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error load d_fv_gpu\n");
			fprintf(file, "error load d_box_gpu\n"); break;
		}

		//=====================================================================
		//	KERNEL
		//=====================================================================

		time0 = get_time();

		// launch kernel - all boxes
		kernel_gpu_cuda<<<blocks, threads>>>( par_cpu, dim_cpu, d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu);

		cuda_error = cudaThreadSynchronize();
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error logic: %s\n",error_string);
			fprintf(file, "error logic: %s\n",error_string); break;
		}

		time1 = get_time();

		//=====================================================================
		//	GPU MEMORY			COPY BACK
		//=====================================================================

		cuda_error = cudaMemcpy( fv_cpu, d_fv_gpu, dim_cpu.space_mem, cudaMemcpyDeviceToHost);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error download fv_cpu\n");
			fprintf(file, "error download fv_cpu\n"); break;
		}

		//=====================================================================
		//	GPU MEMORY DEALLOCATION
		//=====================================================================
		fclose(file);

		cudaFree(d_rv_gpu);
		cudaFree(d_qv_gpu);
		cudaFree(d_fv_gpu);
		cudaFree(d_box_gpu);
		//=====================================================================
		//	COMPARE OUTPUTS
		//=====================================================================

		///////////UPDATE FILE//////////////////////

		file = fopen(file_name, "a");
		if(file == NULL) {
			printf("error to open file '%s'\n",file_name);
			break;
		}

		file_time = time(NULL);
		ptm = gmtime(&file_time);
		snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
		snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
		snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);
								 //write the starting time
		fprintf(file, "\n start time: %s/%s_%s:%s:%s", day,month,hour,minute,second);
		fclose(file);

		//int ea = 0;
		int part_error = 0;
		int thread_error = 0;

		file = fopen(file_name, "a");
		if(file == NULL) {
			printf("error to open file '%s'\n",file_name);
			break;
		}

		for(i=0; i<dim_cpu.space_elem; i=i+1) {
			if(fv_cpu_GOLD[i].v != fv_cpu[i].v) {
				thread_error++;
				/*ea++;
				t_ea++;
				fprintf(file, "\n fv.v position: [%d], read: %1.16e, expected: %1.16e, error: %d\n", i, fv_cpu[i].v, fv_cpu_GOLD[i].v, t_ea);*/
			}
			if(fv_cpu_GOLD[i].x != fv_cpu[i].x) {
				thread_error++;
				/*ea++;
				t_ea++;
				fprintf(file, "\n fv.x position: [%d], read: %1.16e, expected: %1.16e, error: %d\n", i, fv_cpu[i].x, fv_cpu_GOLD[i].x, t_ea);*/
			}
			if(fv_cpu_GOLD[i].y != fv_cpu[i].y) {
				thread_error++;
				/*ea++;
				t_ea++;
				fprintf(file, "\n fv.y position: [%d], read: %1.16e, expected: %1.16e, error: %d\n", i, fv_cpu[i].y, fv_cpu_GOLD[i].y, t_ea);*/
			}
			if(fv_cpu_GOLD[i].z != fv_cpu[i].z) {
				thread_error++;
				/*ea++;
				t_ea++;
				fprintf(file, "\n fv.z position: [%d], read: %1.16e, expected: %1.16e, error: %d\n", i, fv_cpu[i].z, fv_cpu_GOLD[i].z, t_ea);*/
			}
			if (thread_error  > 0) {
				t_ea++;
				part_error++;
				fprintf(file, "\np: [%d], ea: %d, v_r: %1.16e, v_e: %1.16e, x_r: %1.16e, x_e: %1.16e, y_r: %1.16e, y_e: %1.16e, z_r: %1.16e, z_e: %1.16e, error: %d\n", i, thread_error, fv_cpu[i].v, fv_cpu_GOLD[i].v, fv_cpu[i].x, fv_cpu_GOLD[i].x, fv_cpu[i].y, fv_cpu_GOLD[i].y, fv_cpu[i].z, fv_cpu_GOLD[i].z, t_ea);
				thread_error = 0;
			}
		}

		fclose(file);

		///////////UPDATE LOG FILE//////////////////////
		/// PROBABLY NOT NEEDED, OR CAN BE REDUCED
		double kernel_time = (double) (time1-time0) / 1000000;
		total_kernel_time += kernel_time;

		log_file = fopen(file_name_log, "a");
		if(log_file == NULL) {
			printf("error to open file '%s'\n",file_name_log);
			fprintf(file,"error to open file '%s'\n",file_name_log);
			break;
		}
		fprintf(log_file, "\ntest number: %d", loop);
		fclose(log_file);

		/////////////UPDATE TIMESTAMP///////////////////
		UpdateTimestamp();
		////////////////////////////////////////////////

		if(part_error > 0 || (loop % 10 == 0)) {

			log_file = fopen(file_name_log, "a");
			if(log_file == NULL) {
				printf("error to open file '%s'\n",file_name_log);
				fprintf(file,"error to open file '%s'\n",file_name_log);
				break;
			}

			fprintf(log_file, "\ntest number: %d", loop);
			fprintf(log_file, "\nkernel time: %.12f", kernel_time);
			fprintf(log_file, "\naccumulated kernel time: %.12f", total_kernel_time);
			fprintf(log_file, "\nparticles with errors: %d", part_error);
			fprintf(log_file, "\ntotal errors: %d", t_ea);

			//fclose(file);
			fclose(log_file);

			printf("\ntest number: %d", loop);
			printf("\naccumulated kernel time: %f", total_kernel_time);
			printf("\nparticles with errors: %d", part_error);
			printf("\ntotal errors: %d", t_ea);

								 //we NEED this, beause at times the GPU get stuck and it gives a huge amount of error, we cannot let it write a stream of data on the HDD
			if(part_error > 500) break;
			if(part_error > 0 && part_error == old_ea) {
				old_ea = 0;
				break;
			}

			old_ea = part_error;

			if(part_error > 0 && part_error == last_part_errors){
				exit(1);
			}
			last_part_errors = part_error;

		}
		else {
			printf(".");
		}
		//=====================================================================
		//	SYSTEM MEMORY DEALLOCATION
		//=====================================================================

		free(fv_cpu_GOLD);
		free(rv_cpu);
		free(qv_cpu);
		free(fv_cpu);
		free(box_cpu);

	}
	printf("\n");

	return 0;
}
