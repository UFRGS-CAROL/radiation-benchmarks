#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>

#ifdef LOGS
#include "log_helper.h"
#endif

#define INPUT_TEMP "./input_temp_1024"
#define INPUT_POWER "./input_power_1024"
#define OUTPUT_GOLD "./output_1024"


#define BLOCK_SIZE_X 14
#define BLOCK_SIZE_Y 14

#ifndef ITERACTIONS
#define ITERACTIONS 100000000000000000
#endif //first loop, killed when there is a cuda malloc error, cuda thread sync error, too many output error

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD  (3.0e6)
/* required precision in degrees	*/
#define PRECISION   0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

int last_num_errors = 0;

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


#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

								 //number of iteration
__global__ void calculate_temp(int iteration,
float *power,					 //power input
float *temp_src,				 //temperature input/output
float *temp_dst,				 //temperature input/output
int grid_cols,					 //Col of grid
int grid_rows,					 //Row of grid
int border_cols,				 // border offset
int border_rows,				 // border offset
float Cap,						 //Capacitance
float Rx,
float Ry,
float Rz,
float step,
float time_elapsed) {

	__shared__ float temp_on_cuda[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	__shared__ float power_on_cuda[BLOCK_SIZE_Y][BLOCK_SIZE_X];
								 // saving temparary temperature result
	__shared__ float temp_t[BLOCK_SIZE_Y][BLOCK_SIZE_X];

	float amb_temp = 80.0;
	float step_div_Cap;
	float Rx_1,Ry_1,Rz_1;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx=threadIdx.x;
	int ty=threadIdx.y;

	step_div_Cap=step/Cap;

	Rx_1=1/Rx;
	Ry_1=1/Ry;
	Rz_1=1/Rz;

	// each block finally computes result for a small block
	// after N iterations.
	// it is the non-overlapping small blocks that cover
	// all the input data

	// calculate the small block size
								 //EXPAND_RATE
	int small_block_rows = BLOCK_SIZE_Y-iteration*2;
								 //EXPAND_RATE
	int small_block_cols = BLOCK_SIZE_X-iteration*2;

	// calculate the boundary for the block according to
	// the boundary of its small block
	int blkY = small_block_rows*by-border_rows;
	int blkX = small_block_cols*bx-border_cols;
	int blkYmax = blkY+BLOCK_SIZE_Y-1;
	int blkXmax = blkX+BLOCK_SIZE_X-1;

	// calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;

	// load data if it is within the valid input range
	int loadYidx=yidx, loadXidx=xidx;
	int index = grid_cols*loadYidx+loadXidx;

	if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)) {
								 // Load the temperature data from global memory to shared memory
		temp_on_cuda[ty][tx] = temp_src[index];
								 // Load the power data from global memory to shared memory
		power_on_cuda[ty][tx] = power[index];
	}
	__syncthreads();

	// effective range within this block that falls within
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validYmin = (blkY < 0) ? -blkY : 0;
	int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE_Y-1-(blkYmax-grid_rows+1) : BLOCK_SIZE_Y-1;
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE_X-1-(blkXmax-grid_cols+1) : BLOCK_SIZE_X-1;

	int N = ty-1;
	int S = ty+1;
	int W = tx-1;
	int E = tx+1;

	N = (N < validYmin) ? validYmin : N;
	S = (S > validYmax) ? validYmax : S;
	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool computed;
	for (int i=0; i<iteration ; i++) {
		computed = false;
		if( IN_RANGE(tx, i+1, BLOCK_SIZE_X-i-2) && \
			IN_RANGE(ty, i+1, BLOCK_SIZE_Y-i-2) && \
			IN_RANGE(tx, validXmin, validXmax) && \
			IN_RANGE(ty, validYmin, validYmax) ) \
		{ \
			computed = true;
			temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
				(temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 +
				(temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 +
				(amb_temp - temp_on_cuda[ty][tx]) * Rz_1);

		}
		__syncthreads();
		if(i==iteration-1)
			break;
		if(computed)			 //Assign the computation range
			temp_on_cuda[ty][tx]= temp_t[ty][tx];
		__syncthreads();
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the
	// small block perform the calculation and switch on ``computed''
	if (computed) {
		temp_dst[index]= temp_t[ty][tx];
	}
}


/*
   compute N time steps
*/

int compute_tran_temp(float *MatrixPower,float *MatrixTemp[2], int col, int row, \
int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows) {
	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid(blockCols, blockRows);

	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	float t;
	float time_elapsed;
	time_elapsed=0.001;

	int src = 1, dst = 0;

	for (t = 0; t < total_iterations; t+=num_iterations) {
		int temp = src;
		src = dst;
		dst = temp;
		calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations-t), MatrixPower,MatrixTemp[src],MatrixTemp[dst],\
			col,row,borderCols, borderRows, Cap,Rx,Ry,Rz,step,time_elapsed);

		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess) {
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error));
			exit(-1);
		}
	}
	return dst;
}


int main(int argc, char** argv) {

	int size;
	int grid_rows,grid_cols;
	float *inputTemp,*inputPower,*MatrixOut,*output_GOLD;
	int pyramid_height, total_iterations;
	long long time0, time1;
	int t_ea;

	grid_rows = 1024;
	grid_cols = grid_rows;
	pyramid_height = 1;
	total_iterations = 20000;

	size=grid_rows*grid_cols;

#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "size:%d", grid_rows);
	start_log_file("cudaHotspot", test_info);
#endif

	//LOOP START
	int loop;

	for(loop=0; loop<ITERACTIONS; loop++) {

		/* --------------- pyramid parameters --------------- */
		# define EXPAND_RATE 2	 // add one iteration will extend the pyramid base by 2 per each borderline
		int borderCols = (pyramid_height)*EXPAND_RATE/2;
		int borderRows = (pyramid_height)*EXPAND_RATE/2;
		int smallBlockCol = BLOCK_SIZE_X-(pyramid_height)*EXPAND_RATE;
		int smallBlockRow = BLOCK_SIZE_Y-(pyramid_height)*EXPAND_RATE;
		int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
		int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

		inputTemp = (float *) malloc(size*sizeof(float));
		inputPower = (float *) malloc(size*sizeof(float));
		output_GOLD = (float *) malloc(size*sizeof(float));
		MatrixOut = (float *) calloc (size, sizeof(float));

		if( !inputPower || !inputTemp || !output_GOLD || !MatrixOut) {
			printf("error unable to allocate CPU memory\n");
#ifdef LOGS
			log_error_detail("error : unable to allocate CPU memory"); end_log_file(); 
#endif
			return 0;
		}

		/*printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",\ 
		pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);*/

		FILE *fp;
		int i, j, return_value;

		if( (fp = fopen(INPUT_TEMP, "rb" )) == 0 ) {
#ifdef LOGS
			log_error_detail("error the file 'input_temp_1024' was not opened"); end_log_file(); 
#endif
			return 0;
		}
		for (i=0; i <= grid_rows-1; i++)
		for (j=0; j <= grid_cols-1; j++) {
			return_value = fread(&(inputTemp[i*grid_cols+j]), 1, sizeof(float), fp);
			if(return_value == 0) {
				printf("error reading input_temp\n");
#ifdef LOGS
				log_error_detail("error reading the input file"); end_log_file(); 
#endif
				return 0;
			}
		}
		fclose(fp);

		if( (fp = fopen(INPUT_POWER, "rb" )) == 0 ) {
			printf("error the file 'input_power_1024' was not opened\n");
#ifdef LOGS
			log_error_detail("error the file input_power_1024 was not opened"); end_log_file(); 
#endif
			return 0;
		}
		for (i=0; i <= grid_rows-1; i++)
		for (j=0; j <= grid_cols-1; j++) {
			return_value = fread(&(inputPower[i*grid_cols+j]), 1, sizeof(float), fp);
			if(return_value == 0) {
				printf("error reading input_power\n");
#ifdef LOGS
				log_error_detail("error reading the input power"); end_log_file(); 
#endif
				return 0;
			}
		}
		fclose(fp);

		if( (fp = fopen(OUTPUT_GOLD, "rb" )) == 0 ) {
			printf("error the file 'output_1024' was not opened\n");
#ifdef LOGS
				log_error_detail("error the file 'output_1024' was not opened"); end_log_file(); 
#endif
			return 0;
		}
		for (i=0; i <= grid_rows-1; i++)
		for (j=0; j <= grid_cols-1; j++) {
			return_value = fread(&(output_GOLD[i*grid_cols+j]), 1, sizeof(float), fp);
			if(return_value == 0) {
				printf("error reading gold output\n");
#ifdef LOGS
				log_error_detail("error reading gold output"); end_log_file(); 
#endif
				return 0;
			}
		}
		fclose(fp);

		cudaError_t cuda_error;
		const char *error_string;

		float *MatrixTemp[2], *MatrixPower;
		cuda_error = cudaMalloc((void**)&MatrixTemp[0], sizeof(float)*size);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error MatrixTemp[0] cudaMalloc\n");
#ifdef LOGS
			log_error_detail("error cudaMalloc MatrixTemp[0]"); end_log_file();
#endif
			return 0;
		}

		cuda_error = cudaMalloc((void**)&MatrixTemp[1], sizeof(float)*size);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error MatrixTemp[1] cudaMalloc\n");
#ifdef LOGS
			log_error_detail("error cudaMalloc MatrixTemp[1]"); end_log_file();
#endif
			return 0;
		}

		cuda_error = cudaMemcpy(MatrixTemp[0], inputTemp, sizeof(float)*size, cudaMemcpyHostToDevice);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error MatrixTemp[0] cudaMemcpy\n");
#ifdef LOGS
			log_error_detail("error cudaMemcpy MatrixTemp[0]"); end_log_file(); 
#endif
			return 0;
		}

		cuda_error = cudaMalloc((void**)&MatrixPower, sizeof(float)*size);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error MatrixPower cudaMalloc\n");
#ifdef LOGS
			log_error_detail("error cudaMalloc MatrixPower"); end_log_file(); 
#endif
			return 0;
		}

		cuda_error = cudaMemcpy(MatrixPower, inputPower, sizeof(float)*size, cudaMemcpyHostToDevice);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error MatrixPower cudaMemcpy\n");
#ifdef LOGS
			log_error_detail("error cudaMemcpy MatrixPower"); end_log_file(); 
#endif
			return 0;
		}

		time0 = get_time();
#ifdef LOGS
		start_iteration();
#endif
		int ret = compute_tran_temp(MatrixPower,MatrixTemp,grid_cols,grid_rows, \
			total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows);

		cuda_error = cudaThreadSynchronize();
#ifdef LOGS
		end_iteration();
#endif

		time1 = get_time();

		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error logic: %s\n",error_string);
#ifdef LOGS
			log_error_detail("error logic:"); log_error_detail("error_string"); end_log_file(); 
#endif
			return 0;
		}

		cuda_error = cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(float)*size, cudaMemcpyDeviceToHost);
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error download MatrixOut cudaMemcpy\n");
#ifdef LOGS
			log_error_detail("error download MatrixOut cudaMemcpy"); end_log_file(); 
#endif
			return 0;
		}

		int num_errors = 0;
		for (i=0; i <= grid_rows-1; i++)
		for (j=0; j <= grid_cols-1; j++) {
			if(MatrixOut[i*grid_cols+j] != output_GOLD[i*grid_cols+j])
				num_errors++;
		}

		if(num_errors > 0) {
			t_ea++;
		}
		if (num_errors!=0) printf("Errors: %d\n", num_errors);
#ifdef LOGS
		log_error_count(num_errors);
#endif

		if(num_errors > 0 || (loop % 10 == 0)) {
			printf("\ntest number: %d", loop);
			printf("\namount of errors in the matrix: %d", num_errors);
			printf("\ntotal matrices with errors: %d", t_ea);

			if(num_errors > 0 && num_errors == last_num_errors){
				exit(1);
			}
			last_num_errors = num_errors;

		}
		else {
			printf(".");
			fflush(stdout);
		}

		cudaFree(MatrixPower);
		cudaFree(MatrixTemp[0]);
		cudaFree(MatrixTemp[1]);
		free(MatrixOut);
	}
	printf("\n");
	return EXIT_SUCCESS;
}
