#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>

#define BLOCK_SIZE_X 14
#define BLOCK_SIZE_Y 14

#define ITERACTIONS 100000 //first loop, killed when there is a cuda malloc error, cuda thread sync error, too many output error
#define ITERACTIONS2 10 //second loop

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

// name of the files used to save the output and logging
char file_name[60];
char file_name_log[60];



// Returns the current system time in microseconds
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void UpdateTimestamp(){
	time_t timestamp;
	timestamp = time(NULL);
	FILE *timefile;
	timefile = fopen("/home/carol/TestGPU/timestamp.txt", "w");
	fprintf(timefile, "%d\n", (int)timestamp);
	fclose(timefile);
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__device__ int error_count_gpu;
__global__ void check_errors(int iteration, int grid_cols, int grid_rows, int border_cols, int border_rows, float *matrix1, float *matrix2){

	int bx = blockIdx.x;
        int by = blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

        // calculate the small block size
	int small_block_rows = BLOCK_SIZE_Y-iteration*2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE_X-iteration*2;//EXPAND_RATE

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkY = small_block_rows*by-border_rows;
        int blkX = small_block_cols*bx-border_cols;
        // calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;
        // load data if it is within the valid input range
	int loadYidx=yidx, loadXidx=xidx;
        int index = grid_cols*loadYidx+loadXidx;

	if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
		if (matrix1[index] != matrix2[index]){
			atomicAdd(&error_count_gpu, 1);
		}
	}
}

__global__ void calculate_temp(int iteration,  //number of iteration
                               float *power,   //power input
                               float *temp_src,    //temperature input/output
                               float *temp_dst,    //temperature input/output
                               float *temp_error_check,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
                               int border_cols,  // border offset 
                               int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx, 
                               float Ry, 
                               float Rz, 
                               float step, 
                               float time_elapsed){
	
        __shared__ float temp_on_cuda[BLOCK_SIZE_Y][BLOCK_SIZE_X];
        __shared__ float power_on_cuda[BLOCK_SIZE_Y][BLOCK_SIZE_X];
        __shared__ float temp_t[BLOCK_SIZE_Y][BLOCK_SIZE_X]; // saving temparary temperature result


	float amb_temp = 80.0;
        float step_div_Cap;
        float Rx_1,Ry_1,Rz_1;
        
	int bx = blockIdx.x % (gridDim.x/2);
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
	int small_block_rows = BLOCK_SIZE_Y-iteration*2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE_X-iteration*2;//EXPAND_RATE

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
       
	if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
            temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
            power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
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
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE_X-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE_Y-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
                  computed = true;
                  temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
	       	         (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 + 
		             (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 + 
		             (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
	
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                temp_on_cuda[ty][tx]= temp_t[ty][tx];
            __syncthreads();
          }

	// update the global memory
	// after the last iteration, only threads coordinated within the 
	// small block perform the calculation and switch on ``computed''
	if (computed){
		if(blockIdx.x < (gridDim.x/2)){
			temp_dst[index]= temp_t[ty][tx];
		}else{//impar
			temp_error_check[index]= temp_t[ty][tx];
		}
	}
}

/*
   compute N time steps
*/

int compute_tran_temp(float *MatrixPower,float *MatrixTemp[2], int col, int row, \
		int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows) 
{
        dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	blockCols *= 2;
        dim3 dimGrid(blockCols, blockRows);  
        dim3 dimGrid2(blockCols/2, blockRows);  
//printf("GRID[%d, %d] BLOCKS[%d, %d]\n", blockCols, blockRows, BLOCK_SIZE, BLOCK_SIZE);

	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	int t;
        float time_elapsed;
	time_elapsed=0.001;

        int src = 1, dst = 0;
	
	int zero = 0;
	int error_count = 0;
	int old_iteration = 0;
	int repeated_errors = 0;
        cudaMemcpyToSymbol(error_count_gpu, &zero, sizeof(int));
	for (t = 0; t < total_iterations; t+=num_iterations) {
            int temp = src;
            src = dst;
            dst = temp;
	    
            calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations-t), MatrixPower,MatrixTemp[src],MatrixTemp[dst], MatrixTemp[2],\
		col,row,borderCols, borderRows, Cap,Rx,Ry,Rz,step,time_elapsed);
	    cudaDeviceSynchronize();

	    check_errors<<<dimGrid2, dimBlock>>>(MIN(num_iterations, total_iterations-t), col, row, borderCols, borderRows, MatrixTemp[dst], MatrixTemp[2]);
	    cudaDeviceSynchronize();

	    cudaMemcpyFromSymbol(&error_count, error_count_gpu, sizeof(unsigned int));
	    if(error_count > 0){ //recompute

		if (t == old_iteration){
                	repeated_errors++;
		}else{
			old_iteration = t;
			repeated_errors = 0;
		}

		FILE *log_file;
		log_file = fopen(file_name_log, "a");
		if(log_file == NULL){
			printf("error to open file '%s'\n",file_name_log);
		}
		fprintf(log_file, "\nerrors found; count: %d", error_count);

		if(repeated_errors > 5){
			fprintf(log_file, "\nno more recalc; ending with errors");
			fclose(log_file);
			break;
		}
		fprintf(log_file, "\nit %d; # recalc %d", t, (repeated_errors+1));
		fclose(log_file);

                printf("error_count = %d\n", error_count);
                printf("iteration with error = %d\n", t);
                temp = src;
                src = dst;
                dst = temp;
                t-=num_iterations;
		error_count = 0;
		cudaMemcpyToSymbol(error_count_gpu, &zero, sizeof(unsigned int));
            }


cudaError_t error = cudaGetLastError();
if(error != cudaSuccess){
// print the CUDA error message and exit
printf("CUDA error: %s\n", cudaGetErrorString(error));
exit(-1);}
	    
	}
        return dst;
}


int main(int argc, char** argv){

	int size;
	int grid_rows,grid_cols;
	float *inputTemp,*inputPower,*MatrixOut,*output_GOLD; 
	int pyramid_height, total_iterations;
	long long time0, time1;

	int t_ea = 0;
	double total_kernel_time = 0;

	///////////////////////////////////////////////////////
	////////////////FILE NAME//////////////////////////////
	///////////////////////////////////////////////////////
// infos : dd_mm_yyyy_hh_mm_ss_LavaMD_192_13


// to be modified if you have better ideas

	time_t file_time;
	struct tm *ptm;
	char day[2], month[2], year[4], hour[2], second[2], minute[2];
	
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
	strcat(file_name,"_Hotspotx20_hard");
	strcpy(file_name_log, file_name);
	
	strcat(file_name,".txt");
	strcat(file_name_log,"_log.txt");

	//LOOP START
	int loop;
	int loop2;

	for(loop2=0; loop2<ITERACTIONS2; loop2++){
		printf("loop2 : %d\n", loop2);
		int errors_loop = 0;
		for(loop=0; loop<ITERACTIONS; loop++){

			FILE *file;
			file = fopen(file_name, "a");
			if(file == NULL){
				printf("error to open file '%s'\n",file_name);
				break;
			}

			grid_rows = 1024;
			grid_cols = grid_rows;
			pyramid_height = 1;
			total_iterations = 20000;

			size=grid_rows*grid_cols;

			/* --------------- pyramid parameters --------------- */
			# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
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

			if( !inputPower || !inputTemp || !output_GOLD || !MatrixOut){
				printf("error unable to allocate CPU memory\n");
				fprintf(file, "error unable to allocate CPU memory\n");
				fflush(file); fclose(file);
				break;
			}


			/*printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",\
			pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);*/

			FILE *fp;
			int i, j, return_value;

			if( (fp = fopen("/home/carol/TestGPU/Hotspot/input_temp_1024", "rb" )) == 0 ){
				printf("error the file 'input_temp_1024' was not opened\n");
				fprintf(file, "error the file 'input_temp_1024' was not opened\n");
				fflush(file); fclose(file);
				break;
			}
			for (i=0; i <= grid_rows-1; i++) 
				for (j=0; j <= grid_cols-1; j++) {
					return_value = fread(&(inputTemp[i*grid_cols+j]), 1, sizeof(float), fp);
					if(return_value == 0){
						printf("error reading input_temp\n");
						fprintf(file, "error reading input_temp\n");
						fflush(file); fclose(file);
						break;
					}
				}
			fclose(fp);

			if( (fp = fopen("/home/carol/TestGPU/Hotspot/input_power_1024", "rb" )) == 0 ){
				printf("error the file 'input_power_1024' was not opened\n");
				fprintf(file, "error the file 'input_power_1024' was not opened\n");
				fflush(file); fclose(file);
				break;
			}
			for (i=0; i <= grid_rows-1; i++) 
				for (j=0; j <= grid_cols-1; j++) {
					return_value = fread(&(inputPower[i*grid_cols+j]), 1, sizeof(float), fp);
					if(return_value == 0){
						printf("error reading input_power\n");
						fprintf(file, "error reading input_power\n");
						break;
					}
				}
			fclose(fp);

			if( (fp = fopen("/home/carol/TestGPU/Hotspot/output_1024", "rb" )) == 0 ){
				printf("error the file 'output_1024' was not opened\n");
				fprintf(file, "error the file 'output_1024' was not opened\n");
				fflush(file); fclose(file);
				break;
			}
			for (i=0; i <= grid_rows-1; i++) 
				for (j=0; j <= grid_cols-1; j++) {
					return_value = fread(&(output_GOLD[i*grid_cols+j]), 1, sizeof(float), fp);
					if(return_value == 0){
						printf("error reading gold output\n");
						fprintf(file, "error reading gold output\n");
						fflush(file); fclose(file);
						break;
					}
				}
			fclose(fp);

			cudaError_t cuda_error;
			const char *error_string;

			float *MatrixTemp[3], *MatrixPower;
			cuda_error = cudaMalloc((void**)&MatrixTemp[0], sizeof(float)*size);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error MatrixTemp[0] cudaMalloc\n");
				fprintf(file, "error MatrixTemp[0] cudaMalloc\n");fflush(file); fclose(file); break; }

			cuda_error = cudaMalloc((void**)&MatrixTemp[1], sizeof(float)*size);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error MatrixTemp[1] cudaMalloc\n");
				fprintf(file, "error MatrixTemp[1] cudaMalloc\n");fflush(file); fclose(file); break; }

			cuda_error = cudaMalloc((void**)&MatrixTemp[2], sizeof(float)*size);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error MatrixTemp[2] cudaMalloc\n");
				fprintf(file, "error MatrixTemp[2] cudaMalloc\n");fflush(file); fclose(file); break; }

			cuda_error = cudaMemcpy(MatrixTemp[0], inputTemp, sizeof(float)*size, cudaMemcpyHostToDevice);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error MatrixTemp[0] cudaMemcpy\n");
				fprintf(file, "error MatrixTemp[0] cudaMemcpy\n");fflush(file); fclose(file); break; }

			cuda_error = cudaMalloc((void**)&MatrixPower, sizeof(float)*size);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error MatrixPower cudaMalloc\n");
				fprintf(file, "error MatrixPower cudaMalloc\n");fflush(file); fclose(file); break; }

			cuda_error = cudaMemcpy(MatrixPower, inputPower, sizeof(float)*size, cudaMemcpyHostToDevice);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error MatrixPower cudaMemcpy\n");
				fprintf(file, "error MatrixPower cudaMemcpy\n");fflush(file); fclose(file); break; }


			time0 = get_time();

			int ret = compute_tran_temp(MatrixPower,MatrixTemp,grid_cols,grid_rows, \
				total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows);

			cuda_error = cudaThreadSynchronize();
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error logic: %s\n",error_string);
				fprintf(file, "error logic: %s\n",error_string);fflush(file); fclose(file); break;}

			time1 = get_time();


			cuda_error = cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(float)*size, cudaMemcpyDeviceToHost);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error download MatrixOut cudaMemcpy\n");
				fprintf(file, "error download MatrixOut cudaMemcpy\n");fflush(file); fclose(file); break; }


			int num_errors = 0;
			for (i=0; i <= grid_rows-1; i++) 
			for (j=0; j <= grid_cols-1; j++) {
				if(MatrixOut[i*grid_cols+j] != output_GOLD[i*grid_cols+j])
				num_errors++;
			}

			if(num_errors > 0){
				t_ea++;
				errors_loop++;
			}

			///////////UPDATE LOG FILE//////////////////////
			/// PROBABLY NOT NEEDED, OR CAN BE REDUCED
			double kernel_time = (double) (time1-time0) / 1000000;
			total_kernel_time += kernel_time;
			FILE *log_file;
			log_file = fopen(file_name_log, "a");
			if(log_file == NULL){
				printf("error to open file '%s'\n",file_name_log);
				fprintf(file,"error to open file '%s'\n",file_name_log);
				fflush(file); fclose(file);
				break;
			}
			fprintf(log_file, "\ntest number: %d", loop);
			fprintf(log_file, "\nkernel time: %.12f", kernel_time);
			fprintf(log_file, "\naccumulated kernel time: %.12f", total_kernel_time);
			//fprintf(log_file, "\nerrors: %d", ea);
			fprintf(log_file, "\namount of errors in the matrix: %d", num_errors);
			fprintf(log_file, "\ntotal matrix with errors: %d", t_ea);

			fclose(file);
			fclose(log_file);

			/////////////UPDATE TIMESTAMP///////////////////
			UpdateTimestamp();
			////////////////////////////////////////////////

			if(num_errors > 0 || (loop % 10 == 0)){
				printf("\ntest number: %d", loop);
				printf("\naccumulated kernel time: %f", total_kernel_time);
				printf("\namount of errors in the matrix: %d", num_errors);
				printf("\ntotal matrices with errors: %d", t_ea);

				if(errors_loop >= loop/2 && loop > 10) break; //we NEED this, beause at times the GPU get stuck and it gives a huge amount of error, we cannot let it write a stream of data on the HDD

			}
			else
			{
				printf(".");
			}


			cudaFree(MatrixPower);
			cudaFree(MatrixTemp[0]);
			cudaFree(MatrixTemp[1]);
			free(MatrixOut);
		}
		printf("\n");
	}
	return EXIT_SUCCESS;
}
