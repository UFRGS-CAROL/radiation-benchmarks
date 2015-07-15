#include "hotspot.h"

#ifdef LOGS
#include "/home/carol/radiation-benchmarks/include/log_helper.h"
#endif /* LOGS */

//#define IMPUT_TEMPE "input_temp"
//#define IMPUT_POWER "input_power"
//#define OUTPUT_GOLD "output"
//#define ALGORITHM_ITERATIONS 1000

int check_output(float *vect, int grid_rows, int grid_cols, float *gold) {

    int i;
    int errors = 0;

    //#pragma omp parallel for reduction(+:errors)
    for (i=0; i < grid_rows; i++) {
        int j;
        for (j=0; j < grid_cols; j++) {
            if(vect[i*grid_cols+j] != gold[i*grid_cols+j] ) {
                errors++;
#ifdef LOGS
		char error_detail[150];
                snprintf(error_detail, 150, "r:%f e:%f [%d,%d]\n", vect[i*grid_cols+j], gold[i*grid_cols+j], i, j);
                log_error_detail(error_detail);
#else
		printf("r:%f e:%f [%d,%d]\n", vect[i*grid_cols+j], gold[i*grid_cols+j], i, j);
		if (errors == 500) exit(0);
#endif /* LOGS */
            }
        }
    }

    return errors;

}


void readinput(float *vect, int grid_rows, int grid_cols, char *file){

  	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if( (fp  = fopen(file, "r" )) ==0 )
            printf( "The file was not opened\n" );


	for (i=0; i <= grid_rows-1; i++) 
	 for (j=0; j <= grid_cols-1; j++)
	 {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
			fatal("not enough lines in file");
		//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
		if ((sscanf(str, "%f", &val) != 1))
			fatal("invalid file format");
		vect[i*grid_cols+j] = val;
	}

	fclose(fp);	

}


/*
   compute N time steps
*/

int compute_tran_temp(cl_mem MatrixPower, cl_mem MatrixTemp[2], int col, int row, \
                      int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows)
{

    float grid_height = chip_height / row;
    float grid_width = chip_width / col;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    int t;

    int src = 0, dst = 1;

    cl_int error;

    // Determine GPU work group grid
    size_t global_work_size[2];
    global_work_size[0] = BLOCK_SIZE * blockCols;
    global_work_size[1] = BLOCK_SIZE * blockRows;
    size_t local_work_size[2];
    local_work_size[0] = BLOCK_SIZE;
    local_work_size[1] = BLOCK_SIZE;


#ifdef LOGS
    start_iteration();
#endif /* LOGS */
    for (t = 0; t < total_iterations; t += num_iterations) {

        // Specify kernel arguments
        int iter = MIN(num_iterations, total_iterations - t);
        clSetKernelArg(kernel, 0, sizeof(int), (void *) &iter);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &MatrixPower);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &MatrixTemp[src]);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &MatrixTemp[dst]);
        clSetKernelArg(kernel, 4, sizeof(int), (void *) &col);
        clSetKernelArg(kernel, 5, sizeof(int), (void *) &row);
        clSetKernelArg(kernel, 6, sizeof(int), (void *) &borderCols);
        clSetKernelArg(kernel, 7, sizeof(int), (void *) &borderRows);
        clSetKernelArg(kernel, 8, sizeof(float), (void *) &Cap);
        clSetKernelArg(kernel, 9, sizeof(float), (void *) &Rx);
        clSetKernelArg(kernel, 10, sizeof(float), (void *) &Ry);
        clSetKernelArg(kernel, 11, sizeof(float), (void *) &Rz);
        clSetKernelArg(kernel, 12, sizeof(float), (void *) &step);

        // Launch kernel
        error = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

        // Flush the queue
        error = clFlush(command_queue);
        if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

        // Swap input and output GPU matrices
        src = 1 - src;
        dst = 1 - dst;
    }

    // Wait for all operations to finish
    error = clFinish(command_queue);
#ifdef LOGS
    end_iteration();
#endif /* LOGS */

    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    return src;
}

void usage(){
        printf("Usage: hotspot_check <sim_iter> <input_size> <cl_device_tipe> <ocl_kernel_file> <input_temp_file> <input_power_file> <output_gold_file> <#iterations>\n");
        printf("  cl_device_types\n");
        printf("    Default: %d\n",CL_DEVICE_TYPE_DEFAULT);
        printf("    CPU: %d\n",CL_DEVICE_TYPE_CPU);
        printf("    GPU: %d\n",CL_DEVICE_TYPE_GPU);
        printf("    ACCELERATOR: %d\n",CL_DEVICE_TYPE_ACCELERATOR);
        printf("    ALL: %d\n",CL_DEVICE_TYPE_ALL);
}


int main(int argc, char** argv) {

    int iterations, devType;
    int grid_rows,grid_cols = 0;
    int tot_iterations=1;
    char *kernel_file, *input_temp, *input_power, *output;
    if(argc == 9) {
        iterations = atoi(argv[1]);
	grid_rows = atoi(argv[2]);
        devType = atoi(argv[3]);
        kernel_file = argv[4];
        input_temp = argv[5];
        input_power = argv[6];
        output = argv[7];
	tot_iterations = atoi(argv[8]);
    } else {
        usage();
        exit(1);
    }
    grid_cols = grid_rows;
    printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

    cl_int error;
    cl_uint num_platforms;

    // Get the number of platforms
    error = clGetPlatformIDs(0, NULL, &num_platforms);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

    // Get the list of platforms
    cl_platform_id* platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
    error = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

    // Print the chosen platform (if there are multiple platforms, choose the first one)
    cl_platform_id platform = platforms[0];
    char pbuf[100];
    error = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    printf("Platform: %s\n", pbuf);

    // Create a GPU context
    cl_context_properties context_properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
    context = clCreateContextFromType(context_properties, devType, NULL, NULL, &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

    // Get and print the chosen device (if there are multiple devices, choose the first one)
    size_t devices_size;
    error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &devices_size);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    cl_device_id *devices = (cl_device_id *) malloc(devices_size);
    error = clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size, devices, NULL);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    device = devices[0];
    error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(pbuf), pbuf, NULL);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    printf("Device: %s\n", pbuf);

    // Create a command queue
    command_queue = clCreateCommandQueue(context, device, 0, &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);



    int size;
    float *FilesavingTemp,*FilesavingPower, *gold;
    int pyramid_height = 1;
    int total_iterations= iterations;

    size=grid_rows*grid_cols;

    // --------------- pyramid parameters ---------------
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));
    gold = (float *) malloc(size*sizeof(float));
    // MatrixOut = (float *) calloc (size, sizeof(float));

    if( !FilesavingPower || !FilesavingTemp) // || !MatrixOut)
        fatal("unable to allocate memory");

    // Read input data from disk
    readinput(FilesavingTemp, grid_rows, grid_cols, input_temp);
    readinput(FilesavingPower, grid_rows, grid_cols, input_power);
    readinput(gold, grid_rows, grid_cols, output);
    // Load kernel source from file
    const char *source = load_kernel_source(kernel_file);
    size_t sourceSize = strlen(source);

    // Compile the kernel
    cl_program program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

    char clOptions[110];
    //  sprintf(clOptions,"-I../../src");
    sprintf(clOptions," ");
#ifdef BLOCK_SIZE
    sprintf(clOptions + strlen(clOptions), " -DBLOCK_SIZE=%d", BLOCK_SIZE);
#endif

    // Create an executable from the kernel
    error = clBuildProgram(program, 1, &device, clOptions, NULL, NULL);
    // Show compiler warnings/errors
    static char log[65536];
    memset(log, 0, sizeof(log));
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
    if (strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    kernel = clCreateKernel(program, "hotspot", &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
#ifdef LOGS
    char test_info[100];
    snprintf(test_info, 100, "simIter:%d gridSize:%dx%d",iterations, grid_rows, grid_rows);
    start_log_file("openclHotspot", test_info);
    set_max_errors_iter(100);
    set_iter_interval_print(10);
#endif /* LOGS */
	int loop1;
	for (loop1=0; loop1 < tot_iterations; loop1++)
	{

		// Create two temperature matrices and copy the temperature input data
		cl_mem MatrixTemp[2];
		// Create input memory buffers on device
		MatrixTemp[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingTemp, &error);
		if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
		MatrixTemp[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * size, NULL, &error);
		if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

		// Copy the power input data
		cl_mem MatrixPower = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingPower, &error);
		if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

		// Perform the computation
		int ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows, total_iterations, pyramid_height,
		                            blockCols, blockRows, borderCols, borderRows);

		// Copy final temperature data back
		cl_float *MatrixOut = (cl_float *) clEnqueueMapBuffer(command_queue, MatrixTemp[ret], CL_TRUE, CL_MAP_READ, 0, sizeof(float) * size, 0, NULL, NULL, &error);
		if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

		// Write final output to output file
		int errors = check_output(MatrixOut, grid_rows, grid_cols, gold);

		if (errors!=0) 
		{
			printf("kernel errors: %d\n", errors);
#ifdef LOGS
    log_error_count(errors);
#endif /* LOGS */
		}
		else
		{
			printf(".");
		}
		fflush(stdout);

		error = clEnqueueUnmapMemObject(command_queue, MatrixTemp[ret], (void *) MatrixOut, 0, NULL, NULL);
		if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
		clFinish(command_queue);

		clReleaseMemObject(MatrixTemp[0]);
		clReleaseMemObject(MatrixTemp[1]);
		clReleaseMemObject(MatrixPower);

		readinput(FilesavingTemp, grid_rows, grid_cols, input_temp); // Reload inputs from disk.
    	readinput(FilesavingPower, grid_rows, grid_cols, input_power);
	}

#ifdef LOGS
    end_log_file();
#endif /* LOGS */
    clReleaseContext(context);

    return 0;
}
