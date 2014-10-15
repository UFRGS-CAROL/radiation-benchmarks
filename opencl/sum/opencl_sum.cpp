
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>


#define N 67108864
#define num_sum 10000

#define ITERACTIONS 1000000

#define CL_CHECK_ERROR(err) \
	{ \
		if (err != CL_SUCCESS) \
		std::cerr << "Error: " \
		<< CLErrorString(err) \
		<< " in " << __FILE__ \
		<< " line " << __LINE__ \
		<< std::endl; \
	}

	inline const char *CLErrorString(cl_int err)
{
	switch (err)
	{
								 // break;
		case CL_SUCCESS:                         return "CL_SUCCESS";
								 // break;
		case CL_DEVICE_NOT_FOUND:                return "CL_DEVICE_NOT_FOUND";
								 // break;
		case CL_DEVICE_NOT_AVAILABLE:            return "CL_DEVICE_NOT_AVAILABLE";
								 // break;
		case CL_COMPILER_NOT_AVAILABLE:          return "CL_COMPILER_NOT_AVAILABLE";
								 // break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:   return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
								 // break;
		case CL_OUT_OF_RESOURCES:                return "CL_OUT_OF_RESOURCES";
								 // break;
		case CL_OUT_OF_HOST_MEMORY:              return "CL_OUT_OF_HOST_MEMORY";
								 // break;
		case CL_PROFILING_INFO_NOT_AVAILABLE:    return "CL_PROFILING_INFO_NOT_AVAILABLE";
								 // break;
		case CL_MEM_COPY_OVERLAP:                return "CL_MEM_COPY_OVERLAP";
								 // break;
		case CL_IMAGE_FORMAT_MISMATCH:           return "CL_IMAGE_FORMAT_MISMATCH";
								 // break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
								 // break;
		case CL_BUILD_PROGRAM_FAILURE:           return "CL_BUILD_PROGRAM_FAILURE";
								 // break;
		case CL_MAP_FAILURE:                     return "CL_MAP_FAILURE";
								 // break;
		case CL_INVALID_VALUE:                   return "CL_INVALID_VALUE";
								 // break;
		case CL_INVALID_DEVICE_TYPE:             return "CL_INVALID_DEVICE_TYPE";
								 // break;
		case CL_INVALID_PLATFORM:                return "CL_INVALID_PLATFORM";
								 // break;
		case CL_INVALID_DEVICE:                  return "CL_INVALID_DEVICE";
								 // break;
		case CL_INVALID_CONTEXT:                 return "CL_INVALID_CONTEXT";
								 // break;
		case CL_INVALID_QUEUE_PROPERTIES:        return "CL_INVALID_QUEUE_PROPERTIES";
								 // break;
		case CL_INVALID_COMMAND_QUEUE:           return "CL_INVALID_COMMAND_QUEUE";
								 // break;
		case CL_INVALID_HOST_PTR:                return "CL_INVALID_HOST_PTR";
								 // break;
		case CL_INVALID_MEM_OBJECT:              return "CL_INVALID_MEM_OBJECT";
								 // break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
								 // break;
		case CL_INVALID_IMAGE_SIZE:              return "CL_INVALID_IMAGE_SIZE";
								 // break;
		case CL_INVALID_SAMPLER:                 return "CL_INVALID_SAMPLER";
								 // break;
		case CL_INVALID_BINARY:                  return "CL_INVALID_BINARY";
								 // break;
		case CL_INVALID_BUILD_OPTIONS:           return "CL_INVALID_BUILD_OPTIONS";
								 // break;
		case CL_INVALID_PROGRAM:                 return "CL_INVALID_PROGRAM";
								 // break;
		case CL_INVALID_PROGRAM_EXECUTABLE:      return "CL_INVALID_PROGRAM_EXECUTABLE";
								 // break;
		case CL_INVALID_KERNEL_NAME:             return "CL_INVALID_KERNEL_NAME";
								 // break;
		case CL_INVALID_KERNEL_DEFINITION:       return "CL_INVALID_KERNEL_DEFINITION";
								 // break;
		case CL_INVALID_KERNEL:                  return "CL_INVALID_KERNEL";
								 // break;
		case CL_INVALID_ARG_INDEX:               return "CL_INVALID_ARG_INDEX";
								 // break;
		case CL_INVALID_ARG_VALUE:               return "CL_INVALID_ARG_VALUE";
								 // break;
		case CL_INVALID_ARG_SIZE:                return "CL_INVALID_ARG_SIZE";
								 // break;
		case CL_INVALID_KERNEL_ARGS:             return "CL_INVALID_KERNEL_ARGS";
								 // break;
		case CL_INVALID_WORK_DIMENSION:          return "CL_INVALID_WORK_DIMENSION";
								 // break;
		case CL_INVALID_WORK_GROUP_SIZE:         return "CL_INVALID_WORK_GROUP_SIZE";
								 // break;
		case CL_INVALID_WORK_ITEM_SIZE:          return "CL_INVALID_WORK_ITEM_SIZE";
								 // break;
		case CL_INVALID_GLOBAL_OFFSET:           return "CL_INVALID_GLOBAL_OFFSET";
								 // break;
		case CL_INVALID_EVENT_WAIT_LIST:         return "CL_INVALID_EVENT_WAIT_LIST";
								 // break;
		case CL_INVALID_EVENT:                   return "CL_INVALID_EVENT";
								 // break;
		case CL_INVALID_OPERATION:               return "CL_INVALID_OPERATION";
								 // break;
		case CL_INVALID_GL_OBJECT:               return "CL_INVALID_GL_OBJECT";
								 // break;
		case CL_INVALID_BUFFER_SIZE:             return "CL_INVALID_BUFFER_SIZE";
								 // break;
		case CL_INVALID_MIP_LEVEL:               return "CL_INVALID_MIP_LEVEL";
								 // break;
		case CL_INVALID_GLOBAL_WORK_SIZE:        return "CL_INVALID_GLOBAL_WORK_SIZE";
								 // break;
		case CL_INVALID_PROPERTY:                return "CL_INVALID_PROPERTY";
								 // break;
		default:                                 return "UNKNOWN";
	}
}

double *A;
double *B;
double *C;

double fRand(double fmin, double fmax){

	double f = (double) rand() / RAND_MAX;

	return (double )fmin + f * (fmax - fmin);
}


void ReadMatrixFromFile(){	
	

	FILE* f_A;
	FILE* f_B;
	FILE* f_GOLD;

	int i;
	int j;
printf("open...");
//	f_A = fopen("/home/carol/TestGPU/GenerateGoldMatrix/Double_A_8196.matrix","rb");
//	f_B = fopen("/home/carol/TestGPU/GenerateGoldMatrix/Double_B_8196.matrix","rb");
	//f_GOLD = fopen("/home/carol/TestGPU/GenerateGoldMatrix/Double_GOLD_4096.matrix","rb");
printf("read...");
	for(i=0; i<N; i++)
	{
		
			A[i] = fRand(-15900.35, 19870.59);
			B[i] = fRand(-15400.65, 15480.68);

			//GOLD[i] = 101.0;//4096.0; 

			
//			fread(&A[i],sizeof(double), 1, f_A);
//			fread(&B[i],sizeof(double), 1, f_B);
			
			//GOLD[i] = A[i] + (num_sum * B[i]);
			//fread(&GOLD[j + N * i],sizeof(double), 1, f_GOLD);
			
	}

//GOLD[55] = 5.5;
//GOLD[40] = 4.7;

//	fclose(f_A);
//	fclose(f_B);
	//fclose(f_GOLD);
}


double mysecond(){
   struct timeval tp;
   struct timezone tzp;
   int i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

cl_platform_id          platform_id[100];
cl_device_id            device_id[100];
cl_context              context;
cl_command_queue        command_queue;
cl_int ret;
void getDevices(cl_device_type deviceType)
{
	cl_uint         platforms_n = 0;
	cl_uint         devices_n   = 0;

	/* The following code queries the number of platforms and devices, and
	 * lists the information about both.
	 */
	clGetPlatformIDs(100, platform_id, &platforms_n);
	if (0)
	{
		printf("\n=== %d OpenCL platform(s) found: ===\n", platforms_n);
		for (int i = 0; i < platforms_n; i++)
		{
			char buffer[10240];
			printf("  -- %d --\n", i);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_PROFILE, 10240, buffer,
				NULL);
			printf("  PROFILE = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_VERSION, 10240, buffer,
				NULL);
			printf("  VERSION = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, 10240, buffer, NULL);
			printf("  NAME = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL);
			printf("  VENDOR = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_EXTENSIONS, 10240, buffer,
				NULL);
			printf("  EXTENSIONS = %s\n", buffer);
		}
	}

	clGetDeviceIDs(platform_id[0], deviceType, 100, device_id, &devices_n);
	if (0)
	{
		printf("Using the default platform (platform 0)...\n\n");
		printf("=== %d OpenCL device(s) found on platform:\n", devices_n);
		for (int i = 0; i < devices_n; i++)
		{
			char buffer[10240];
			cl_uint buf_uint;
			cl_ulong buf_ulong;
			printf("  -- %d --\n", i);
			clGetDeviceInfo(device_id[i], CL_DEVICE_NAME, sizeof(buffer), buffer,
				NULL);
			printf("  DEVICE_NAME = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer,
				NULL);
			printf("  DEVICE_VENDOR = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_VERSION, sizeof(buffer), buffer,
				NULL);
			printf("  DEVICE_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DRIVER_VERSION, sizeof(buffer), buffer,
				NULL);
			printf("  DRIVER_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_COMPUTE_UNITS,
				sizeof(buf_uint), &buf_uint, NULL);
			printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int) buf_uint);
			clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_CLOCK_FREQUENCY,
				sizeof(buf_uint), &buf_uint, NULL);
			printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int) buf_uint);
			clGetDeviceInfo(device_id[i], CL_DEVICE_GLOBAL_MEM_SIZE,
				sizeof(buf_ulong), &buf_ulong, NULL);
			printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n",
				(unsigned long long) buf_ulong);
			clGetDeviceInfo(device_id[i], CL_DEVICE_LOCAL_MEM_SIZE,
				sizeof(buf_ulong), &buf_ulong, NULL);
			printf("  CL_DEVICE_LOCAL_MEM_SIZE = %llu\n",
				(unsigned long long) buf_ulong);
		}
		printf("\n");
	}

	// Create an OpenCL context.
	context = clCreateContext(NULL, devices_n, device_id, NULL, NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("\nError at clCreateContext! Error code %i\n\n", ret);
		exit(1);
	}

	// Create a command queue.
	command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
		exit(1);
	}
}

double total_time = 0;
int t_ea=0;

FILE* file;
FILE* log_file;
FILE* timefile;

void UpdateTimestamp(){
	time_t timestamp = time(NULL);
	char time_s[50];
	sprintf(time_s, "%d", int(timestamp));

	char string[100] = "echo ";
	strcat(string, time_s);
	strcat(string, " > /home/carol/TestGPU/timestamp.txt");
	system(string);
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////

int
main(int argc, char** argv)
{

	///////////////////////////////////////////////////////
	////////////////FILE NAME//////////////////////////////
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
	strcat(file_name,"SUM_OpenCL");
	strcpy(file_name_log, file_name);
	
	strcat(file_name,".txt");
	strcat(file_name_log,"_log.txt");
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////

	// 1. allocate host memory for matrices A and B
	int size = N;

	A = ( double* ) malloc( size * sizeof( double ) );
	B = ( double* ) malloc( size * sizeof( double ) );

	C = ( double* ) malloc( size * sizeof( double ) );

	ReadMatrixFromFile();

	printf( "OpenCL_SUM\n" );

	int old_ea = 0;

	unsigned int loop2;
	for(loop2=0; loop2<ITERACTIONS; loop2++)
	{

	file = fopen(file_name, "a");	

printf("loop...\n");
	// 4. allocate host memory for the result C

	// 5. Initialize OpenCL
	// OpenCL specific variables
	cl_program clProgram;
	cl_kernel clKernel;
	//cl_kernel clKernelCheck;

	size_t dataBytes;
	size_t kernelLength;
	cl_int errcode;

printf("cl_mem...\n");
	// OpenCL device memory for matrices
	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_num_errors;

	/*****************************************/
	/* Initialize OpenCL */
	/*****************************************/
	getDevices(CL_DEVICE_TYPE_GPU);

printf("getDevices...\n");
	// Setup device memory
	unsigned int mem_size = sizeof(double) * size;
	d_num_errors = clCreateBuffer(context,
		CL_MEM_READ_WRITE,
		sizeof(int), NULL, &errcode);
	d_A = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		mem_size, A, &errcode);
	d_B = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		mem_size, B, &errcode);

printf("createBuffer...\n");

	// 6. Load and build OpenCL kernel
	FILE*  theFile = fopen("/home/carol/TestGPU/opencl_sum/sum_kernel.cl", "r");
	if (!theFile)
	{
		fprintf(stderr, "Failed to load kernel file.\n");
		exit(1);
	}
	char* source_str;

	// Obtain length of source file.
	fseek(theFile, 0, SEEK_END);
	size_t source_size = ftell(theFile);
	rewind(theFile);

printf("read kernel.cl...\n");
	// Read in the file.
	source_str = (char*) malloc(sizeof(char) * source_size);
	fread(source_str, 1, source_size, theFile);
	fclose(theFile);
	source_str[source_size] = '\0';
	CL_CHECK_ERROR(errcode);
	clProgram = clCreateProgramWithSource(context,
		1, (const char **)&source_str,
		NULL, &errcode);
	CL_CHECK_ERROR(errcode);

	free(source_str);

printf("BuildProgram...\n");
	errcode = clBuildProgram(clProgram, 1,
		&device_id[0], NULL, NULL, NULL);
	CL_CHECK_ERROR(errcode);

	cl_build_status status;
	size_t logSize;
	char *programLog;
	// check build error and build status first
        clGetProgramBuildInfo(clProgram, device_id[0], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);

	
        // check build log
        clGetProgramBuildInfo(clProgram, device_id[0],
                CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        programLog = (char*) calloc (logSize+1, sizeof(char));
        clGetProgramBuildInfo(clProgram, device_id[0],
                CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
        printf("Build; error=%d, status=%d, programLog:\n\n%s",
        errcode, status, programLog);
        free(programLog);


printf("createKernel...\n");
	clKernel = clCreateKernel(clProgram,
		"simpleSum", &errcode);
	CL_CHECK_ERROR(errcode);


	// 7. Launch OpenCL kernel
	size_t localWorkSize[2], globalWorkSize[2];

printf("parameters...\n");
	int kernel_errors = 0;
	clEnqueueWriteBuffer(command_queue, d_num_errors, CL_FALSE, 0, 
sizeof(int), &kernel_errors, 0, NULL, NULL); 
	clFinish(command_queue);

	int n = num_sum;
	errcode |= clSetKernelArg(clKernel, 0,
		sizeof(cl_mem), (void *)&d_A);
	errcode |= clSetKernelArg(clKernel, 1,
		sizeof(cl_mem), (void *)&d_B);
	errcode |= clSetKernelArg(clKernel, 2,
		sizeof(cl_mem), (void *)&d_num_errors);
	errcode |= clSetKernelArg(clKernel, 3,
		sizeof(int), (void *)&n);
	CL_CHECK_ERROR(errcode);

	localWorkSize[0] = 1024;
	//localWorkSize[1] = 32;
	globalWorkSize[0] = N;
	//globalWorkSize[1] = N;

	clFinish(command_queue);

printf("RUN...\n");
	double timeG = mysecond();
	errcode = clEnqueueNDRangeKernel(command_queue,
		clKernel, 1, NULL, globalWorkSize,
		localWorkSize, 0, NULL, NULL);
	CL_CHECK_ERROR(errcode);
	clFinish(command_queue);
	timeG = mysecond() - timeG;

	total_time += timeG;

/*
	// 8. Retrieve result from device
	errcode = clEnqueueReadBuffer(command_queue,
		d_A, CL_TRUE, 0, mem_size,
		A, 0, NULL, NULL);
	CL_CHECK_ERROR(errcode);

	// CHECK GOLD 
	clKernelCheck = clCreateKernel(clProgram,
		"checkGold", &errcode);
	CL_CHECK_ERROR(errcode);

	clEnqueueWriteBuffer(command_queue, d_GOLD, CL_FALSE, 0, 
mem_size, GOLD, 0, NULL, NULL); 

	int kernel_errors = 0;
	clEnqueueWriteBuffer(command_queue, d_num_errors, CL_FALSE, 0, 
sizeof(int), &kernel_errors, 0, NULL, NULL); 

	clFinish(command_queue);
	errcode = clSetKernelArg(clKernelCheck, 0,
		sizeof(cl_mem), (void *)&d_A);
	errcode |= clSetKernelArg(clKernelCheck, 1,
		sizeof(cl_mem), (void *)&d_GOLD);
	errcode |= clSetKernelArg(clKernelCheck, 2,
		sizeof(cl_mem), (void *)&d_num_errors);
	CL_CHECK_ERROR(errcode);

	double timeCheck = mysecond();
	errcode = clEnqueueNDRangeKernel(command_queue,
		clKernelCheck, 1, NULL, globalWorkSize,
		localWorkSize, 0, NULL, NULL);
	CL_CHECK_ERROR(errcode);
	clFinish(command_queue);
	timeCheck = mysecond() - timeCheck;
*/
	errcode = clEnqueueReadBuffer(command_queue,
		d_num_errors, CL_TRUE, 0, sizeof(int),
		&kernel_errors, 0, NULL, NULL);
	CL_CHECK_ERROR(errcode);

	printf("check kernel ea = %d\n",kernel_errors);

	///////////UPDATE FILE//////////////////////
	file_time = time(NULL);
	ptm = gmtime(&file_time);
	snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
	snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
	snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);
	fprintf(file, "\n start time: %s/%s_%s:%s:%s", day,month,hour,minute,second);


	/////////////UPDATE TIMESTAMP///////////////////
	UpdateTimestamp();
	////////////////////////////////////////////////

int ea = 0;
int i, j;
if (kernel_errors!=0)
	{

	errcode = clEnqueueReadBuffer(command_queue,
		d_A, CL_TRUE, 0, mem_size,
		C, 0, NULL, NULL);
	CL_CHECK_ERROR(errcode);
	clFinish(command_queue);
		//file = fopen(file_name, "a");
		
		printf("\n kernel error: %d\n", kernel_errors);
	

		//malloc_mem1 = cudaMemcpy(A, d_C, size * sizeof( double ), cudaMemcpyDeviceToHost);
		//erro_malloc = cudaGetErrorString(malloc_mem1);
		//if(strcmp(erro_malloc, "no error") != 0)
		//	{printf("error mem load a %s", erro_malloc); fprintf(file, "error mem load a %s", erro_malloc); return 1;}

		for(i=0; (i<N) && (ea < 500); i++)
		{
//GOLD = A + num_sum * B

				if ((fabs((C[i]- (A[i] + num_sum*B[i]) )/C[i]) > 0.0000000001)||(fabs((C[i]-(A[i] + num_sum*B[i]))/(A[i] + num_sum*B[i])) > 0.0000000001))
				{
					ea++;
								
					fprintf(file, "\n p: [%d], r: %1.16e, e: %1.16e, error: %d\n", i, C[i], (A[i] + num_sum*B[i]), ea);
										
				}
		}	

}
/*			ea = 0;
			int i,j;

for(i=0; (i<N); i++)
		{
			for(j=0; (j<N); j++)
			{
				if ((fabs((C[i+N*j]-GOLD[i+N*j])/C[i+N*j]) > 0.0000000001)||(fabs((C[i+N*j]-GOLD[i+N*j])/GOLD[i+N*j]) > 0.0000000001))
				{
					ea++;
					if(ea <= 500){
						fprintf(file,"\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, C[i + N * j], GOLD[i + N * j], ea);
					}									
				}
			}
		}
*/
printf("errors: %d\n",ea);
	t_ea += kernel_errors;
printf("total errors: %d\n",t_ea);




			///////////UPDATE LOG FILE//////////////////////
			log_file = fopen(file_name_log, "a");
			fprintf(log_file, "\ntest number: %d", loop2);
			fprintf(log_file, "\ntime: %f", timeG);
			fprintf(log_file, "\ntotal time: %f", total_time);
			fprintf(log_file, "\nerrors: %d", kernel_errors);
			fprintf(log_file, "\ntotal errors: %d", t_ea);
			fclose(log_file);
			fclose(file);

		if(ea > 0){
			ReadMatrixFromFile();
		}

if(kernel_errors > 0 || (loop2 % 10 == 0))
	{
		printf("\ntest number: %d", loop2);
		printf("\ntime: %f", timeG);
		printf("\ntotal time: %f", total_time);
		printf("\nerrors: %d", kernel_errors);
		printf("\ntotal errors: %d", t_ea);
		if((kernel_errors != 0) && (kernel_errors == old_ea))
			{
				old_ea = 0;
				return 1;
			}
				
			old_ea = kernel_errors;
	}
	else
	{
		printf(".");
	}

	// 10. clean up memory

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	//clReleaseMemObject(d_GOLD);
	clReleaseMemObject(d_num_errors);

	clReleaseContext(context);
	clReleaseKernel(clKernel);
	//clReleaseKernel(clKernelCheck);
	clReleaseProgram(clProgram);
	clReleaseCommandQueue(command_queue);

	}// for

}
