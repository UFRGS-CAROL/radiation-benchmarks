#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include "kmeans.h"

#include <pthread.h>
#include <sys/time.h>
#include <time.h>

#ifdef LOGS
#include "/home/carol/radiation-benchmarks/include/log_helper.h"
#endif /* LOGS */

#ifdef NV
#include <oclUtils.h>
#else
#include <CL/cl.h>
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#define AVOIDZERO 1e-200
#define ACCEPTDIFF 1e-3

int workgroup_blocksize = 256;
int devType = 1;

//long long get_time() {
//        struct timeval tv;
//        gettimeofday(&tv, NULL);
//        return (tv.tv_sec * 1000000) + tv.tv_usec;
//}



// local variables
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;


static int shutdown()
{
    // release resources
    if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
    if( context ) clReleaseContext( context );
    if( device_list ) delete device_list;

    // reset all variables
    cmd_queue = 0;
    context = 0;
    device_list = 0;
    num_devices = 0;
    device_type = 0;

    return 0;
}

cl_mem d_feature;
cl_mem d_feature_swap;
cl_mem d_cluster;
cl_mem d_membership;

cl_kernel kernel;
cl_kernel kernel_s;
cl_kernel kernel2;

int   *membership_OCL;
int   *membership_d;
float *feature_d;
float *clusters_d;
float *center_d;

char *kernel_file;

static int initialize(int use_gpu)
{
    cl_int result;
    size_t size;

    // create OpenCL context
    cl_platform_id platform_id;
    if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) {
        printf("ERROR: clGetPlatformIDs(1,*,0) failed\n");
        return -1;
    }
    cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
    device_type = devType;
    context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
    if( !context ) {
        printf("ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU");
        return -1;
    }

    // get the list of GPUs
    result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
    num_devices = (int) (size / sizeof(cl_device_id));

    if( result != CL_SUCCESS || num_devices < 1 ) {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }
    device_list = new cl_device_id[num_devices];
    if( !device_list ) {
        printf("ERROR: new cl_device_id[] failed\n");
        return -1;
    }
    result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
    if( result != CL_SUCCESS ) {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }

    // create command queue for the first device
    cmd_queue = clCreateCommandQueue( context, device_list[0], 0, NULL );
    if( !cmd_queue ) {
        printf("ERROR: clCreateCommandQueue() failed\n");
        return -1;
    }

int sourcesize = 1024*1024;
    char * source = (char *)calloc(sourcesize, sizeof(char));
    if(!source) {
        printf("ERROR: calloc(%d) failed\n", sourcesize);
        return -1;
    }

    // read the kernel core source
    char * tempchar = kernel_file;//"./kmeans.cl";
    FILE * fp = fopen(tempchar, "rb");
    if(!fp) {
        printf("ERROR: unable to open '%s'\n", tempchar);
        return -1;
    }
    fread(source + strlen(source), sourcesize, 1, fp);
    fclose(fp);

    // compile kernel
    cl_int err = 0;
    const char * slist[2] = { source, 0 };
    cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
    if(err != CL_SUCCESS) {
        printf("ERROR: clCreateProgramWithSource() => %d\n", err);
        return -1;
    }
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    {   // show warnings/errors
        //	static char log[65536]; memset(log, 0, sizeof(log));
        //	cl_device_id device_id = 0;
        //	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
        //	clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
        //	if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
    }
    if(err != CL_SUCCESS) {
        printf("ERROR: clBuildProgram() => %d\n", err);
        return -1;
    }

    char * kernel_kmeans_c  = "kmeans_kernel_c";
    char * kernel_swap  = "kmeans_swap";

    kernel_s = clCreateKernel(prog, kernel_kmeans_c, &err);
    if(err != CL_SUCCESS) {
        printf("ERROR: clCreateKernel() 0 => %d\n", err);
        return -1;
    }
    kernel2 = clCreateKernel(prog, kernel_swap, &err);
    if(err != CL_SUCCESS) {
        printf("ERROR: clCreateKernel() 0 => %d\n", err);
        return -1;
    }

    clReleaseProgram(prog);

    return 0;
}


int allocate(int n_points, int n_features, int n_clusters, float **feature)
{
	cl_int err = 0;
   

    d_feature = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err );
    if(err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer d_feature (size:%d) => %d\n", n_points * n_features, err);
        return -1;
    }
    d_feature_swap = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err );
    if(err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer d_feature_swap (size:%d) => %d\n", n_points * n_features, err);
        return -1;
    }
    d_cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, n_clusters * n_features  * sizeof(float), NULL, &err );
    if(err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer d_cluster (size:%d) => %d\n", n_clusters * n_features, err);
        return -1;
    }
    d_membership = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) {
        printf("ERROR: clCreateBuffer d_membership (size:%d) => %d\n", n_points, err);
        return -1;
    }

    //write buffers
    err = clEnqueueWriteBuffer(cmd_queue, d_feature, 1, 0, n_points * n_features * sizeof(float), feature[0], 0, 0, 0);
    if(err != CL_SUCCESS) {
        printf("ERROR: clEnqueueWriteBuffer d_feature (size:%d) => %d\n", n_points * n_features, err);
        return -1;
    }

    clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &d_feature);
    clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &d_feature_swap);
    clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*) &n_points);
    clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &n_features);

    size_t global_work[3] = { n_points, 1, 1 };
    /// Ke Wang adjustable local group size 2013/08/07 10:37:33
    size_t local_work_size= workgroup_blocksize; // work group size is defined by RD_WG_SIZE_0 or RD_WG_SIZE_0_0 2014/06/10 17:00:51
    if(global_work[0]%local_work_size !=0)
        global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;

    err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, &local_work_size, 0, 0, 0);
    if(err != CL_SUCCESS) {
        printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err);
        return -1;
    }

    membership_OCL = (int*) malloc(n_points * sizeof(int));
}

void deallocateMemory()
{
    clReleaseMemObject(d_feature);
    clReleaseMemObject(d_feature_swap);
    clReleaseMemObject(d_cluster);
    clReleaseMemObject(d_membership);
    free(membership_OCL);

}

void usage(char *argv0) {
    printf("Usage: %s <cl_device_tipe> <ocl_kernel_file> <#points> <#features> <input_file> <output_gold_file> <#iterations> <workgroup_block_size>\n", argv0);
    printf("  cl_device_types\n");
    printf("    Default: %d\n",CL_DEVICE_TYPE_DEFAULT);
    printf("    CPU: %d\n",CL_DEVICE_TYPE_CPU);
    printf("    GPU: %d\n",CL_DEVICE_TYPE_GPU);
    printf("    ACCELERATOR: %d\n",CL_DEVICE_TYPE_ACCELERATOR);
    printf("    ALL: %d\n",CL_DEVICE_TYPE_ALL);
}

void readSize(char *input_file, int *ptrnpoints, int *ptrnfeatures)
{
	FILE *infile;
	int i, ret;

    if ((infile = fopen(input_file, "rb")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", input_file);
#ifdef LOGS
		log_error_detail("Cant open input"); end_log_file(); 
#endif
        exit(1);
    }
    ret=fread(ptrnpoints,   1, sizeof(int), infile);
    ret=fread(ptrnfeatures, 1, sizeof(int), infile);

    fclose(infile);
}

void readInput(char *input_file, int npoints, int nfeatures, float **features, float *buf)
{
	FILE *infile;
	int i, ret;

    if ((infile = fopen(input_file, "rb")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", input_file);
#ifdef LOGS
		log_error_detail("Cant open input"); end_log_file(); 
#endif
        exit(1);
    }

    for (i=1; i<npoints; i++)
        features[i] = features[i-1] + nfeatures;
    ret=fread(buf, 1, npoints*nfeatures*sizeof(float), infile);

    fclose(infile);
}

void readGold(char *gold_file, int max_nclusters, int nfeatures, float **gold_cluster_centres)
{
	int i, j;
	FILE *fgold;
	if ((fgold=fopen(gold_file, "rb"))==NULL)
	{	fprintf(stderr, "Error: no such file (%s)\n", gold_file);
#ifdef LOGS
		log_error_detail("Cant open gold"); end_log_file(); 
#endif
        exit(1);
    }	
	for (i=1; i<max_nclusters; i++)
		gold_cluster_centres[i]=gold_cluster_centres[i-1]+nfeatures;

	int ret = fread(gold_cluster_centres[0], 1, max_nclusters*nfeatures*sizeof(float), fgold);
	printf("Got %d gold entries.\n", ret);
	fclose(fgold);
}


int main( int argc, char** argv)
{

    char   *input_filename = 0;
    float  *buf;
    char	line[1024];
    float	threshold = 0.001;		/* default value */
    int		max_nclusters=5;		/* default value */
    int		min_nclusters=5;		/* default value */
    int		nfeatures = 0;
    int		npoints = 0;
    float	len;

    float **features;
    float **cluster_centres=NULL;
	float **gold_cluster_centres;
    int		i, j;
    int		nloops = 1;				/* default value */
    int		isOutput = 0;
	int 	loop1, iteractions;

	int enable_perfmeasure = 0;

    char *output;
    if(argc == 9) {
        devType = atoi(argv[1]);
        kernel_file = argv[2];
        npoints = atoi(argv[3]);
        nfeatures = atoi(argv[4]);
        input_filename = argv[5];
        output = argv[6];
        iteractions = atoi(argv[7]);
        workgroup_blocksize = atoi(argv[8]);
    } else {
        usage(argv[0]);
        exit(1);
    }
    printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d \n", workgroup_blocksize, workgroup_blocksize);

    isOutput = 1;

    if (input_filename == 0) usage(argv[0]);

    /* ============== I/O begin ==============*/
    /* get nfeatures and npoints */
    readSize(input_filename, &npoints, &nfeatures);

	gold_cluster_centres    = (float**)malloc(max_nclusters*          sizeof(float*));
	gold_cluster_centres[0] = (float*) malloc(max_nclusters*nfeatures*sizeof(float));
	readGold(output, max_nclusters, nfeatures, gold_cluster_centres);

	// OpenCL initialization
    int use_gpu = 1;
    if(initialize(use_gpu)) return -1;

    printf("clKmeans. npoints=%d nfeatures=%d threshold=%f clusters=%d ITERACTIONS=%d\n", npoints, nfeatures, threshold, max_nclusters, iteractions);fflush(stdout);
#ifdef LOGS
    char test_info[100];
    snprintf(test_info, 100, "npoints:%d nfeatures:%d threshold:%f clusters:%d", npoints, nfeatures, threshold, max_nclusters);
    start_log_file("openclKmeans", test_info);
    set_max_errors_iter(500);
    set_iter_interval_print(10);
#endif /* LOGS */
	for (loop1=0; loop1<iteractions; loop1++)
	{
		/* ============== I/O begin ==============*/
		/* allocate space for features[][] and read attributes of all objects */
		buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
		features    = (float**)malloc(npoints*          sizeof(float*));
		features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
		readInput(input_filename, npoints, nfeatures, features, buf);
	
		/* ============== I/O end ==============*/

		// error check for clusters
		if (npoints < min_nclusters)
		{
		    printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", min_nclusters, npoints);
		    exit(0);
		}

		srand(7);												/* seed for future random number generator */
		memcpy(features[0], buf, npoints*nfeatures*sizeof(float)); /* now features holds 2-dimensional array of features */
		free(buf);

		/* ======================= core of the clustering ===================*/

		cluster_centres = NULL;
#ifdef LOGS
//        start_iteration();
#endif /* LOGS */
		cluster(npoints, nfeatures, features, min_nclusters, max_nclusters, threshold, &cluster_centres, nloops, enable_perfmeasure);
#ifdef LOGS
//        end_iteration();
#endif /* LOGS */

		/* =============== Command Line Output =============== */
		char error_detail[150];
		int kernel_errors=0;
		for(i = 0; i < max_nclusters; i++){
			for(j = 0; j < nfeatures; j++){
            if ((fabs(gold_cluster_centres[i][j])>AVOIDZERO)&&
                    ((fabs((cluster_centres[i][j]-gold_cluster_centres[i][j])/cluster_centres[i][j])>ACCEPTDIFF)||
                     (fabs((cluster_centres[i][j]-gold_cluster_centres[i][j])/gold_cluster_centres[i][j])>ACCEPTDIFF))) {
				//if (gold_cluster_centres[i][j]!=cluster_centres[i][j])
				//{
					kernel_errors++;
					snprintf(error_detail, 150, "p: [%d, %d], r: %1.16e, e: %1.16e", i, j, cluster_centres[i][j], gold_cluster_centres[i][j]);
					printf("%s\n", error_detail);
#ifdef LOGS
					log_error_detail(error_detail); 
#endif
				}
			}
		}
#ifdef LOGS
        log_error_count(kernel_errors);
#endif /* LOGS */
		if (kernel_errors>0)
			printf("\nERROR FOUND. Test number: %d\n", loop1);
		printf(".");
		fflush(stdout);
		
		free(cluster_centres[0]);
		free(cluster_centres);
		free(features[0]);
		free(features);
	}
#ifdef LOGS
    end_log_file();
#endif /* LOGS */
    shutdown();
}

double mysecond()
{
   struct timeval tp;
   struct timezone tzp;
   int i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


int	kmeansOCL(float **feature,    /* in: [npoints][nfeatures] */
              int     n_features,
              int     n_points,
              int     n_clusters,
              int    *membership,
              float **clusters,
              int     *new_centers_len,
              float  **new_centers,
			  	double *kernel_time)
{

    int delta = 0;
    int i, j, k;
    cl_int err = 0;

    size_t global_work[3] = { n_points, 1, 1 };

    /// Ke Wang adjustable local group size 2013/08/07 10:37:33
    size_t local_work_size=workgroup_blocksize; // work group size is defined by RD_WG_SIZE_1 or RD_WG_SIZE_1_0 2014/06/10 17:00:41
    if(global_work[0]%local_work_size !=0)
        global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;

    err = clEnqueueWriteBuffer(cmd_queue, d_cluster, 1, 0, n_clusters * n_features * sizeof(float), clusters[0], 0, 0, 0);
    if(err != CL_SUCCESS) {
        printf("ERROR: clEnqueueWriteBuffer d_cluster (size:%d) => %d\n", n_points, err);
        return -1;
    }


    //long long start_time = get_time();

    int size = 0;
    int offset = 0;

    clSetKernelArg(kernel_s, 0, sizeof(void *), (void*) &d_feature_swap);
    clSetKernelArg(kernel_s, 1, sizeof(void *), (void*) &d_cluster);
    clSetKernelArg(kernel_s, 2, sizeof(void *), (void*) &d_membership);
    clSetKernelArg(kernel_s, 3, sizeof(cl_int), (void*) &n_points);
    clSetKernelArg(kernel_s, 4, sizeof(cl_int), (void*) &n_clusters);
    clSetKernelArg(kernel_s, 5, sizeof(cl_int), (void*) &n_features);
    clSetKernelArg(kernel_s, 6, sizeof(cl_int), (void*) &offset);
    clSetKernelArg(kernel_s, 7, sizeof(cl_int), (void*) &size);

	*kernel_time = mysecond();
    err = clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, &local_work_size, 0, 0, 0);
    if(err != CL_SUCCESS) {
        printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err);
        return -1;
    }
    clFinish(cmd_queue);
    err = clEnqueueReadBuffer(cmd_queue, d_membership, 1, 0, n_points * sizeof(int), membership_OCL, 0, 0, 0);
    if(err != CL_SUCCESS) {
        printf("ERROR: Memcopy Out\n");
        return -1;
    }
	*kernel_time = mysecond() - *kernel_time;

    //long long end_time = get_time();
    //double kernel_time = (float)(end_time - start_time)/(1000*1000);
    //double flops = n_points * n_clusters * n_features * 3;
    //printf("Kernel time: %f\n",kernel_time);
    //printf("FLOPS: %f",flops/kernel_time);

    delta = 0;
    for (i = 0; i < n_points; i++)
    {
        int cluster_id = membership_OCL[i];
        new_centers_len[cluster_id]++;
        if (membership_OCL[i] != membership[i])
        {
            delta++;
            membership[i] = membership_OCL[i];
        }
        for (j = 0; j < n_features; j++)
        {
            new_centers[cluster_id][j] += feature[i][j];
        }
    }

    return delta;
}
