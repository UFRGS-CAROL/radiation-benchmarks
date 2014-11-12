#ifndef _SORT_H
#define _SORT_H

cl_platform_id          platform_id[100];
cl_device_id            device_id[100];
cl_context              context;
cl_command_queue        command_queue[2];
cl_kernel reduce;
cl_kernel top_scan;
cl_kernel bottom_scan;
cl_kernel cpu_reduce;
cl_kernel cpu_top_scan;
cl_kernel cpu_bottom_scan;

#define CL_CHECK_ERROR(err) \
	{ \
		if (err != CL_SUCCESS) \
		std::cerr << "Error: " \
		<< CLErrorString(err) \
		<< " in " << __FILE__ \
		<< " line " << __LINE__ \
		<< std::endl; \
	}

inline const char *CLErrorString(cl_int err) {
    switch (err) {
        // break;
    case CL_SUCCESS:
        return "CL_SUCCESS";
        // break;
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
        // break;
    case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
        // break;
    case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
        // break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        // break;
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
        // break;
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
        // break;
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
        // break;
    case CL_MEM_COPY_OVERLAP:
        return "CL_MEM_COPY_OVERLAP";
        // break;
    case CL_IMAGE_FORMAT_MISMATCH:
        return "CL_IMAGE_FORMAT_MISMATCH";
        // break;
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        // break;
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
        // break;
    case CL_MAP_FAILURE:
        return "CL_MAP_FAILURE";
        // break;
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
        // break;
    case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
        // break;
    case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
        // break;
    case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
        // break;
    case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
        // break;
    case CL_INVALID_QUEUE_PROPERTIES:
        return "CL_INVALID_QUEUE_PROPERTIES";
        // break;
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
        // break;
    case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
        // break;
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
        // break;
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        // break;
    case CL_INVALID_IMAGE_SIZE:
        return "CL_INVALID_IMAGE_SIZE";
        // break;
    case CL_INVALID_SAMPLER:
        return "CL_INVALID_SAMPLER";
        // break;
    case CL_INVALID_BINARY:
        return "CL_INVALID_BINARY";
        // break;
    case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
        // break;
    case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
        // break;
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
        // break;
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
        // break;
    case CL_INVALID_KERNEL_DEFINITION:
        return "CL_INVALID_KERNEL_DEFINITION";
        // break;
    case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
        // break;
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
        // break;
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
        // break;
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
        // break;
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
        // break;
    case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
        // break;
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
        // break;
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
        // break;
    case CL_INVALID_GLOBAL_OFFSET:
        return "CL_INVALID_GLOBAL_OFFSET";
        // break;
    case CL_INVALID_EVENT_WAIT_LIST:
        return "CL_INVALID_EVENT_WAIT_LIST";
        // break;
    case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
        // break;
    case CL_INVALID_OPERATION:
        return "CL_INVALID_OPERATION";
        // break;
    case CL_INVALID_GL_OBJECT:
        return "CL_INVALID_GL_OBJECT";
        // break;
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
        // break;
    case CL_INVALID_MIP_LEVEL:
        return "CL_INVALID_MIP_LEVEL";
        // break;
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
        // break;
    case CL_INVALID_PROPERTY:
        return "CL_INVALID_PROPERTY";
        // break;
    default:
        return "UNKNOWN";
    }
}


// Returns the current system time in microseconds
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}


void initOpenCL() {
    cl_uint         platforms_n = 0;
    cl_uint         devices_n   = 0;
    cl_int                  ret;

    clGetPlatformIDs(100, platform_id, &platforms_n);
    /*
    CL_DEVICE_TYPE_DEFAULT
    CL_DEVICE_TYPE_CPU
    CL_DEVICE_TYPE_GPU
    CL_DEVICE_TYPE_ACCELERATOR
    CL_DEVICE_TYPE_ALL
    */
    clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_ALL, 100, device_id, &devices_n);

    // Create an OpenCL context.
    context = clCreateContext(NULL, devices_n, device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf("\nError at clCreateContext! Error code %i\n\n", ret);
        exit(1);
    }

    // Create a command queue.
    command_queue[0] = clCreateCommandQueue(context, device_id[0], 0, &ret);
    if (ret != CL_SUCCESS) {
        printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
        exit(1);
    }

    // Create a command queue.
    command_queue[1] = clCreateCommandQueue(context, device_id[1], 0, &ret);
    if (ret != CL_SUCCESS) {
        printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
        exit(1);
    }


    int err = 0;

    FILE*  theFile = fopen("sort.cl", "r");
    if (!theFile) {
        fprintf(stderr, "Failed to load kernel file.\n");
        exit(1);
    }
    char* source_str;
    // Obtain length of source file.
    fseek(theFile, 0, SEEK_END);
    size_t source_size = ftell(theFile);
    rewind(theFile);
    // Read in the file.
    source_str = (char*) malloc(sizeof(char) * source_size);
    fread(source_str, 1, source_size, theFile);
    fclose(theFile);
    source_str[source_size] = '\0';
    // Program Setup
    cl_program gpuprog = clCreateProgramWithSource(context,
                         1,
                         (const char **) &source_str,
                         NULL,
                         &err);
    CL_CHECK_ERROR(err);
    cl_program cpuprog = clCreateProgramWithSource(context,
                         1,
                         (const char **) &source_str,
                         NULL,
                         &err);
    CL_CHECK_ERROR(err);

    err = clBuildProgram(gpuprog, 1, &device_id[0], NULL, NULL, NULL);
    CL_CHECK_ERROR(err);
    if (err != CL_SUCCESS) {
        printf("Error compiling sort kernels\n");
        exit(0);
    }
    err = clBuildProgram(cpuprog, 1, &device_id[1], NULL, NULL, NULL);
    CL_CHECK_ERROR(err);
    if (err != CL_SUCCESS) {
        printf("Error compiling sort kernels\n");
        exit(0);
    }

    reduce = clCreateKernel(gpuprog, "reduce", &err);
    CL_CHECK_ERROR(err);
    top_scan = clCreateKernel(gpuprog, "top_scan", &err);
    CL_CHECK_ERROR(err);
    bottom_scan = clCreateKernel(gpuprog, "bottom_scan", &err);
    CL_CHECK_ERROR(err);
    cpu_reduce = clCreateKernel(cpuprog, "reduce", &err);
    CL_CHECK_ERROR(err);
    cpu_top_scan = clCreateKernel(cpuprog, "top_scan", &err);
    CL_CHECK_ERROR(err);
    cpu_bottom_scan = clCreateKernel(cpuprog, "bottom_scan", &err);
    CL_CHECK_ERROR(err);
}






#endif // _SORT_H
