/*
		2011 Takahiro Harada
*/
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>

#include <stdio.h>
#include <fstream>

struct DeviceDataCL : public DeviceDataBase
{
	DeviceDataCL() : DeviceDataBase( TYPE_CL ), m_context(0), m_commandQueue(0){}
	enum
	{
		MAX_NUM_DEVICES = 2,
	};
	
	union
	{
		cl_context m_context;
		cl_context m_contexts[MAX_NUM_DEVICES];
	};

	union
	{
		cl_command_queue m_commandQueue;
		cl_command_queue m_commandQueues[MAX_NUM_DEVICES];
	};
	cl_device_id m_deviceIds[MAX_NUM_DEVICES];
};

struct MapBufferCL
{
	MapBufferCL() : m_ptr(0) {}
	template<typename T>
	T* getPtr(){ return (T*)m_ptr;}

	void* m_ptr;
};

struct DeviceBufferCL : public DeviceBufferBase
{
	DeviceBufferCL() : m_buffer(0) {}

	cl_mem m_buffer;

	template<typename T>
	__inline
	void allocate(const DeviceDataBase* deviceData, int numElems, 
		DeviceBufferBase::Type type = DeviceBufferBase::BUFFER);
	__inline
	void deallocate(const DeviceDataBase* deviceData);
	template<typename T>
	__inline
	void write(const DeviceDataBase* deviceData, int numElems, const void* hostPtr, int offsetNumElems = 0);
	template<typename T>
	__inline
	void read(const DeviceDataBase* deviceData, int numElems, void* hostPtr, DeviceBufferCL* stagingBuffer11, int offsetNumElems = 0);

	template<typename T>
	__inline
	void map(const DeviceDataBase* deviceData, int numElems, MapBufferCL& mappedBuffer, DeviceBufferCL* stagingBuffer);
	__inline
	void unmap(const DeviceDataBase* deviceData, int numElems, MapBufferCL& mappedBuffer, DeviceBufferCL* stagingBuffer);
};

struct DeviceKernelCL : public DeviceKernel
{
	DeviceKernelCL() : m_kernel(0) {}

	cl_kernel m_kernel;
};

typedef DeviceUtilsBase<DeviceBufferCL> DeviceUtilsCL;
typedef KernelBuilder<DeviceKernelCL> KernelBuilderCL;
typedef KernelLauncher<DeviceBufferCL, DeviceKernelCL> KernelLauncherCL;

template<typename T>
void DeviceBufferCL::allocate(const DeviceDataBase* deviceData, int numElems, 
	DeviceBufferBase::Type type)
{
	DeviceUtilsCL::createDeviceBuffer<T>( deviceData, numElems, *this, type );
}

template<>
void DeviceUtilsCL::deleteDeviceBuffer( const DeviceDataBase* deviceDataBase, DeviceBufferCL& deviceBuffer )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;

	if( deviceBuffer.m_buffer == 0 ) return;

	if( deviceBuffer.m_buffer )
		clReleaseMemObject( deviceBuffer.m_buffer );
	deviceBuffer.m_buffer = 0;
}

void DeviceBufferCL::deallocate(const DeviceDataBase* deviceData)
{
	DeviceUtilsCL::deleteDeviceBuffer( deviceData, *this );
}

template<typename T>
void DeviceBufferCL::write(const DeviceDataBase* deviceData, int numElems, const void* hostPtr, int offsetNumElems)
{
	DeviceUtilsCL::writeDataToDevice<T>( deviceData, numElems, *this, hostPtr, offsetNumElems );
}

template<typename T>
void DeviceBufferCL::read(const DeviceDataBase* deviceData, int numElems, void* hostPtr, DeviceBufferCL* stagingBuffer11, int offsetNumElems)
{
	DeviceUtilsCL::readDataFromDevice<T>( deviceData, numElems, *this, hostPtr, stagingBuffer11, offsetNumElems );
}

template<typename T>
void DeviceBufferCL::map(const DeviceDataBase* deviceDataBase, int numElems, MapBufferCL& mappedBuffer, DeviceBufferCL* stagingBuffer)
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;

	cl_int e;
	mappedBuffer.m_ptr = clEnqueueMapBuffer( deviceData->m_commandQueue, m_buffer, false, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(T)*numElems, 0,0,0,&e );
	CLASSERT( e == CL_SUCCESS );
}

void DeviceBufferCL::unmap(const DeviceDataBase* deviceDataBase, int numElems, MapBufferCL& mappedBuffer, DeviceBufferCL* stagingBuffer)
{
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;
	clEnqueueUnmapMemObject( deviceData->m_commandQueue, m_buffer, mappedBuffer.m_ptr, 0,0,0 );
}

template<>
int DeviceUtilsCL::getNumDevices()
{
	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
	cl_int status;

	cl_platform_id platform;
	{
		cl_uint nPlatforms = 0;
		status = clGetPlatformIDs(0, NULL, &nPlatforms);

		cl_platform_id pIdx[5];
		status = clGetPlatformIDs(nPlatforms, pIdx, NULL);


		cl_uint nvIdx = -1;
		cl_uint atiIdx = -1;
		for(cl_uint i=0; i<nPlatforms; i++)
		{
			char buff[512];
			status = clGetPlatformInfo( pIdx[i], CL_PLATFORM_VENDOR, 512, buff, 0 );
			CLASSERT( status == CL_SUCCESS );

			if( strcmp( buff, "NVIDIA Corporation" )==0 ) nvIdx = i;
			if( strcmp( buff, "Advanced Micro Devices, Inc." )==0 ) atiIdx = i;
		}

		if( deviceType == CL_DEVICE_TYPE_GPU )
		{
			if( nvIdx != -1 ) platform = pIdx[nvIdx];
			else platform = pIdx[atiIdx];
		}
		else if( deviceType == CL_DEVICE_TYPE_CPU )
		{
			platform = pIdx[atiIdx];
		}
	}

	cl_uint numDevice;
	status = clGetDeviceIDs( platform, deviceType, 0, NULL, &numDevice );
	CLASSERT( status == CL_SUCCESS );

	return numDevice;
}

template<>
void DeviceUtilsCL::initDevice( DeviceDataBase* deviceDataBase, DriverType driverType, int deviceIdx )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;

	cl_device_type deviceType = (driverType == DRIVER_HARDWARE)? CL_DEVICE_TYPE_GPU:CL_DEVICE_TYPE_CPU;
	//cl_device_type deviceType = CL_DEVICE_TYPE_CPU;
	int numContextQueuePairsToCreate = 1;
	bool enableProfiling = false;
#ifdef _DEBUG
	enableProfiling = true;
#endif
	cl_int status;

	cl_platform_id platform;
	{
		cl_uint nPlatforms = 0;
		status = clGetPlatformIDs(0, NULL, &nPlatforms);
		CLASSERT( status == CL_SUCCESS );

		cl_platform_id pIdx[5];
		status = clGetPlatformIDs(nPlatforms, pIdx, NULL);
		CLASSERT( status == CL_SUCCESS );

		cl_uint nvIdx = -1;
		cl_uint atiIdx = -1;
		for(cl_uint i=0; i<nPlatforms; i++)
		{
			char buff[512];
			status = clGetPlatformInfo( pIdx[i], CL_PLATFORM_VENDOR, 512, buff, 0 );
			CLASSERT( status == CL_SUCCESS );

			if( strcmp( buff, "NVIDIA Corporation" )==0 ) nvIdx = i;
			if( strcmp( buff, "Advanced Micro Devices, Inc." )==0 ) atiIdx = i;
		}

		if( deviceType == CL_DEVICE_TYPE_GPU )
		{
			if( nvIdx != -1 ) platform = pIdx[nvIdx];
			else platform = pIdx[atiIdx];
		}
		else if( deviceType == CL_DEVICE_TYPE_CPU )
		{
			platform = pIdx[atiIdx];
		}
	}

	cl_uint numDevice;
	status = clGetDeviceIDs( platform, deviceType, 0, NULL, &numDevice );

	//printf("%d %s Devices found\n", numDevice, (deviceType==CL_DEVICE_TYPE_GPU)? "GPU":"CPU");

	numContextQueuePairsToCreate = min2((int)numDevice, numContextQueuePairsToCreate );

	status = clGetDeviceIDs( platform, deviceType, numContextQueuePairsToCreate, deviceData->m_deviceIds, NULL );
	CLASSERT( status == CL_SUCCESS );

	for(int i=0; i<numContextQueuePairsToCreate; i++)
	{
		deviceData->m_contexts[i] = clCreateContext( NULL, 1, &deviceData->m_deviceIds[i], NULL, NULL, &status );
		CLASSERT( status == CL_SUCCESS );

		char buff[512];
		status = clGetDeviceInfo( deviceData->m_deviceIds[i], CL_DEVICE_NAME, sizeof(buff), &buff, NULL );
		CLASSERT( status == CL_SUCCESS );

		//printf("Using %s\n", buff);

		deviceData->m_commandQueues[i] = clCreateCommandQueue( deviceData->m_contexts[i], deviceData->m_deviceIds[i], (enableProfiling)?CL_QUEUE_PROFILING_ENABLE:NULL, NULL );

		CLASSERT( status == CL_SUCCESS );

	//	status = clSetCommandQueueProperty( commandQueue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, 0 );
	//	CLASSERT( status == CL_SUCCESS );

		cl_bool image_support;
		clGetDeviceInfo(deviceData->m_deviceIds[0], CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
		//debugPrintf("	CL_DEVICE_IMAGE_SUPPORT : %s\n", image_support?"Yes":"No");
	}

//	return numContextQueuePairsToCreate;
}

template<>
void DeviceUtilsCL::releaseDevice( DeviceDataBase* deviceDataBase )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;

	clReleaseCommandQueue( deviceData->m_commandQueue );
	clReleaseContext( deviceData->m_context );
}

template<>
template<typename T>
void DeviceUtilsCL::createDeviceBuffer( const DeviceDataBase* deviceDataBase, int numElems, DeviceBufferCL& deviceBuffer,
								DeviceBufferBase::Type type)
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;

	deviceBuffer.m_buffer = 0;
	if( type == DeviceBufferBase::BUFFER_STAGING ) return;

	cl_int status = 0;
	if( type == DeviceBufferBase::BUFFER_CPU_GPU )
	{
		deviceBuffer.m_buffer = clCreateBuffer( deviceData->m_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
				sizeof(T)*numElems, 0, &status );
	}
	else
	{
		deviceBuffer.m_buffer = clCreateBuffer( deviceData->m_context, CL_MEM_READ_WRITE, 
				sizeof(T)*numElems, 0, &status );
	}
	CLASSERT( status == CL_SUCCESS );
}




template<>
template<typename T>
void DeviceUtilsCL::writeDataToDevice( const DeviceDataBase* deviceDataBase, int numElems, DeviceBufferCL& deviceBuffer, const void* hostPtr, int offsetNumElems )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;

	cl_int status = 0;
	status = clEnqueueWriteBuffer( deviceData->m_commandQueue, deviceBuffer.m_buffer, 0, sizeof(T)*offsetNumElems, sizeof(T)*numElems,
		hostPtr, 0,0,0 );
	CLASSERT( status == CL_SUCCESS );
}

template<>
template<typename T>
void DeviceUtilsCL::readDataFromDevice( const DeviceDataBase* deviceDataBase, int numElems, DeviceBufferCL& deviceBuffer, void* hostPtr, DeviceBufferCL* stagingBuffer11, int offsetNumElems )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;

	cl_int status = 0;
	status = clEnqueueReadBuffer( deviceData->m_commandQueue, deviceBuffer.m_buffer, 0, sizeof(T)*offsetNumElems, sizeof(T)*numElems, 
		hostPtr, 0,0,0 );
	CLASSERT( status == CL_SUCCESS );
}

template<>
void DeviceUtilsCL::waitForCompletion( const DeviceDataBase* deviceDataBase )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;

	clFinish( deviceData->m_commandQueue );
}

template<>
KernelBuilderCL::KernelBuilder( const DeviceDataBase* deviceDataBase, char* fileName, const char* option, bool addExtension )
{
	char fileNameWithExtension[256];

	if( addExtension )
		sprintf( fileNameWithExtension, "%s.cl", fileName );
	else
		sprintf( fileNameWithExtension, "%s", fileName );
	

	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;
	m_deviceData = deviceData;

		class File
		{
		public:
			__inline
			bool open(const char* fileNameWithExtension)
			{
				size_t      size;
				char*       str;

				// Open file stream
				std::fstream f(fileNameWithExtension, (std::fstream::in | std::fstream::binary));

				// Check if we have opened file stream
				if (f.is_open()) {
					size_t  sizeFile;
					// Find the stream size
					f.seekg(0, std::fstream::end);
					size = sizeFile = (size_t)f.tellg();
					f.seekg(0, std::fstream::beg);

					str = new char[size + 1];
					if (!str) {
						f.close();
						return  NULL;
					}

					// Read file
					f.read(str, sizeFile);
					f.close();
					str[size] = '\0';

					m_source  = str;

					delete[] str;

					return true;
				}

				return false;
			}
			const std::string& getSource() const {return m_source;}

		private:
			std::string m_source;
		};

	cl_program& program = (cl_program&)m_ptr;
    cl_int status = 0;
	File kernelFile;
	CLASSERT( kernelFile.open( fileNameWithExtension ) );
	const char* source = kernelFile.getSource().c_str();
	size_t sourceSize[] = {strlen(source)};
	program = clCreateProgramWithSource( deviceData->m_context, 1, &source, sourceSize, &status );
	CLASSERT( status == CL_SUCCESS );
	status = clBuildProgram( program, 1, deviceData->m_deviceIds, option, NULL, NULL );
	if( status != CL_SUCCESS )
	{
		char *build_log;
		size_t ret_val_size;
		clGetProgramBuildInfo(program, deviceData->m_deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
		build_log = new char[ret_val_size+1];
		clGetProgramBuildInfo(program, deviceData->m_deviceIds[0], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);

		build_log[ret_val_size] = '\0';

		printf("%s\n", build_log);

		delete build_log;
		CLASSERT(0);
	}
}

template<>
void KernelBuilderCL::createKernel( const char* funcName, DeviceKernelCL& kernelOut )
{
	cl_program program = (cl_program)m_ptr;
    cl_int status = 0;
	kernelOut.m_kernel = clCreateKernel(program, funcName, &status );
	CLASSERT( status == CL_SUCCESS );
}

template<>
KernelBuilderCL::~KernelBuilderCL()
{
	cl_program program = (cl_program)m_ptr;
	clReleaseProgram( program );
}

template<>
void KernelBuilderCL::deleteKernel( const DeviceDataBase* deviceDataBase, DeviceKernelCL& kernel )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;

	if( kernel.m_kernel )
	{
		clReleaseKernel( kernel.m_kernel );
		kernel.m_kernel = 0;
	}
}

template<>
KernelLauncherCL::KernelLauncher( const DeviceDataBase* deviceDataBase, DeviceKernelCL& kernel )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)deviceDataBase;

	m_deviceData = deviceData;
	m_kernel = &kernel;
	m_idx = 0;
	m_idxRW = 0;
}

template<>
KernelLauncherCL::~KernelLauncher()
{

}

template<>
void KernelLauncherCL::pushBackR( const DeviceBufferCL& buffer )
{
//	OpenCLUtils::setKernelArg( m_kernel, m_idx++, buffer->m_buffer );
	cl_int status = clSetKernelArg( m_kernel->m_kernel, m_idx++, sizeof(cl_mem), &buffer.m_buffer );
	CLASSERT( status == CL_SUCCESS );
}

template<>
void KernelLauncherCL::pushBackRW( DeviceBufferCL& buffer, const int* counterInitValues )
{
	pushBackR( buffer );
}

//	todo. better to copy host data to buffer, then set arg
template<>
template<typename T>
void KernelLauncherCL::setConst( const DeviceBufferCL& buffer, const T* hostData )
{
//	DeviceUtilsCL::writeDataToDevice<T>( m_deviceData, 1, (DeviceBufferCL&)buffer, hostData );
//	cl_int status = clSetKernelArg( m_kernel->m_kernel, m_idx++, sizeof(cl_mem), &buffer.m_buffer );
	cl_int status = clSetKernelArg( m_kernel->m_kernel, m_idx++, sizeof(T), hostData );
	CLASSERT( status == CL_SUCCESS );
}

template<>
void KernelLauncherCL::launch1D( int numThreads, int localSize )
{
	CLASSERT( m_deviceData->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)m_deviceData;

	static u32 s_nKernelLaunched;

	size_t gRange[3] = {1,1,1};
	size_t lRange[3] = {1,1,1};
	lRange[0] = localSize;
	gRange[0] = max2((size_t)1, (numThreads/lRange[0])+(!(numThreads%lRange[0])?0:1));
	gRange[0] *= lRange[0];

	cl_int status;

	if( COMMAND_BUFFER_SIZE == 0 )
	{
		status = clEnqueueNDRangeKernel( deviceData->m_commandQueue, 
			m_kernel->m_kernel, 1, NULL, gRange, lRange, 0,0,0 );
		CLASSERT( status == CL_SUCCESS );
	}
	else
	{
		if( s_nKernelLaunched%COMMAND_BUFFER_SIZE == COMMAND_BUFFER_SIZE-1 )
		{
			cl_event e;
			status = clEnqueueNDRangeKernel( deviceData->m_commandQueue, 
				m_kernel->m_kernel, 1, NULL, gRange, lRange, 0,0,&e );

			CLASSERT( status == CL_SUCCESS );
			clFlush( deviceData->m_commandQueue );

			cl_int er = clGetEventInfo(e, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, 0 );
			while( status > CL_SUBMITTED ) // split command buffer and send them here
			{
				er = clGetEventInfo(e, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, 0 );
			}
		}
		else
		{
			status = clEnqueueNDRangeKernel( deviceData->m_commandQueue, 
				m_kernel->m_kernel, 1, NULL, gRange, lRange, 0,0,0 );
			CLASSERT( status == CL_SUCCESS );
		}
	}

	s_nKernelLaunched++;
}

template<>
void KernelLauncherCL::launch2D( int numThreadsX, int numThreadsY, int localSizeX, int localSizeY )
{
	CLASSERT( m_deviceData->m_type == DeviceDataBase::TYPE_CL );
	DeviceDataCL* deviceData = (DeviceDataCL*)m_deviceData;

	size_t gRange[3] = {1,1,1};
	size_t lRange[3] = {1,1,1};
	lRange[0] = localSizeX;
	lRange[1] = localSizeY;
	gRange[0] = max2((size_t)1, (numThreadsX/lRange[0])+(!(numThreadsX%lRange[0])?0:1));
	gRange[0] *= lRange[0];
	gRange[1] = max2((size_t)1, (numThreadsY/lRange[1])+(!(numThreadsY%lRange[1])?0:1));
	gRange[1] *= lRange[1];

	cl_int status = clEnqueueNDRangeKernel( deviceData->m_commandQueue, 
		m_kernel->m_kernel, 2, NULL, gRange, lRange, 0,0,0 );
	CLASSERT( status == CL_SUCCESS );
}

template<>
void KernelLauncherCL::launch1DOnDevice( DeviceBufferCL& numElemsBuffer, u32 alignedOffset, int localSize )
{
	CLASSERT(0);
}

/*#include <windows.h>
#include <time.h>

struct StopwatchCL
{
	public:
		__inline
		StopwatchCL() : m_deviceData(0){}
		__inline
		StopwatchCL( const DeviceDataBase* deviceData );
		__inline
		~StopwatchCL();

		__inline
		void init( const DeviceDataBase* deviceData );
		__inline
		void start();
		__inline
		void split();
		__inline
		void stop();
		__inline
		float getMs();
		__inline
		void getMs( float* times, int capacity );
		__inline
		int getNIntervals() const { return m_idx-1; }

	public:
		enum
		{
			CAPACITY = 64,
		};
		const DeviceDataBase* m_deviceData;
		LARGE_INTEGER m_t[CAPACITY];
		int m_idx;
};

StopwatchCL::StopwatchCL(const DeviceDataBase *deviceData)
{
	init( deviceData );
}

StopwatchCL::~StopwatchCL()
{

}

void StopwatchCL::init( const DeviceDataBase* deviceData )
{
	m_deviceData = deviceData;
}

void StopwatchCL::start()
{
	m_idx = 0;
	split();
}

void StopwatchCL::split()
{
	DeviceUtilsCL::waitForCompletion( m_deviceData );
	QueryPerformanceCounter(&m_t[m_idx++]);
}

void StopwatchCL::stop()
{
	split();
}

float StopwatchCL::getMs()
{
	LARGE_INTEGER m_frequency;
	QueryPerformanceFrequency( &m_frequency );
	return (float)(1000*(m_t[1].QuadPart - m_t[0].QuadPart))/m_frequency.QuadPart;
}

void StopwatchCL::getMs( float* times, int capacity )
{
	LARGE_INTEGER m_frequency;
	QueryPerformanceFrequency( &m_frequency );

	for(int i=0; i<capacity; i++) times[i] = 0.f;

	for(int i=0; i<min2(capacity, m_idx); i++)
	{
		times[i] = (float)(1000*(m_t[i+1].QuadPart - m_t[i].QuadPart))/m_frequency.QuadPart;
	}
}*/
