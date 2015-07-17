

struct DeviceDataCpu : public DeviceDataBase
{
	DeviceDataCpu() : DeviceDataBase( TYPE_CPU ) {}

};

struct DeviceBufferCpu : public DeviceBufferBase
{
	DeviceBufferCpu() : m_buffer(0), m_size(0) {}

	void* m_buffer;
	u32 m_size;
};

struct DeviceKernelCpu : public DeviceKernel
{
	enum
	{
		KERNEL_NAME_SIZE = 128, 
		KERNEL_MAX_ARGS = 32,
	};

	void* m_code;
	void* m_args[KERNEL_MAX_ARGS];
	int m_nArgs;
};

//--
struct KernelCpuPtr
{
	KernelCpuPtr(){}

	KernelCpuPtr(void* pCode, char* pName)
	{
		m_code = pCode;
		m_name = pName;
	}

	void* m_code;
	char* m_name;
};

#define PX_MAX_NUM_KERNELS 64

extern int g_numRegisterdKernels;
extern KernelCpuPtr g_kernelPtr[PX_MAX_NUM_KERNELS];


#define KERNELCPU_REGISTER(kernelName) g_kernelPtr[g_numRegisterdKernels++] = KernelCpuPtr((void*)kernelName, #kernelName);


typedef void (*KernelPtr1)(void*);
typedef void (*KernelPtr2)(void*, void*);
typedef void (*KernelPtr3)(void*, void*, void*);
typedef void (*KernelPtr4)(void*, void*, void*, void*);
typedef void (*KernelPtr5)(void*, void*, void*, void*, void*);
typedef void (*KernelPtr6)(void*, void*, void*, void*, void*, void*);
typedef void (*KernelPtr7)(void*, void*, void*, void*, void*, void*, void*);
typedef void (*KernelPtr8)(void*, void*, void*, void*, void*, void*, void*, void*);
typedef void (*KernelPtr9)(void*, void*, void*, void*, void*, void*, void*, void*, void*);
typedef void (*KernelPtr10)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);

typedef void (*KernelExecute)( DeviceKernelCpu* kernel );


static void KernelExecute0( DeviceKernelCpu* kernel )
{
	CLASSERT(0);
}
static void KernelExecute1( DeviceKernelCpu* kernel )
{
	KernelPtr1 func = (KernelPtr1)(kernel->m_code);
	func( kernel->m_args[0] );
}
static void KernelExecute2( DeviceKernelCpu* kernel )
{
	KernelPtr2 func = (KernelPtr2)(kernel->m_code);
	func( kernel->m_args[0], kernel->m_args[1] );
}
static void KernelExecute3( DeviceKernelCpu* kernel )
{
	KernelPtr3 func = (KernelPtr3)(kernel->m_code);
	func( kernel->m_args[0], kernel->m_args[1], kernel->m_args[2] );
}
static void KernelExecute4( DeviceKernelCpu* kernel )
{
	KernelPtr4 func = (KernelPtr4)(kernel->m_code);
	func( kernel->m_args[0], kernel->m_args[1], kernel->m_args[2], kernel->m_args[3] );
}

static void KernelExecute5( DeviceKernelCpu* kernel )
{
	KernelPtr5 func = (KernelPtr5)(kernel->m_code);
	func( kernel->m_args[0], kernel->m_args[1], kernel->m_args[2], kernel->m_args[3] , kernel->m_args[4] );
}

static void KernelExecute6( DeviceKernelCpu* kernel )
{
	KernelPtr6 func = (KernelPtr6)(kernel->m_code);
	func( kernel->m_args[0], kernel->m_args[1], kernel->m_args[2], kernel->m_args[3], kernel->m_args[4], kernel->m_args[5] );
}

static void KernelExecute7( DeviceKernelCpu* kernel )
{
	KernelPtr7 func = (KernelPtr7)(kernel->m_code);
	func( kernel->m_args[0], kernel->m_args[1], kernel->m_args[2], kernel->m_args[3], kernel->m_args[4], kernel->m_args[5], kernel->m_args[6] );
}

static void KernelExecute8( DeviceKernelCpu* kernel )
{
	KernelPtr8 func = (KernelPtr8)(kernel->m_code);
	func( kernel->m_args[0], kernel->m_args[1], kernel->m_args[2], kernel->m_args[3], kernel->m_args[4], kernel->m_args[5], kernel->m_args[6], kernel->m_args[7] );
}

static void KernelExecute9( DeviceKernelCpu* kernel )
{
	KernelPtr9 func = (KernelPtr9)(kernel->m_code);
	func( kernel->m_args[0], kernel->m_args[1], kernel->m_args[2], kernel->m_args[3], kernel->m_args[4], kernel->m_args[5], kernel->m_args[6], kernel->m_args[7], kernel->m_args[8] );
}

extern KernelExecute g_kernelPtrs[];

//--

extern __declspec(thread) int g_iThread;
extern __declspec(thread) int g_jThread;
extern __declspec(thread) int g_kThread;

extern __declspec(thread) int g_iThreadLocal;
extern __declspec(thread) int g_jThreadLocal;
extern __declspec(thread) int g_kThreadLocal;


#define get_global_id(x) (x==0)?g_iThread:(x==1)?g_jThread:g_kThread
#define get_local_id(x) (x==0)?g_iThreadLocal:(x==1)?g_jThreadLocal:g_kThreadLocal
#define __global
#define __local
#define __kernel
#define barrier(x) CLASSERT(0)
#define __constant


typedef DeviceUtilsBase<DeviceBufferCpu> DeviceUtilsCpu;
typedef KernelBuilder<DeviceKernelCpu> KernelBuilderCpu;
typedef KernelLauncher<DeviceBufferCpu, DeviceKernelCpu> KernelLauncherCpu;



template<>
void DeviceUtilsCpu::initDevice( DeviceDataBase* deviceData, DriverType driverType, int deviceIdx )
{
	CLASSERT( deviceData->m_type == DeviceDataBase::TYPE_CPU );

}

template<>
void DeviceUtilsCpu::releaseDevice( DeviceDataBase* deviceData )
{
	CLASSERT( deviceData->m_type == DeviceDataBase::TYPE_CPU );

}

template<>
template<typename T>
void DeviceUtilsCpu::createDeviceBuffer( const DeviceDataBase* deviceData, int numElems, DeviceBufferCpu& deviceBuffer,
								DeviceBufferBase::Type type)
{
	CLASSERT( deviceData->m_type == DeviceDataBase::TYPE_CPU );
	deviceBuffer.m_buffer = 0;
	if( type == DeviceBufferBase::BUFFER_STAGING || type == DeviceBufferBase::BUFFER_CONST ) return;

	deviceBuffer.m_size = sizeof(T)*numElems;
	deviceBuffer.m_buffer = _aligned_malloc( deviceBuffer.m_size, 16 );
	for(int i=0; i<numElems; i++)
	{
		T* ptr = addByteOffset<T>(deviceBuffer.m_buffer, i*sizeof(T));
		new(ptr)T;
	}

}


template<>
void DeviceUtilsCpu::deleteDeviceBuffer( const DeviceDataBase* deviceData, DeviceBufferCpu& deviceBuffer )
{
	CLASSERT( deviceData->m_type == DeviceDataBase::TYPE_CPU );
	if( deviceBuffer.m_buffer == 0 ) return;

	_aligned_free( deviceBuffer.m_buffer );
	deviceBuffer.m_buffer = 0;
	deviceBuffer.m_size = 0;
}

template<>
template<typename T>
void DeviceUtilsCpu::writeDataToDevice( const DeviceDataBase* deviceData, int numElems, DeviceBufferCpu& deviceBuffer, const void* hostPtr, int offsetNumElems )
{
	CLASSERT( deviceData->m_type == DeviceDataBase::TYPE_CPU );
	CLASSERT( deviceBuffer.m_size <= sizeof(T)*numElems );
	memcpy( addByteOffset<T>( deviceBuffer.m_buffer, sizeof(T)*offsetNumElems ), hostPtr, sizeof(T)*numElems );
}

template<>
template<typename T>
void DeviceUtilsCpu::readDataFromDevice( const DeviceDataBase* deviceData, int numElems, DeviceBufferCpu& deviceBuffer, void* hostPtr, DeviceBufferCpu* stagingBuffer11, int offsetNumElems )
{
	CLASSERT( deviceData->m_type == DeviceDataBase::TYPE_CPU );
	CLASSERT( deviceBuffer.m_size <= sizeof(T)*numElems );
	memcpy( addByteOffset<T>( hostPtr, sizeof(T)*offsetNumElems ), deviceBuffer.m_buffer, sizeof(T)*numElems );
}

template<>
void DeviceUtilsCpu::waitForCompletion( const DeviceDataBase* deviceData )
{
	CLASSERT( deviceData->m_type == DeviceDataBase::TYPE_CPU );

}

template<>
KernelBuilderCpu::KernelBuilder( const DeviceDataBase* deviceData, char* fileName, const char* option, bool addExtension )
//:m_deviceData( deviceData )
{
	if( addExtension )
	{
		CLASSERT(0);
	}
	CLASSERT( deviceData->m_type == DeviceDataBase::TYPE_CPU );
	m_deviceData = (DeviceDataCpu*)deviceData;

}

template<>
void KernelBuilderCpu::createKernel( const char* funcName, DeviceKernelCpu& kernelOut )
{
	for(int i=0; i<g_numRegisterdKernels; i++)
	{
		if( strcmp(funcName, g_kernelPtr[i].m_name ) == 0 )
		{
			kernelOut.m_code = g_kernelPtr[i].m_code;
			return;
		}
	}
	CLASSERT(0);
}

template<>
KernelBuilderCpu::~KernelBuilderCpu()
{

}

template<>
void KernelBuilderCpu::deleteKernel( const DeviceDataBase* deviceData, DeviceKernelCpu& kernel )
{
	CLASSERT( deviceData->m_type == DeviceDataBase::TYPE_CPU );

}

template<>
KernelLauncherCpu::KernelLauncher( const DeviceDataBase* deviceData, DeviceKernelCpu& kernel )
{
	CLASSERT( deviceData->m_type == DeviceDataBase::TYPE_CPU );
	m_kernel = &kernel;
	m_kernel->m_nArgs = 0;
}

template<>
KernelLauncherCpu::~KernelLauncher()
{

}

template<>
void KernelLauncherCpu::pushBackR( const DeviceBufferCpu& buffer )
{
	m_kernel->m_args[ m_kernel->m_nArgs++ ] = buffer.m_buffer;
}

template<>
void KernelLauncherCpu::pushBackRW( DeviceBufferCpu& buffer, const int* counterInitValues )
{
	m_kernel->m_args[ m_kernel->m_nArgs++ ] = buffer.m_buffer;
}

//	todo. better to copy host data to buffer, then set arg
template<>
template<typename T>
void KernelLauncherCpu::setConst( const DeviceBufferCpu& buffer, const T* hostData )
{
	m_kernel->m_args[ m_kernel->m_nArgs++ ] = (void*)hostData;
}

template<>
void KernelLauncherCpu::launch1D( int numThreads, int localSize )
{
	g_iThread = g_jThread = g_kThread = 0;

	int gRange[3] = {1,1,1};
	int lRange[3] = {1,1,1};
	lRange[0] = localSize;
	gRange[0] = max2(1, (numThreads/lRange[0])+(!(numThreads%lRange[0])?0:1));

#pragma omp parallel for
	for(int iblock=0; iblock<gRange[0]; iblock++)
	{
		g_iThread = iblock*localSize;
		for(g_iThreadLocal=0; g_iThreadLocal<localSize; g_iThreadLocal++)
		{
			g_kernelPtrs[ m_kernel->m_nArgs ]( m_kernel );
			g_iThread++;
		}
	}
}

template<>
void KernelLauncherCpu::launch2D( int numThreadsX, int numThreadsY, int localSizeX, int localSizeY )
{
//	int g_iThreadLocal, g_jThreadLocal;

	int gRange[3] = {1,1,1};
	int lRange[3] = {1,1,1};
	lRange[0] = localSizeX;
	lRange[1] = localSizeY;
	gRange[0] = max2(1, (numThreadsX/lRange[0])+(!(numThreadsX%lRange[0])?0:1));
	gRange[1] = max2(1, (numThreadsY/lRange[1])+(!(numThreadsY%lRange[1])?0:1));

/*
	for(int iblock=0; iblock<gRange[0]; iblock++) 
	{
		for(int jblock=0; jblock<gRange[1]; jblock++)
		{
			for(g_iThreadLocal=0; g_iThreadLocal<localSizeX; g_iThreadLocal++)
			{
				for(g_jThreadLocal=0; g_jThreadLocal<localSizeY; g_jThreadLocal++)
				{
					g_kernelPtrs[ m_kernel->m_nArgs ]( m_kernel );
				}
			}
		}
	}
*/
#pragma omp parallel for
	for(int bIdx=0; bIdx<gRange[0]*gRange[1]; bIdx++)
	{
		int iBlock = bIdx%gRange[0];
		int jBlock = bIdx/gRange[0];

		for(g_iThreadLocal=0, g_iThread = iBlock*localSizeX; g_iThreadLocal<localSizeX; g_iThreadLocal++, g_iThread++)
		{
			for(g_jThreadLocal=0, g_jThread = jBlock*localSizeY; g_jThreadLocal<localSizeY; g_jThreadLocal++, g_jThread++)
			{
				g_kernelPtrs[ m_kernel->m_nArgs ]( m_kernel );
			}
		}
	}
}

template<>
void KernelLauncherCpu::launch1DOnDevice( DeviceBufferCpu& numElemsBuffer, u32 alignedOffset, int localSize )
{

}







