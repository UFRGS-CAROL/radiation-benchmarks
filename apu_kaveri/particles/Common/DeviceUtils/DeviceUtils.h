/*
		2011 Takahiro Harada
*/
#ifndef DX11UTILS_H
#define DX11UTILS_H

struct DeviceDataBase
{
	enum Type
	{
		TYPE_CL, 
		TYPE_DX11,
		TYPE_CPU,
	};

	DeviceDataBase( Type type ) : m_type( type ) {}
	virtual ~DeviceDataBase(){}
	
	Type m_type;
};

struct DeviceBufferBase
{
	virtual ~DeviceBufferBase(){}

	enum Type
	{
		BUFFER = 1,
		BUFFER_STAGING = BUFFER | (1<<1),
		BUFFER_APPEND = BUFFER | (1<<2),
		BUFFER_CONST = BUFFER | (1<<3),
		BUFFER_RAW = BUFFER | (1<<4),
		BUFFER_W_COUNTER = BUFFER | (1<<5),
		BUFFER_INDEX = BUFFER | (1<<6),
		BUFFER_VERTEX = BUFFER | (1<<7),
		BUFFER_CPU_GPU = BUFFER | (1<<8), // For Fusion
	};
};

struct DeviceKernel
{
	virtual ~DeviceKernel(){}

};


template<typename DEVICEBUFFER>
class DeviceUtilsBase
{
	public:
		enum DriverType
		{
			DRIVER_HARDWARE = 0,
			DRIVER_REFERENCE,
		};

		__inline
		static int getNumDevices();

		__inline
		static void initDevice( DeviceDataBase* deviceData, DriverType driverType = DRIVER_HARDWARE, int deviceIdx = 0 );

		__inline
		static void releaseDevice( DeviceDataBase* deviceData );


		//	on CL, non blocking.
		//	on DX11, blocking. 
		template<typename T>
		__inline
		static void createDeviceBuffer( const DeviceDataBase* deviceData, int numElems, DEVICEBUFFER& deviceBuffer,
								DeviceBufferBase::Type type = DeviceBufferBase::BUFFER);
//impl
		__inline
		static void deleteDeviceBuffer( const DeviceDataBase* deviceData, DEVICEBUFFER& deviceBuffer );

		template<typename T>
		__inline
		static void writeDataToDevice( const DeviceDataBase* deviceData, int numElems, DEVICEBUFFER& deviceBuffer, const void* hostPtr, int offsetNumElems = 0 );

		template<typename T>
		__inline
		static void readDataFromDevice( const DeviceDataBase* deviceData, int numElems, DEVICEBUFFER& deviceBuffer, void* hostPtr, DEVICEBUFFER* stagingBuffer11, int offsetNumElems = 0 );

		__inline
		static void waitForCompletion( const DeviceDataBase* deviceData );
};

template<typename KERNEL>
class KernelBuilder
{
	public:
		__inline
		KernelBuilder( const DeviceDataBase* deviceData, char* fileName, const char* option = NULL, bool addExtension = false );

		__inline
		void createKernel( const char* funcName, KERNEL& kernelOut );

		static
		__inline
		void createKernel( const DeviceDataBase* deviceData, const char* funcName, KERNEL& kernelOut, const char* shader, int size );

		__inline
		~KernelBuilder();

//impl
		__inline
		static void deleteKernel( const DeviceDataBase* deviceData, KERNEL& kernel );

	private:
		enum
		{
			MAX_PATH_LENGTH = 260,
		};
		const DeviceDataBase* m_deviceData;
#ifdef UNICODE
		wchar_t m_path[MAX_PATH_LENGTH];
#else
		char m_path[MAX_PATH_LENGTH];
#endif
		void* m_ptr;
};

template<typename DEVICEBUFFER, typename KERNEL>
class KernelLauncher
{
	public:
		__inline
		KernelLauncher( const DeviceDataBase* deviceData, KERNEL& kernel );

		__inline
		~KernelLauncher();

		__inline
		void pushBackR( const DEVICEBUFFER& buffer );

		__inline
		void pushBackRW( DEVICEBUFFER& buffer, const int* counterInitValues = NULL );

		template<typename T>
		__inline
		void setConst( const DEVICEBUFFER& buffer, const T* hostData );

		__inline
		void launch1D( int numThreads, int localSize = 64 );

		__inline
		void launch2D( int numThreadsX, int numThreadsY, int localSizeX = 8, int localSizeY = 8 );

		__inline
		void launch1DOnDevice( DEVICEBUFFER& numElemsBuffer, u32 alignedOffset, int localSize = 64 );

	private:
		enum
		{
			COMMAND_BUFFER_SIZE = 0, 
		};
		const DeviceDataBase* m_deviceData;
		KERNEL* m_kernel;
		int m_idx;
		int m_idxRW;
};


#include <Common/DeviceUtils/DeviceUtilsCL.inl>

#endif


#include <Common/DeviceUtils/DeviceUtilsSelector.h>
