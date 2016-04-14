/*
		2011 Takahiro Harada
*/
#ifndef DEVICE_SELECTOR_H
#define DEVICE_SELECTOR_H

#include <Common/DeviceUtils/DeviceUtils.h>

//#if defined(COMPUTE_DEVICE_CL)
	#define DeviceData DeviceDataCL
	#define DeviceBuffer DeviceBufferCL
	#define DeviceUtils DeviceUtilsCL
	#define DeviceKernelBuilder KernelBuilderCL
	#define DeviceKernelLauncher KernelLauncherCL
	#define DeviceKernel DeviceKernelCL
//#endif


template<typename BUFFER, typename KERNEL>
class DUtilsBase : public DeviceUtilsBase<BUFFER>
{
public:
	__inline
	static DeviceDataBase* createDeviceData();

	class Builder : public KernelBuilder<KERNEL>
	{
	public:
		Builder( const DeviceDataBase* deviceData, char* fileName, const char* option = NULL, bool addExtension = false ): KernelBuilder<KERNEL>(deviceData, fileName, option, addExtension){}
	};

	class Launcher : public KernelLauncher<BUFFER, KERNEL>
	{
	public:
		Launcher( const DeviceDataBase* deviceData, KERNEL& kernel ):KernelLauncher<BUFFER, KERNEL>( deviceData, kernel ){}
	};
};

template<>
DeviceDataBase* DUtilsBase<DeviceBufferCL, DeviceKernelCL>::createDeviceData()
{
	DeviceDataBase* dd = new DeviceDataCL;
	return dd;
}

#define COMPUTE_DX11 \
	typedef StopwatchDx11 DStopwatch; \
	typedef DeviceBufferDX11 DBuffer; \
	typedef DeviceKernelDX11 DKernel; \
	typedef MapBufferDX11 DMapBuffer; \
	typedef DUtilsBase<DeviceBufferDX11, DKernel> DUtils; \
	typedef DUtils::Builder DKernelBuilder; \
	typedef DUtils::Launcher DKernelLauncher; \
	void INITIALIZE_DEVICE_DATA( const DeviceDataBase* deviceData, Demo* demo ){ demo->m_deviceData = deviceData; demo->m_ddCreated = true; \
		if( deviceData ) if( deviceData->m_type == DeviceData::TYPE_DX11 ) demo->m_ddCreated = false; \
		if( demo->m_ddCreated ) \
		{ \
			demo->m_deviceData = DUtils::createDeviceData(); \
			DUtils::initDevice( (DeviceDataBase*)demo->m_deviceData ); \
		} } \
	void DESTROY_DEVICE_DATA( Demo* demo ){ if( demo->m_ddCreated ) delete demo->m_deviceData; m_ddCreated=false; }

#define COMPUTE_CL \
	typedef DeviceBufferCL DBuffer; \
	typedef DeviceKernelCL DKernel; \
	typedef MapBufferCL DMapBuffer; \
	typedef DUtilsBase<DeviceBufferCL, DKernel> DUtils; \
	typedef DUtils::Builder DKernelBuilder; \
	typedef DUtils::Launcher DKernelLauncher; \
	void INITIALIZE_DEVICE_DATA( const DeviceDataBase* deviceData, Demo* demo ){ demo->m_deviceData = deviceData; demo->m_ddCreated = true; \
		if( deviceData ) if( deviceData->m_type == DeviceData::TYPE_CL ) demo->m_ddCreated = false; \
		if( demo->m_ddCreated ) \
		{ \
			demo->m_deviceData = DUtils::createDeviceData(); \
			DUtils::initDevice( (DeviceDataBase*)demo->m_deviceData ); \
		} } \
	void DESTROY_DEVICE_DATA( Demo* demo ){ if( demo->m_ddCreated ) delete demo->m_deviceData; m_ddCreated=false; }



#endif
