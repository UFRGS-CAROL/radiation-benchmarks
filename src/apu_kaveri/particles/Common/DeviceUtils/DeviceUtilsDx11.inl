#include <windows.h>
#include <d3d11.h>
#include <d3dx11.h>
#include <d3dcompiler.h>
#pragma comment(lib,"d3dx11.lib")
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"dxgi.lib")


struct DeviceDataDX11 : public DeviceDataBase
{
	DeviceDataDX11() : DeviceDataBase( TYPE_DX11 ){}

	ID3D11DeviceContext* m_context;
	ID3D11Device* m_device;
	IDXGISwapChain* m_swapChain;
};

struct MapBufferDX11
{
	MapBufferDX11() { m_res.pData = NULL; }
	template<typename T>
	T* getPtr(){ return (T*)m_res.pData;}

	D3D11_MAPPED_SUBRESOURCE m_res;
};

struct DeviceBufferDX11 : public DeviceBufferBase
{
	DeviceBufferDX11() : m_buffer(0), m_uav(0), m_srv(0){}

	ID3D11Buffer* m_buffer;
	ID3D11UnorderedAccessView* m_uav;
	ID3D11ShaderResourceView* m_srv;
	DeviceBufferBase::Type m_type;

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
	void read(const DeviceDataBase* deviceData, int numElems, void* hostPtr, DeviceBufferDX11* stagingBuffer11, int offsetNumElems = 0);

	template<typename T>
	__inline
	void map(const DeviceDataBase* deviceData, int numElems, MapBufferDX11& mappedBuffer, DeviceBufferDX11* stagingBuffer);
	__inline
	void unmap(const DeviceDataBase* deviceData, int numElems, MapBufferDX11& mappedBuffer, DeviceBufferDX11* stagingBuffer);
};

struct DeviceKernelDX11 : public DeviceKernel
{
	DeviceKernelDX11() : m_kernel(0){}

	ID3D11ComputeShader* m_kernel;
};

typedef DeviceUtilsBase<DeviceBufferDX11> DeviceUtilsDX11;
typedef KernelBuilder<DeviceKernelDX11> KernelBuilderDX11;
typedef KernelLauncher<DeviceBufferDX11, DeviceKernelDX11> KernelLauncherDX11;

template<typename T>
void DeviceBufferDX11::allocate(const DeviceDataBase* deviceData, int numElems, 
	DeviceBufferBase::Type type)
{
	if( type == DeviceBufferBase::BUFFER_CPU_GPU ) 
	{
		CLASSERT( AMDExDx::g_mapperExDx );
		AMDExDx::g_mapperExDx->SetOverrideAccessFlags( D3D11_MAP_READ_WRITE );
		DeviceUtilsDX11::createDeviceBuffer<T>( deviceData, numElems, *this, DeviceBufferBase::BUFFER );
		m_type = DeviceBufferBase::BUFFER_CPU_GPU;
		AMDExDx::g_mapperExDx->SetOverrideAccessFlags( 0 );
	}
	else
	{
		DeviceUtilsDX11::createDeviceBuffer<T>( deviceData, numElems, *this, type );
	}
}

void DeviceBufferDX11::deallocate(const DeviceDataBase* deviceData)
{
	DeviceUtilsDX11::deleteDeviceBuffer( deviceData, *this );
}

template<typename T>
void DeviceBufferDX11::write(const DeviceDataBase* deviceData, int numElems, const void* hostPtr, int offsetNumElems)
{
	DeviceUtilsDX11::writeDataToDevice<T>( deviceData, numElems, *this, hostPtr, offsetNumElems );
}

template<typename T>
void DeviceBufferDX11::read(const DeviceDataBase* deviceData, int numElems, void* hostPtr, DeviceBufferDX11* stagingBuffer11, int offsetNumElems)
{
	DeviceUtilsDX11::readDataFromDevice<T>( deviceData, numElems, *this, hostPtr, stagingBuffer11, offsetNumElems );
}

template<typename T>
void DeviceBufferDX11::map(const DeviceDataBase* deviceDataBase, int numElems, MapBufferDX11& mappedBuffer, DeviceBufferDX11* stagingBuffer)
{
	CLASSERT(0);
}

void DeviceBufferDX11::unmap(const DeviceDataBase* deviceDataBase, int numElems, MapBufferDX11& mappedBuffer, DeviceBufferDX11* stagingBuffer)
{
	CLASSERT(0);
}



#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if(p) { (p)->Release(); (p)=NULL; } }
#endif

__inline
#ifdef UNICODE
HRESULT FindDXSDKShaderFileCch( __in_ecount(cchDest) WCHAR* strDestPath,
                                int cchDest, 
                                __in LPCWSTR strFilename )
#else
HRESULT FindDXSDKShaderFileCch( __in_ecount(cchDest) CHAR* strDestPath,
                                int cchDest, 
                                __in LPCSTR strFilename )
#endif
{
    if( NULL == strFilename || strFilename[0] == 0 || NULL == strDestPath || cchDest < 10 )
        return E_INVALIDARG;

    // Get the exe name, and exe path
#ifdef UNICODE
    WCHAR strExePath[MAX_PATH] =
#else
    CHAR strExePath[MAX_PATH] =
#endif
    {
        0
    };
#ifdef UNICODE
    WCHAR strExeName[MAX_PATH] =
#else
    CHAR strExeName[MAX_PATH] =
#endif
    {
        0
    };
#ifdef UNICODE
    WCHAR* strLastSlash = NULL;
#else
    CHAR* strLastSlash = NULL;
#endif
    GetModuleFileName( NULL, strExePath, MAX_PATH );
    strExePath[MAX_PATH - 1] = 0;
#ifdef UNICODE
    strLastSlash = wcsrchr( strExePath, TEXT( '\\' ) );
#else
    strLastSlash = strrchr( strExePath, TEXT( '\\' ) );
#endif
    if( strLastSlash )
    {
#ifdef UNICODE
        wcscpy_s( strExeName, MAX_PATH, &strLastSlash[1] );
#else

#endif
        // Chop the exe name from the exe path
        *strLastSlash = 0;

        // Chop the .exe from the exe name
#ifdef UNICODE
        strLastSlash = wcsrchr( strExeName, TEXT( '.' ) );
#else
        strLastSlash = strrchr( strExeName, TEXT( '.' ) );
#endif
        if( strLastSlash )
            *strLastSlash = 0;
    }

    // Search in directories:
    //      .\
    //      %EXE_DIR%\..\..\%EXE_NAME%
#ifdef UNICODE
    wcscpy_s( strDestPath, cchDest, strFilename );
#else
	strcpy_s( strDestPath, cchDest, strFilename );
#endif
    if( GetFileAttributes( strDestPath ) != 0xFFFFFFFF )
        return S_OK;

//    swprintf_s( strDestPath, cchDest, L"%s\\..\\..\\%s\\%s", strExePath, strExeName, strFilename );
#ifdef UNICODE
    swprintf_s( strDestPath, cchDest, L"%s\\..\\%s\\%s", strExePath, strExeName, strFilename );
#else
    sprintf_s( strDestPath, cchDest, "%s\\..\\%s\\%s", strExePath, strExeName, strFilename );
#endif
    if( GetFileAttributes( strDestPath ) != 0xFFFFFFFF )
        return S_OK;    

    // On failure, return the file as the path but also return an error code
#ifdef UNICODE
    wcscpy_s( strDestPath, cchDest, strFilename );
#else
    strcpy_s( strDestPath, cchDest, strFilename );
#endif

	CLASSERT( 0 );

    return E_FAIL;
}

template<>
int DeviceUtilsDX11::getNumDevices()
{
	IDXGIFactory* factory = NULL;
	IDXGIAdapter* adapter = NULL;
	CreateDXGIFactory( __uuidof(IDXGIFactory), (void**)&factory );

	u32 i = 0;
	while( factory->EnumAdapters( i, &adapter ) != DXGI_ERROR_NOT_FOUND )
	{
		i++;
	}

	factory->Release();
	return i;
}

template<>
void DeviceUtilsDX11::initDevice( DeviceDataBase* deviceDataBase, DriverType driverType, int deviceIdx )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)deviceDataBase;

	HRESULT hr = S_OK;
	UINT createDeviceFlg = 0;
#ifdef _DEBUG
	createDeviceFlg |= D3D11_CREATE_DEVICE_DEBUG;
#endif
	D3D_FEATURE_LEVEL fl[] = {
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0
	};

typedef HRESULT (WINAPI * LPD3D11CREATEDEVICE)( IDXGIAdapter*, D3D_DRIVER_TYPE, HMODULE, u32, D3D_FEATURE_LEVEL*, UINT, u32, ID3D11Device**, D3D_FEATURE_LEVEL*, ID3D11DeviceContext** );

	HMODULE moduleD3D11 = 0; 
#ifdef UNICODE
	moduleD3D11 = LoadLibrary( L"d3d11.dll" );
#else
	moduleD3D11 = LoadLibrary( "d3d11.dll" );
#endif
	CLASSERT( moduleD3D11 );

	LPD3D11CREATEDEVICE _DynamicD3D11CreateDevice; 
	_DynamicD3D11CreateDevice = ( LPD3D11CREATEDEVICE )GetProcAddress( moduleD3D11, "D3D11CreateDevice" );

	D3D_DRIVER_TYPE type = D3D_DRIVER_TYPE_HARDWARE;
	//	http://msdn.microsoft.com/en-us/library/ff476082(v=VS.85).aspx
	//	If you set the pAdapter parameter to a non-NULL value, you must also set the DriverType parameter to the D3D_DRIVER_TYPE_UNKNOWN value. If you set the pAdapter parameter to a non-NULL value and the DriverType parameter to the D3D_DRIVER_TYPE_HARDWARE value, D3D11CreateDevice returns an HRESULT of E_INVALIDARG.
	type = D3D_DRIVER_TYPE_UNKNOWN;
	if( driverType == DeviceUtilsDX11::DRIVER_REFERENCE )
	{
		type = D3D_DRIVER_TYPE_REFERENCE;
	}

	IDXGIAdapter* adapter = NULL;
	{//	get adapter of the index
		IDXGIFactory* factory = NULL;
		CreateDXGIFactory( __uuidof(IDXGIFactory), (void**)&factory );

		u32 i = 0;
		while( factory->EnumAdapters( i, &adapter ) != DXGI_ERROR_NOT_FOUND )
		{
			if( i==deviceIdx ) break;
			i++;
		}

		factory->Release();
	}

	// Create a hardware Direct3D 11 device
	hr = D3D11CreateDevice( adapter, 
		type, 
//			D3D_DRIVER_TYPE_REFERENCE,
		NULL, createDeviceFlg,
		fl, _countof(fl), D3D11_SDK_VERSION, &deviceData->m_device, NULL, &deviceData->m_context );

	CLASSERT( hr == S_OK );

   // Check if the hardware device supports Compute Shader 4.0
    D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts;
    deviceData->m_device->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &hwopts, sizeof(hwopts));

	if( !hwopts.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x )
	{
		SAFE_RELEASE( deviceData->m_context );
		SAFE_RELEASE( deviceData->m_device );

		debugPrintf("DX11 GPU is not present\n");
		CLASSERT( 0 );
	}
}

template<>
void DeviceUtilsDX11::releaseDevice( DeviceDataBase* deviceDataBase )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)deviceDataBase;

	SAFE_RELEASE( deviceData->m_context );
	SAFE_RELEASE( deviceData->m_device );
}

template<>
template<typename T>
void DeviceUtilsDX11::createDeviceBuffer( const DeviceDataBase* deviceDataBase, int numElems, DeviceBufferDX11& deviceBuffer,
								DeviceBufferBase::Type type)
{
	deviceBuffer.m_type = type;

	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)deviceDataBase;

	if( type & DeviceBufferBase::BUFFER )
	{
		HRESULT hr = S_OK;

		if( type == DeviceBufferBase::BUFFER_CONST )
		{
			CLASSERT( numElems == 1 );
			D3D11_BUFFER_DESC constant_buffer_desc;
			ZeroMemory( &constant_buffer_desc, sizeof(constant_buffer_desc) );
			constant_buffer_desc.ByteWidth = NEXTMULTIPLEOF( sizeof(T), 16 );
			constant_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
			constant_buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			constant_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			hr = deviceData->m_device->CreateBuffer( &constant_buffer_desc, NULL, &deviceBuffer.m_buffer );
			CLASSERT( hr == S_OK );
			return;
		}

		D3D11_BUFFER_DESC buffer_desc;
		ZeroMemory(&buffer_desc, sizeof(buffer_desc));
		buffer_desc.ByteWidth = numElems * sizeof(T);

		if( type != DeviceBufferBase::BUFFER_RAW )
		{
			buffer_desc.StructureByteStride = sizeof(T);
//		    buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		}

		if( type == DeviceBufferBase::BUFFER_STAGING )
		{
			buffer_desc.Usage = D3D11_USAGE_STAGING;
		    buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		}
		else if( type == DeviceBufferBase::BUFFER_INDEX )
		{
			buffer_desc.Usage = D3D11_USAGE_DEFAULT;
			buffer_desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
		}
		else if( type == DeviceBufferBase::BUFFER_VERTEX )
		{
			buffer_desc.Usage = D3D11_USAGE_DEFAULT;
			buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		}
		else
		{
			buffer_desc.Usage = D3D11_USAGE_DEFAULT;
			
			buffer_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
			buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;

//	check this
			if(type == DeviceBufferBase::BUFFER_RAW)
			{
//				buffer_desc.BindFlags |= D3D11_BIND_INDEX_BUFFER | D3D11_BIND_VERTEX_BUFFER;
				buffer_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS | D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS; // need this to be used for DispatchIndirect
			}
		}
		hr = deviceData->m_device->CreateBuffer(&buffer_desc, NULL, &deviceBuffer.m_buffer);

		CLASSERT( hr == S_OK );

		if( type == DeviceBufferBase::BUFFER_INDEX ) return;

		if( type == DeviceBufferBase::BUFFER || 
			type == DeviceBufferBase::BUFFER_RAW || 
			type == DeviceBufferBase::BUFFER_W_COUNTER )
		{
			// Create UAVs for all CS buffers
			D3D11_UNORDERED_ACCESS_VIEW_DESC uavbuffer_desc;
			ZeroMemory(&uavbuffer_desc, sizeof(uavbuffer_desc));
			uavbuffer_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;

			if( type == DeviceBufferBase::BUFFER_RAW )
			{
				uavbuffer_desc.Format = DXGI_FORMAT_R32_TYPELESS;
				uavbuffer_desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
				uavbuffer_desc.Buffer.NumElements = buffer_desc.ByteWidth / 4; 
			}
			else
			{
				uavbuffer_desc.Format = DXGI_FORMAT_UNKNOWN;
				uavbuffer_desc.Buffer.NumElements = numElems;
			}

			if( type == DeviceBufferBase::BUFFER_W_COUNTER )
			{
				uavbuffer_desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_COUNTER;
			}

			hr = deviceData->m_device->CreateUnorderedAccessView(deviceBuffer.m_buffer, &uavbuffer_desc, &deviceBuffer.m_uav);
			CLASSERT( hr == S_OK );

			// Create SRVs for all CS buffers
			D3D11_SHADER_RESOURCE_VIEW_DESC srvbuffer_desc;
			ZeroMemory(&srvbuffer_desc, sizeof(srvbuffer_desc));
			if( type == DeviceBufferBase::BUFFER_RAW )
			{
				CLASSERT( sizeof(T) <= 16 );
				srvbuffer_desc.Format = DXGI_FORMAT_R32_UINT;
				srvbuffer_desc.Buffer.ElementWidth = numElems;
//			if ( buffer_desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS )
//			{
//				srvbuffer_desc.Format = DXGI_FORMAT_R32_TYPELESS;
//				srvbuffer_desc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
//				srvbuffer_desc.BufferEx.NumElements = buffer_desc.ByteWidth / 4;
			}
			else
			{
				srvbuffer_desc.Format = DXGI_FORMAT_UNKNOWN;
				srvbuffer_desc.Buffer.ElementWidth = numElems;
			}
			srvbuffer_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;

			hr = deviceData->m_device->CreateShaderResourceView(deviceBuffer.m_buffer, &srvbuffer_desc, &deviceBuffer.m_srv);
			CLASSERT( hr == S_OK );
		}
		else if( type == DeviceBufferBase::BUFFER_APPEND )
		{
			D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
			ZeroMemory( &desc, sizeof(desc) );
			desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			desc.Buffer.FirstElement = 0;

			desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_APPEND;

			desc.Format = DXGI_FORMAT_UNKNOWN;      // Format must be must be DXGI_FORMAT_UNKNOWN, when creating a View of a Structured Buffer
			desc.Buffer.NumElements = buffer_desc.ByteWidth / buffer_desc.StructureByteStride; 

			hr = deviceData->m_device->CreateUnorderedAccessView( deviceBuffer.m_buffer, &desc, &deviceBuffer.m_uav );
			CLASSERT( hr == S_OK );
		}
	}
	else
	{
		CLASSERT(0);
	}
}

template<>
void DeviceUtilsDX11::deleteDeviceBuffer( const DeviceDataBase* deviceDataBase, DeviceBufferDX11& deviceBuffer )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)deviceDataBase;

	if( deviceBuffer.m_buffer )
	{
		deviceBuffer.m_buffer->Release();
		deviceBuffer.m_buffer = NULL;
	}
	if( deviceBuffer.m_uav )
	{
		deviceBuffer.m_uav->Release();
		deviceBuffer.m_uav = NULL;
	}
	if( deviceBuffer.m_srv )
	{
		deviceBuffer.m_srv->Release();
		deviceBuffer.m_srv = NULL;
	}
}


template<>
template<typename T>
void DeviceUtilsDX11::writeDataToDevice( const DeviceDataBase* deviceDataBase, int numElems, DeviceBufferDX11& deviceBuffer, const void* hostPtr, int offsetNumElems )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)deviceDataBase;

/*
    D3D11_QUERY_DESC queryDesc;
    queryDesc.Query = D3D11_QUERY_EVENT;
    queryDesc.MiscFlags = 0;
    ID3D11Query *localQuery;
    dddx11->m_device->CreateQuery(&queryDesc, &localQuery);
    dddx11->m_context->Begin(localQuery);
*/
    D3D11_BOX destRegion;
    destRegion.left = offsetNumElems*sizeof(T);
    destRegion.front = 0;
    destRegion.top = 0;
    destRegion.bottom = 1;
    destRegion.back = 1;
    destRegion.right = (offsetNumElems+numElems)*sizeof(T);
	deviceData->m_context->UpdateSubresource(deviceBuffer.m_buffer, 0, &destRegion, hostPtr, 0, 0);

/*
	if( localQuery )
	{
		// Perform query to make call blocking
		context->End(localQuery);
		while( S_OK != context->GetData(localQuery, NULL, 0, 0) )
		{
		}
		SAFE_RELEASE( localQuery );
		localQuery = NULL;
	}
*/
}

template<>
template<typename T>
void DeviceUtilsDX11::readDataFromDevice( const DeviceDataBase* deviceDataBase, int numElems, DeviceBufferDX11& deviceBuffer, void* hostPtr, DeviceBufferDX11* stagingBuffer11, int offsetNumElems )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)deviceDataBase;

	ID3D11Buffer *StagingBuffer = stagingBuffer11->m_buffer;
    D3D11_MAPPED_SUBRESOURCE MappedVelResource = {0};

    D3D11_BOX destRegion;
    destRegion.left = offsetNumElems*sizeof(T);
    destRegion.front = 0;
    destRegion.top = 0;
    destRegion.bottom = 1;
    destRegion.back = 1;
    destRegion.right = (offsetNumElems+numElems)*sizeof(T);

    deviceData->m_context->CopySubresourceRegion(
            StagingBuffer,
            0,
            0,
            0,
            0 ,
			deviceBuffer.m_buffer,
            0,
            &destRegion
    );

    deviceData->m_context->Map(StagingBuffer, 0, D3D11_MAP_READ, 0, &MappedVelResource);
    memcpy(hostPtr, MappedVelResource.pData, numElems*sizeof(T));
    deviceData->m_context->Unmap(StagingBuffer, 0);
}

template<>
void DeviceUtilsDX11::waitForCompletion( const DeviceDataBase* deviceDataBase )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)deviceDataBase;

	ID3D11Query* syncQuery;
	D3D11_QUERY_DESC qDesc;
	qDesc.Query = D3D11_QUERY_EVENT;
	qDesc.MiscFlags = 0;
	deviceData->m_device->CreateQuery( &qDesc, &syncQuery );
//	deviceData->m_context->Begin( syncQuery );
	deviceData->m_context->End( syncQuery );
	while( deviceData->m_context->GetData( syncQuery, 0,0,0 ) == S_FALSE ){}
	syncQuery->Release();
}

template<>
KernelBuilderDX11::KernelBuilder( const DeviceDataBase* deviceDataBase, char* fileName, const char* option, bool addExtension )
{
	char fileNameWithExtension[256];

	if( addExtension )
		sprintf_s( fileNameWithExtension, "%s.hlsl", fileName );
	else
		sprintf_s( fileNameWithExtension, "%s", fileName );

	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)deviceDataBase;

	m_deviceData = deviceData;

	int nameLength = strlen(fileNameWithExtension)+1;
#ifdef UNICODE
	WCHAR* wfileNameWithExtension = new WCHAR[nameLength];
#else
	CHAR* wfileNameWithExtension = new CHAR[nameLength];
#endif
	memset(wfileNameWithExtension,0,nameLength);
#ifdef UNICODE
	MultiByteToWideChar(CP_ACP,0,fileNameWithExtension,-1, wfileNameWithExtension, nameLength);
#else
	sprintf_s(wfileNameWithExtension, nameLength, "%s", fileNameWithExtension);
#endif
//			swprintf_s(wfileNameWithExtension, nameLength*2, L"%s", fileNameWithExtension);

	HRESULT hr;

	// Finds the correct path for the shader file.
	// This is only required for this sample to be run correctly from within the Sample Browser,
	// in your own projects, these lines could be removed safely
	hr = FindDXSDKShaderFileCch( m_path, MAX_PATH, wfileNameWithExtension );

	delete [] wfileNameWithExtension;

	CLASSERT( hr == S_OK );
}

template<>
void KernelBuilderDX11::createKernel( const char* funcName, DeviceKernelDX11& kernelOut )
{
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;
	HRESULT hr;

	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
	// Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
	// Setting this flag improves the shader debugging experience, but still allows 
	// the shaders to be optimized and to run exactly the way they will run in 
	// the release configuration of this program.
	dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

	const D3D_SHADER_MACRO defines[] = 
	{
#ifdef USE_STRUCTURED_BUFFERS
		"USE_STRUCTURED_BUFFERS", "1",
#endif

#ifdef TEST_DOUBLE
		"TEST_DOUBLE", "1",
#endif
		NULL, NULL
	};

	// We generally prefer to use the higher CS shader profile when possible as CS 5.0 is better performance on 11-class hardware
	LPCSTR pProfile = ( deviceData->m_device->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0 ) ? "cs_5_0" : "cs_4_0";

	ID3DBlob* pErrorBlob = NULL;
	ID3DBlob* pBlob = NULL;
	hr = D3DX11CompileFromFile( m_path, defines, NULL, funcName, pProfile, 
		dwShaderFlags, NULL, NULL, &pBlob, &pErrorBlob, NULL );

	if ( FAILED(hr) )
	{
		if ( pErrorBlob ) OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
	}
	CLASSERT( hr == S_OK );

	hr = deviceData->m_device->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, 
		&kernelOut.m_kernel );

#if defined(DEBUG) || defined(PROFILE)
	if ( kernelOut.m_kernel )
		kernelOut.m_kernel->SetPrivateData( WKPDID_D3DDebugObjectName, lstrlenA(pFunctionName), pFunctionName );
#endif

	SAFE_RELEASE( pErrorBlob );
	SAFE_RELEASE( pBlob );
}

template<>
void KernelBuilderDX11::createKernel( const DeviceDataBase* dd, const char* funcName, DeviceKernelDX11& kernelOut, const char* shader, int size )
{
	DeviceDataDX11* deviceData = (DeviceDataDX11*)dd;
	HRESULT hr;

	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
	// Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
	// Setting this flag improves the shader debugging experience, but still allows 
	// the shaders to be optimized and to run exactly the way they will run in 
	// the release configuration of this program.
	dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

	const D3D_SHADER_MACRO defines[] = 
	{
#ifdef USE_STRUCTURED_BUFFERS
		"USE_STRUCTURED_BUFFERS", "1",
#endif

#ifdef TEST_DOUBLE
		"TEST_DOUBLE", "1",
#endif
		NULL, NULL
	};

	// We generally prefer to use the higher CS shader profile when possible as CS 5.0 is better performance on 11-class hardware
	LPCSTR pProfile = ( deviceData->m_device->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0 ) ? "cs_5_0" : "cs_4_0";

	ID3DBlob* pErrorBlob = NULL;
	ID3DBlob* pBlob = NULL;
	hr = D3DX11CompileFromMemory( shader, size, 0, defines, NULL, funcName, pProfile, 
		dwShaderFlags, NULL, NULL, &pBlob, &pErrorBlob, NULL );

	if ( FAILED(hr) )
	{
		if ( pErrorBlob ) OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
	}
	CLASSERT( hr == S_OK );

	hr = deviceData->m_device->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, 
		&kernelOut.m_kernel );

#if defined(DEBUG) || defined(PROFILE)
	if ( kernelOut.m_kernel )
		kernelOut.m_kernel->SetPrivateData( WKPDID_D3DDebugObjectName, lstrlenA(pFunctionName), pFunctionName );
#endif

	SAFE_RELEASE( pErrorBlob );
	SAFE_RELEASE( pBlob );
}

template<>
KernelBuilderDX11::~KernelBuilderDX11()
{

}

template<>
void KernelBuilderDX11::deleteKernel( const DeviceDataBase* deviceDataBase, DeviceKernelDX11& kernel )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)deviceDataBase;

	if( kernel.m_kernel )
	{
		kernel.m_kernel->Release();
		kernel.m_kernel = NULL;
	}
}

template<>
KernelLauncherDX11::KernelLauncher( const DeviceDataBase* deviceDataBase, DeviceKernelDX11& kernel )
{
	CLASSERT( deviceDataBase->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)deviceDataBase;

	m_deviceData = deviceData;
	m_kernel = &kernel;
	m_idx = 0;
	m_idxRW = 0;
}

template<>
KernelLauncherDX11::~KernelLauncher()
{

}

template<>
void KernelLauncherDX11::pushBackR( const DeviceBufferDX11& buffer )
{
	CLASSERT( m_deviceData->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;
	if( buffer.m_srv == 0 )
	{
		m_idx++;
		return;
	}
	deviceData->m_context->CSSetShaderResources( m_idx++, 1, &buffer.m_srv );
}

template<>
void KernelLauncherDX11::pushBackRW( DeviceBufferDX11& buffer, const int* counterInitValues )
{
	CLASSERT( m_deviceData->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;

	if( buffer.m_buffer == 0 )
	{
		deviceData->m_context->CSSetUnorderedAccessViews( m_idxRW++, 1, &buffer.m_uav, (const UINT*)counterInitValues );
		return;
	}
	deviceData->m_context->CSSetUnorderedAccessViews( m_idxRW++, 1, &buffer.m_uav, (const UINT*)counterInitValues );
}

template<>
template<typename T>
void KernelLauncherDX11::setConst( const DeviceBufferDX11& buffer, const T* hostData )
{
	CLASSERT( m_deviceData->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;
	CLASSERT( buffer.m_buffer );

    D3D11_MAPPED_SUBRESOURCE MappedResource;
    deviceData->m_context->Map( buffer.m_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource );
    memcpy( MappedResource.pData, hostData, sizeof(T) );
    deviceData->m_context->Unmap( buffer.m_buffer, 0 );
    ID3D11Buffer* ppCB[1] = { buffer.m_buffer };
    deviceData->m_context->CSSetConstantBuffers( 0, 1, ppCB );
}

template<>
void KernelLauncherDX11::launch1D( int numThreads, int localSize )
{
	CLASSERT( m_deviceData->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;

	deviceData->m_context->CSSetShader( m_kernel->m_kernel, NULL, 0 );

	int nx, ny, nz;
	nx = max2( 1, (numThreads/localSize)+(!(numThreads%localSize)?0:1) );
	ny = 1;
	nz = 1;

	deviceData->m_context->Dispatch( nx, ny, nz );

	//	set 0 to registers
	{
	    deviceData->m_context->CSSetShader( NULL, NULL, 0 );

		if( m_idxRW )
		{
			ID3D11UnorderedAccessView* aUAViewsNULL[ 16 ] = { 0 };
			deviceData->m_context->CSSetUnorderedAccessViews( 0, 
				min2( (unsigned int)m_idxRW, sizeof(aUAViewsNULL)/sizeof(*aUAViewsNULL) ), aUAViewsNULL, NULL );
		}

		if( m_idx )
		{
			ID3D11ShaderResourceView* ppSRVNULL[16] = { 0 };
			deviceData->m_context->CSSetShaderResources( 0, 
				min2( (unsigned int)m_idx, sizeof(ppSRVNULL)/sizeof(*ppSRVNULL) ), ppSRVNULL );
		}
	}
}

template<>
void KernelLauncherDX11::launch2D( int numThreadsX, int numThreadsY, int localSizeX, int localSizeY )
{
	CLASSERT( m_deviceData->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;

	deviceData->m_context->CSSetShader( m_kernel->m_kernel, NULL, 0 );

	int nx, ny, nz;
	nx = max2( 1, (numThreadsX/localSizeX)+(!(numThreadsX%localSizeX)?0:1) );
	ny = max2( 1, (numThreadsY/localSizeY)+(!(numThreadsY%localSizeY)?0:1) );
	nz = 1;

	deviceData->m_context->Dispatch( nx, ny, nz );

	//	set 0 to registers
	{
	    deviceData->m_context->CSSetShader( NULL, NULL, 0 );

		if( m_idxRW )
		{
			ID3D11UnorderedAccessView* aUAViewsNULL[ 16 ] = { 0 };
			deviceData->m_context->CSSetUnorderedAccessViews( 0, 
				min2( (unsigned int)m_idxRW, sizeof(aUAViewsNULL)/sizeof(*aUAViewsNULL) ), aUAViewsNULL, NULL );
		}

		if( m_idx )
		{
			ID3D11ShaderResourceView* ppSRVNULL[16] = { 0 };
			deviceData->m_context->CSSetShaderResources( 0, 
				min2( (unsigned int)m_idx, sizeof(ppSRVNULL)/sizeof(*ppSRVNULL) ), ppSRVNULL );
		}
	}
}

template<>
void KernelLauncherDX11::launch1DOnDevice( DeviceBufferDX11& numElemsBuffer, u32 alignedOffset, int localSize )
{
	CLASSERT( m_deviceData->m_type == DeviceDataBase::TYPE_DX11 );
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;

	deviceData->m_context->CSSetShader( m_kernel->m_kernel, NULL, 0 );

	deviceData->m_context->DispatchIndirect( numElemsBuffer.m_buffer, alignedOffset );

	//	set 0 to registers
	{
	    deviceData->m_context->CSSetShader( NULL, NULL, 0 );

		if( m_idxRW )
		{
			ID3D11UnorderedAccessView* aUAViewsNULL[ 16 ] = { 0 };
			deviceData->m_context->CSSetUnorderedAccessViews( 0, 
				min2( (unsigned int)m_idxRW, sizeof(aUAViewsNULL)/sizeof(*aUAViewsNULL) ), aUAViewsNULL, NULL );
		}

		if( m_idx )
		{
			ID3D11ShaderResourceView* ppSRVNULL[16] = { 0 };
			deviceData->m_context->CSSetShaderResources( 0, 
				min2( (unsigned int)m_idx, sizeof(ppSRVNULL)/sizeof(*ppSRVNULL) ), ppSRVNULL );
		}
	}
}




struct StopwatchDx11
{
	public:
		__inline
		StopwatchDx11() : m_deviceData(0){}
		__inline
		StopwatchDx11( const DeviceDataBase* deviceData );
		__inline
		~StopwatchDx11();

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
		const DeviceDataDX11* m_deviceData;
		ID3D11Query* m_tQuery[CAPACITY+1];
		ID3D11Query* m_fQuery;
		UINT64 m_t[CAPACITY];
		int m_idx;
};

StopwatchDx11::StopwatchDx11( const DeviceDataBase* deviceData )
{
	init( deviceData );
}

void StopwatchDx11::init( const DeviceDataBase* deviceData )
{
	CLASSERT( deviceData->m_type == DeviceDataBase::TYPE_DX11 );
	m_deviceData = (const DeviceDataDX11*)deviceData;

	{
		D3D11_QUERY_DESC qDesc;
		qDesc.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
		qDesc.MiscFlags = 0;
		m_deviceData->m_device->CreateQuery( &qDesc, &m_fQuery );
	}
	for(int i=0; i<CAPACITY+1; i++)
	{
		D3D11_QUERY_DESC qDesc;
		qDesc.Query = D3D11_QUERY_TIMESTAMP;
		qDesc.MiscFlags = 0;
		m_deviceData->m_device->CreateQuery( &qDesc, &m_tQuery[i] );
	}
}

StopwatchDx11::~StopwatchDx11()
{
	m_fQuery->Release();
	for(int i=0; i<CAPACITY+1; i++)
	{
		m_tQuery[i]->Release();
	}
}

void StopwatchDx11::start()
{
	m_idx = 0;
	m_deviceData->m_context->Begin( m_fQuery );
	m_deviceData->m_context->End( m_tQuery[m_idx++] );
}

void StopwatchDx11::split()
{
	if( m_idx < CAPACITY )
		m_deviceData->m_context->End( m_tQuery[m_idx++] );
}

void StopwatchDx11::stop()
{
	m_deviceData->m_context->End( m_tQuery[m_idx++] );
	m_deviceData->m_context->End( m_fQuery );
}

float StopwatchDx11::getMs()
{
	D3D11_QUERY_DATA_TIMESTAMP_DISJOINT d;
//	m_deviceData->m_context->End( m_fQuery );
	while( m_deviceData->m_context->GetData( m_fQuery, &d,sizeof(D3D11_QUERY_DATA_TIMESTAMP_DISJOINT),0 ) == S_FALSE ) {}

	while( m_deviceData->m_context->GetData( m_tQuery[0], &m_t[0],sizeof(UINT64),0 ) == S_FALSE ){}
	while( m_deviceData->m_context->GetData( m_tQuery[1], &m_t[1],sizeof(UINT64),0 ) == S_FALSE ){}

	CLASSERT( d.Disjoint == false );

	float elapsedMs = (m_t[1] - m_t[0])/(float)d.Frequency*1000;
	return elapsedMs;

}

void StopwatchDx11::getMs( float* times, int capacity )
{
	CLASSERT( capacity <= CAPACITY );

	D3D11_QUERY_DATA_TIMESTAMP_DISJOINT d;
	while( m_deviceData->m_context->GetData( m_fQuery, &d,sizeof(D3D11_QUERY_DATA_TIMESTAMP_DISJOINT),0 ) == S_FALSE ) {}

	for(int i=0; i<m_idx; i++)
	{
		while( m_deviceData->m_context->GetData( m_tQuery[i], &m_t[i],sizeof(UINT64),0 ) == S_FALSE ){}
	}

	CLASSERT( d.Disjoint == false );

	for(int i=0; i<capacity; i++)
	{
		times[i] = (m_t[i+1] - m_t[i])/(float)d.Frequency*1000;
	}
}