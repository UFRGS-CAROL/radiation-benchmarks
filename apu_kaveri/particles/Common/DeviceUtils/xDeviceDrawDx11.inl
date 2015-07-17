#include <xnamath.h>
#include <common/Math/Quaternion.h>
#include <Common/Math/Array.h>


struct DeviceShaderDX11
{
	DeviceShaderDX11() : m_pShader(0), m_vertexLayout(0) {}

	union
	{
		ID3D11VertexShader* m_vShader;
		ID3D11PixelShader* m_pShader;
		ID3D11GeometryShader* m_gShader;
		ID3D11HullShader* m_hShader;
		ID3D11DomainShader* m_dShader;
	};
	ID3D11InputLayout* m_vertexLayout;
};

__inline
bool operator==(const DeviceShaderDX11& a, const DeviceShaderDX11& b)
{
	return a.m_pShader == b.m_pShader;
}

__inline
bool operator!=(const DeviceShaderDX11& a, const DeviceShaderDX11& b)
{
	return a.m_pShader != b.m_pShader;
}

class ShaderUtilsDX11
{
	public:
		ShaderUtilsDX11( const DeviceDataBase* deviceData, const char* fileName )
			: m_deviceData( deviceData )
		{
			CLASSERT( m_deviceData->TYPE_DX11 );
			MultiByteToWideChar(CP_ACP,0,fileName,-1, m_path, strlen(fileName)+1);
		}

		~ShaderUtilsDX11()
		{

		}

		__inline
		void createVertexShader( const char* funcName, DeviceShaderDX11& shaderOut, int inputSize = 0, D3D11_INPUT_ELEMENT_DESC* vtxLayout = 0 );
		__inline
		void createPixelShader( const char* funcName, DeviceShaderDX11& shaderOut );
		__inline
		void createGeometryShader( const char* funcName, DeviceShaderDX11& shaderOut );
		__inline
		void createHullShader( const char* funcName, DeviceShaderDX11& shaderOut );
		__inline
		void createDomainShader( const char* funcName, DeviceShaderDX11& shaderOut );

	public:
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

	public:
		static void deleteShader(DeviceShaderDX11& shader)
		{
			if( shader.m_pShader ) shader.m_pShader->Release();
			shader.m_pShader = 0;
			if( shader.m_vertexLayout ) shader.m_vertexLayout->Release();
			shader.m_vertexLayout = 0;
		}

//	private:
		static HRESULT CompileShaderFromFile( WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut )
		{
			HRESULT hr = S_OK;

			DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
		#if defined( DEBUG ) || defined( _DEBUG )
			// Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
			// Setting this flag improves the shader debugging experience, but still allows 
			// the shaders to be optimized and to run exactly the way they will run in 
			// the release configuration of this program.
			dwShaderFlags |= D3DCOMPILE_DEBUG;
		#endif

			ID3DBlob* pErrorBlob;
			hr = D3DX11CompileFromFile( szFileName, NULL, NULL, szEntryPoint, szShaderModel, 
				dwShaderFlags, 0, NULL, ppBlobOut, &pErrorBlob, NULL );
			if( FAILED(hr) )
			{
				if( pErrorBlob != NULL )
					OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
				if( pErrorBlob ) pErrorBlob->Release();
				return hr;
			}
			if( pErrorBlob ) pErrorBlob->Release();

			return S_OK;
		}

};

void ShaderUtilsDX11::createVertexShader(const char *funcName, DeviceShaderDX11 &shaderOut, int inputSize, D3D11_INPUT_ELEMENT_DESC* vtxLayout)
{
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;

	HRESULT hr = S_OK;
	ID3DBlob* pPSBlob = NULL;
	hr = CompileShaderFromFile( m_path, funcName, "vs_5_0", &pPSBlob );
	if( FAILED( hr ) )
	{
		MessageBox( NULL, L"The FX file cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK ); CLASSERT(0);
	}
	CLASSERT( SUCCEEDED( hr ) );

	// Create the pixel shader
	hr = deviceData->m_device->CreateVertexShader( pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &shaderOut.m_vShader );
	CLASSERT( SUCCEEDED( hr ) );

	if( vtxLayout )
	{
		hr = deviceData->m_device->CreateInputLayout( vtxLayout, inputSize, 
			pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), &shaderOut.m_vertexLayout );
		CLASSERT( SUCCEEDED( hr ) );
	}

	pPSBlob->Release();
}

void ShaderUtilsDX11::createPixelShader(const char *funcName, DeviceShaderDX11 &shaderOut)
{
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;

	HRESULT hr = S_OK;
	ID3DBlob* pPSBlob = NULL;
	hr = CompileShaderFromFile( m_path, funcName, "ps_5_0", &pPSBlob );
	if( FAILED( hr ) )
	{
		MessageBox( NULL, L"The FX file cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK ); CLASSERT(0);
	}
	CLASSERT( SUCCEEDED( hr ) );

	// Create the pixel shader
	hr = deviceData->m_device->CreatePixelShader( pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &shaderOut.m_pShader );
	CLASSERT( SUCCEEDED( hr ) );
	pPSBlob->Release();
}

void ShaderUtilsDX11::createGeometryShader(const char *funcName, DeviceShaderDX11 &shaderOut)
{
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;

	HRESULT hr = S_OK;
	ID3DBlob* pPSBlob = NULL;
	hr = CompileShaderFromFile( m_path, funcName, "gs_5_0", &pPSBlob );
	if( FAILED( hr ) )
	{
		MessageBox( NULL, L"The FX file cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK ); CLASSERT(0);
	}
	CLASSERT( SUCCEEDED( hr ) );

	// Create the pixel shader
	hr = deviceData->m_device->CreateGeometryShader( pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &shaderOut.m_gShader );
	CLASSERT( SUCCEEDED( hr ) );
	pPSBlob->Release();
}

void ShaderUtilsDX11::createHullShader(const char *funcName, DeviceShaderDX11 &shaderOut)
{
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;

	HRESULT hr = S_OK;
	ID3DBlob* pPSBlob = NULL;
	hr = CompileShaderFromFile( m_path, funcName, "hs_5_0", &pPSBlob );
	if( FAILED( hr ) )
	{
		MessageBox( NULL, L"The FX file cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK ); CLASSERT(0);
	}
	CLASSERT( SUCCEEDED( hr ) );

	// Create the pixel shader
	hr = deviceData->m_device->CreateHullShader( pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &shaderOut.m_hShader );
	CLASSERT( SUCCEEDED( hr ) );
	pPSBlob->Release();
}

void ShaderUtilsDX11::createDomainShader(const char *funcName, DeviceShaderDX11 &shaderOut)
{
	DeviceDataDX11* deviceData = (DeviceDataDX11*)m_deviceData;

	HRESULT hr = S_OK;
	ID3DBlob* pPSBlob = NULL;
	hr = CompileShaderFromFile( m_path, funcName, "ds_5_0", &pPSBlob );
	if( FAILED( hr ) )
	{
		MessageBox( NULL, L"The FX file cannot be compiled.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK ); CLASSERT(0);
	}
	CLASSERT( SUCCEEDED( hr ) );

	// Create the pixel shader
	hr = deviceData->m_device->CreateDomainShader( pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &shaderOut.m_dShader );
	CLASSERT( SUCCEEDED( hr ) );
	pPSBlob->Release();
}


//=============================
//=============================
//=============================


struct VertexColorStruct
{
	VertexColorStruct(){ m_normal = XMFLOAT3(0,0,0); }

    XMFLOAT3 m_pos;
	XMFLOAT3 m_normal;
    XMFLOAT4 m_color;
	XMFLOAT2 m_texCrd;
};

struct ConstantBuffer
{
	XMMATRIX m_world;
	XMMATRIX m_view;
	XMMATRIX m_projection;
	float4 m_gData;

	void setParticleRadius(float rad) { m_gData.x = rad; }
};

//=============================
//=============================
//=============================

struct DeviceRenderTargetDX11
{
	DeviceRenderTargetDX11() : m_texture(0), m_renderTarget(0), m_srv(0) {}

	enum BufferType
	{
//		TYPE_SRV = (1),
//		TYPE_RTV = (1<<1),
		TYPE_RENDER_TARGET,
		TYPE_DEPTH_STENCIL,
	};

	__inline
	static void createRenderTarget( const DeviceDataBase* deviceData, int width, int height, DeviceRenderTargetDX11& renderTarget,
		u32 type = TYPE_RENDER_TARGET );
	__inline
	static void deleteRenderTarget( const DeviceDataBase* deviceData, DeviceRenderTargetDX11& renderTarget );


	ID3D11Texture2D* m_texture;
	union
	{
		ID3D11RenderTargetView* m_renderTarget;
		ID3D11DepthStencilView* m_depthStencilView;
	};
	ID3D11ShaderResourceView* m_srv;
};

void DeviceRenderTargetDX11::createRenderTarget( const DeviceDataBase* deviceData, int width, int height, DeviceRenderTargetDX11& renderTarget,
	u32 type )
{
	DeviceDataDX11* dd = (DeviceDataDX11*)deviceData;

	if( type == TYPE_RENDER_TARGET )
	{
		D3D11_TEXTURE2D_DESC Desc;
		ZeroMemory( &Desc, sizeof( D3D11_TEXTURE2D_DESC ) );
		Desc.ArraySize = 1;
		Desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
		Desc.Usage = D3D11_USAGE_DEFAULT;
		Desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		Desc.Width = width;
		Desc.Height = height;
		Desc.MipLevels = 1;
		Desc.SampleDesc.Count = 1;
		dd->m_device->CreateTexture2D( &Desc, NULL, &renderTarget.m_texture );

		D3D11_RENDER_TARGET_VIEW_DESC DescRT;
		DescRT.Format = Desc.Format;
		DescRT.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
		DescRT.Texture2D.MipSlice = 0;
		dd->m_device->CreateRenderTargetView( renderTarget.m_texture, &DescRT, &renderTarget.m_renderTarget );

		// Create the resource view
		{
			D3D11_SHADER_RESOURCE_VIEW_DESC DescRV;
			DescRV.Format = Desc.Format;
			DescRV.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
			DescRV.Texture2D.MipLevels = 1;
			DescRV.Texture2D.MostDetailedMip = 0;
			dd->m_device->CreateShaderResourceView( renderTarget.m_texture, &DescRV, &renderTarget.m_srv );
		}
	}
	else if( type == TYPE_DEPTH_STENCIL )
	{
		D3D11_TEXTURE2D_DESC descDepth;
		ZeroMemory( &descDepth, sizeof( D3D11_TEXTURE2D_DESC ) );
		descDepth.Width = width;
		descDepth.Height = height;
		descDepth.MipLevels = 1;
		descDepth.ArraySize = 1;
		descDepth.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
		descDepth.SampleDesc.Count = 1;
		descDepth.SampleDesc.Quality = 0;
		descDepth.Usage = D3D11_USAGE_DEFAULT;
		descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
		descDepth.CPUAccessFlags = 0;
		descDepth.MiscFlags = 0;
		dd->m_device->CreateTexture2D( &descDepth, NULL, &renderTarget.m_texture );

		D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
		ZeroMemory( &descDSV, sizeof(descDSV) );
		descDSV.Format = descDepth.Format;
		descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		descDSV.Texture2D.MipSlice = 0;
		dd->m_device->CreateDepthStencilView( renderTarget.m_texture, &descDSV, &renderTarget.m_depthStencilView );
	}
	else
	{
		CLASSERT(0);
	}
}

void DeviceRenderTargetDX11::deleteRenderTarget( const DeviceDataBase* deviceData, DeviceRenderTargetDX11& renderTarget )
{
	if( renderTarget.m_renderTarget )
		renderTarget.m_renderTarget->Release();
	if( renderTarget.m_texture )
		renderTarget.m_texture->Release();
	if( renderTarget.m_srv )
		renderTarget.m_srv->Release();

	renderTarget.m_renderTarget = 0;
	renderTarget.m_texture = 0;
	renderTarget.m_srv = 0;
}

//=============================
//=============================
//=============================

template<typename T, DeviceBufferBase::Type BUFFERTYPE>
struct AppendableBuffer
{
	AppendableBuffer( const DeviceDataDX11* deviceData );
	~AppendableBuffer();

	enum
	{
		BUFFER_CAPACITY = 1024*1024*4*4,
	};

	const DeviceDataDX11* m_deviceData;
	DeviceBufferDX11 m_buffer;
	int m_nElems;

	//	return offset
	int append( int numElems, const void* hostPtr );
	ID3D11Buffer* getBuffer() { return m_buffer.m_buffer; }

	void reset(){ m_nElems = 0; }
};

template<typename T, DeviceBufferBase::Type BUFFERTYPE>
AppendableBuffer<T, BUFFERTYPE>::AppendableBuffer( const DeviceDataDX11* deviceData )
: m_deviceData( deviceData )
{
	reset();

	if( m_buffer.m_buffer == 0 )
		DeviceUtilsDX11::createDeviceBuffer<T>( m_deviceData, BUFFER_CAPACITY, m_buffer, BUFFERTYPE );
}

template<typename T, DeviceBufferBase::Type BUFFERTYPE>
AppendableBuffer<T, BUFFERTYPE>::~AppendableBuffer()
{
	DeviceUtilsDX11::deleteDeviceBuffer( m_deviceData, m_buffer );
}

template<typename T, DeviceBufferBase::Type BUFFERTYPE>
int AppendableBuffer<T, BUFFERTYPE>::append( int numElems, const void* hostPtr )
{
	CLASSERT( m_nElems+numElems < BUFFER_CAPACITY );
	int nElems = m_nElems;

	DeviceUtilsDX11::writeDataToDevice<T>( m_deviceData, numElems, m_buffer, hostPtr, m_nElems );
	
	m_nElems += numElems; 

	return nElems;
}

typedef AppendableBuffer<VertexColorStruct, DeviceBufferBase::BUFFER_VERTEX> AppendVertexBuffer;
typedef AppendableBuffer<u32, DeviceBufferBase::BUFFER_INDEX> AppendIndexBuffer;


//=============================
//=============================
//=============================

extern DeviceShaderDX11 g_defaultVertexShader;
extern DeviceShaderDX11 g_defaultPixelShader;

extern DeviceShaderDX11	g_pointSpriteVertexShader;
extern DeviceShaderDX11 g_pointSpritePixelShader;
extern DeviceShaderDX11 g_pointSpriteGeometryShader;

//=============================
//=============================
//=============================

struct RenderObject
{
	struct Resource
	{
		enum Type
		{
			VTX_SHADER, 
			GEOM_SHADER,
			HULL_SHADER,
			DOMAIN_SHADER,
			PIXEL_SHADER,
		};

		Type m_type;
		DeviceBufferDX11 m_buffer;

		void bind(const DeviceDataDX11* dd, int resIdx[5])
		{
			switch (m_type)
			{
			case VTX_SHADER:
				dd->m_context->VSSetShaderResources( resIdx[0]++, 1, &m_buffer.m_srv );
				break;
			case GEOM_SHADER:
				dd->m_context->GSSetShaderResources( resIdx[1]++, 1, &m_buffer.m_srv );
				break;
			case HULL_SHADER:
				dd->m_context->HSSetShaderResources( resIdx[2]++, 1, &m_buffer.m_srv );
				break;
			case DOMAIN_SHADER:
				dd->m_context->DSSetShaderResources( resIdx[3]++, 1, &m_buffer.m_srv );
				break;
			case PIXEL_SHADER:
				dd->m_context->PSSetShaderResources( resIdx[4]++, 1, &m_buffer.m_srv );
				break;
			default:
				CLASSERT(0);
				break;
			};
		}
	};
	enum
	{
		MAX_RESOURCES = 6,
	};
	enum
	{
		DRAW_INDEXED,
		DRAW,
		DRAW_INSTANCED,
	};

	ConstantBuffer m_matrix;

	ID3D11Buffer* m_vtxBuffer;
	ID3D11Buffer* m_idxBuffer;
	DeviceShaderDX11 m_vertexShader;
	DeviceShaderDX11 m_pixelShader;
	DeviceShaderDX11 m_geomShader;
	DeviceShaderDX11 m_hullShader;
	DeviceShaderDX11 m_domainShader;
	int m_nIndices;
	int m_vtxOffset;
	int m_idxOffset;
	int m_nInstances;
	u32 m_vtxStride; // default:0, set if needed. 

	D3D_PRIMITIVE_TOPOLOGY m_topology;
	u16 m_drawType;
	u16 m_nResources;
	Resource m_resources[MAX_RESOURCES];

	RenderObject()
	{ 
		m_matrix.m_world = XMMatrixIdentity(); 
		m_vertexShader = g_defaultVertexShader; m_pixelShader = g_defaultPixelShader;
		m_hullShader.m_hShader=0; m_domainShader.m_dShader=0; m_geomShader.m_gShader=0; m_vtxStride = 0; 
		m_drawType = DRAW_INDEXED; m_nResources = 0; 
	}

	virtual
	void render(const DeviceDataDX11* deviceData, ID3D11Buffer* globalConstBuffer)
	{
		UINT stride = sizeof( VertexColorStruct );
		UINT offset = 0;
		deviceData->m_context->IASetVertexBuffers( 0, 1, &m_vtxBuffer, &stride, &offset );
		deviceData->m_context->IASetIndexBuffer( m_idxBuffer, DXGI_FORMAT_R32_UINT, 0 );
		deviceData->m_context->IASetPrimitiveTopology( m_topology );

		deviceData->m_context->VSSetShader( ( m_vertexShader.m_vShader )?
			m_vertexShader.m_vShader : g_defaultVertexShader.m_vShader, 0, 0 );
		deviceData->m_context->VSSetConstantBuffers( 0, 1, &globalConstBuffer );
		deviceData->m_context->HSSetShader( m_hullShader.m_hShader, 0, 0 );
		deviceData->m_context->DSSetShader( m_domainShader.m_dShader, 0, 0 );
		deviceData->m_context->PSSetShader( ( m_pixelShader.m_pShader)?
			m_pixelShader.m_pShader : g_defaultPixelShader.m_pShader, 0, 0 );

		deviceData->m_context->DrawIndexed( m_nIndices, m_idxOffset, m_vtxOffset );
	}

	void drawCall(const DeviceDataDX11* deviceData)
	{
		switch( m_drawType )
		{
		case DRAW_INDEXED:
			deviceData->m_context->DrawIndexed( m_nIndices, m_idxOffset, m_vtxOffset );
			break;
		case DRAW:
			deviceData->m_context->Draw( m_nIndices, m_vtxOffset );
			break;
		case DRAW_INSTANCED:
			deviceData->m_context->DrawInstanced(m_nIndices, m_nInstances, m_vtxOffset, 0);
			break;
		default:
			CLASSERT(0);
			break;
		};
	}

};

extern Array<RenderObject> g_debugRenderObj;
extern Array<RenderObject> g_renderObj;
extern AppendVertexBuffer* g_appDebugVertexBuffer;
extern AppendIndexBuffer* g_appDebugIndexBuffer;
extern AppendVertexBuffer* g_appVertexBuffer;
extern AppendIndexBuffer* g_appIndexBuffer;

__inline
void drawLine(XMFLOAT4& a, XMFLOAT4& b, XMFLOAT4& color)
{
	VertexColorStruct vtx[2];
	{
		vtx[0].m_pos = XMFLOAT3(a.x,a.y,a.z);
		vtx[0].m_normal = XMFLOAT3(0,0,0);
		vtx[0].m_color = color;

		vtx[1].m_pos = XMFLOAT3(b.x,b.y,b.z);
		vtx[1].m_normal = XMFLOAT3(0,0,0);
		vtx[1].m_color = color;
	}

	u32 idx[] = {0, 1};

	RenderObject& obj = g_debugRenderObj.expandOne();
	new (&obj)RenderObject();
	obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_LINELIST;
	obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
	obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
	obj.m_nIndices = 2;
	obj.m_vtxOffset = g_appDebugVertexBuffer->append( 2, vtx );
	obj.m_idxOffset = g_appDebugIndexBuffer->append( 2, idx );
}

__inline
void drawLine(XMFLOAT3& a, XMFLOAT3& b, XMFLOAT4& color)
{
	XMFLOAT4 aa, bb;
	aa = XMFLOAT4(a.x, a.y, a.z, 1);
	bb = XMFLOAT4(b.x, b.y, b.z, 1);
	drawLine( aa, bb, color );
}

__inline
void drawLineList(XMFLOAT4* vtx, u32* idx, int nVtx, int nIdx, XMFLOAT4& color)
{
	VertexColorStruct* vtxStruct = new VertexColorStruct[nVtx];
	for(int i=0; i<nVtx; i++)
	{
		vtxStruct[i].m_pos = XMFLOAT3(vtx[i].x, vtx[i].y, vtx[i].z);
		vtxStruct[i].m_normal = XMFLOAT3(0,0,0);
		vtxStruct[i].m_color = color;
	}

	RenderObject& obj = g_debugRenderObj.expandOne();
	new (&obj)RenderObject();
	obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_LINELIST;
	obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
	obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
	obj.m_nIndices = nIdx;
	obj.m_vtxOffset = g_appDebugVertexBuffer->append( nVtx, vtxStruct );
	obj.m_idxOffset = g_appDebugIndexBuffer->append( nIdx, idx );

	delete [] vtxStruct;
}

__inline
void drawPoint(XMFLOAT4& a, XMFLOAT4& color)
{
	VertexColorStruct vtx[1];
	{
		vtx[0].m_pos = XMFLOAT3(a.x,a.y,a.z);
		vtx[0].m_normal = XMFLOAT3(0,0,0);
		vtx[0].m_color = color;
	}
	u32 idx[] = {0};

	RenderObject& obj = g_debugRenderObj.expandOne();
	new (&obj)RenderObject();
	obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;
	obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
	obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
	obj.m_nIndices = 1;
	obj.m_vtxOffset = g_appDebugVertexBuffer->append( 1, vtx );
	obj.m_idxOffset = g_appDebugIndexBuffer->append( 1, idx );
}

__inline
void drawPoint(XMFLOAT3& a, XMFLOAT4& color)
{
	XMFLOAT4 aa;
	aa = XMFLOAT4(a.x, a.y, a.z, 1);
	drawPoint( aa, color );
}

__inline
void drawPointList(XMFLOAT4* vtx, XMFLOAT4* color, int nVtx)
{
	VertexColorStruct* vtxWc = new VertexColorStruct[nVtx];

	for(int i=0; i<nVtx; i++)
	{
		vtxWc[i].m_pos = XMFLOAT3(vtx[i].x, vtx[i].y, vtx[i].z);
		vtxWc[i].m_color = color[i];
	}

	RenderObject& obj = g_debugRenderObj.expandOne();
	new (&obj)RenderObject();
	obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;
	obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
	obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
	obj.m_nIndices = nVtx;
	obj.m_vtxOffset = g_appDebugVertexBuffer->append( nVtx, vtxWc );
	obj.m_idxOffset = 0;

	obj.m_drawType = RenderObject::DRAW;

	delete [] vtxWc;
}

__inline
void drawPointSprite(XMFLOAT4* vtx, XMFLOAT4* color, XMFLOAT2* radius, int nVtx)
{
	VertexColorStruct* vtxWc = new VertexColorStruct[nVtx];

	for(int i=0; i<nVtx; i++)
	{
		vtxWc[i].m_pos = XMFLOAT3(vtx[i].x, vtx[i].y, vtx[i].z);
		vtxWc[i].m_color = color[i];
		vtxWc[i].m_texCrd = radius[i];
	}

	RenderObject& obj = g_debugRenderObj.expandOne();
	new (&obj)RenderObject();
	obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;
	obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
	obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
	obj.m_nIndices = nVtx;
	obj.m_vtxOffset = g_appDebugVertexBuffer->append( nVtx, vtxWc );
	obj.m_idxOffset = 0;

	obj.m_drawType = RenderObject::DRAW;

//	obj.m_matrix.setParticleRadius( 0.1f );	// todo. remove this
	obj.m_pixelShader = g_pointSpritePixelShader;
	obj.m_geomShader = g_pointSpriteGeometryShader;

	delete [] vtxWc;
}

__inline
void drawPointSprite(DeviceBufferDX11& vtxBuffer, DeviceBufferDX11& colorBuffer, int nVtx, float radius)
{
	RenderObject& obj = g_debugRenderObj.expandOne();
	new (&obj)RenderObject();
	obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;
	obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
	obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
	obj.m_nIndices = 1;
	obj.m_vtxOffset = 0;
	obj.m_idxOffset = 0;


	obj.m_matrix.setParticleRadius( radius );
	obj.m_vertexShader = g_pointSpriteVertexShader;
	obj.m_pixelShader = g_pointSpritePixelShader;
	obj.m_geomShader = g_pointSpriteGeometryShader;

	obj.m_resources[0].m_buffer = vtxBuffer;
	obj.m_resources[0].m_type = RenderObject::Resource::VTX_SHADER;
	obj.m_resources[1].m_buffer = colorBuffer;
	obj.m_resources[1].m_type = RenderObject::Resource::VTX_SHADER;
	obj.m_nResources = 2;

	obj.m_drawType = RenderObject::DRAW_INSTANCED;
	obj.m_nInstances = nVtx;
}

__inline
void drawPointListTransformed(XMFLOAT4* vtx, XMFLOAT4* color, int nVtx, 
							  const float4& translation, const Quaternion& quat)
{
	//	index is 16 bits 
	const int BATCH_SIZE = 1024*64;

	VertexColorStruct* vtxWc = new VertexColorStruct[BATCH_SIZE];
	u32* idx = new u32[BATCH_SIZE];

	FXMVECTOR t = XMVectorSet(translation.x, translation.y, translation.z, 1);
	FXMVECTOR r = XMVectorSet(quat.x, quat.y, quat.z, quat.w);

	int counter = 0;
	for(int i=0; i<nVtx; i++)
	{
		vtxWc[counter].m_pos = XMFLOAT3(vtx[i].x, vtx[i].y, vtx[i].z);
		vtxWc[counter].m_color = color[i];
		idx[counter] = counter;
		counter++;
		if( counter == BATCH_SIZE )
		{
			RenderObject& obj = g_debugRenderObj.expandOne();
			new (&obj)RenderObject();

			{
				XMMATRIX rotM = XMMatrixRotationQuaternion( r );
				XMMATRIX trM = XMMatrixTranslationFromVector( t );
				XMMATRIX modelM = XMMatrixMultiply( rotM, trM );

				obj.m_matrix.m_world = XMMatrixTranspose( modelM );
			}


			obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;
			obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
			obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
			obj.m_nIndices = counter;
			obj.m_vtxOffset = g_appDebugVertexBuffer->append( counter, vtxWc );
			obj.m_idxOffset = g_appDebugIndexBuffer->append( counter, idx );

			counter = 0;
		}
	}
	if( counter )
	{
		RenderObject& obj = g_debugRenderObj.expandOne();
		new (&obj)RenderObject();

		{
			XMMATRIX rotM = XMMatrixRotationQuaternion( r );
			XMMATRIX trM = XMMatrixTranslationFromVector( t );
			XMMATRIX modelM = XMMatrixMultiply( rotM, trM );

			obj.m_matrix.m_world = XMMatrixTranspose( modelM );
		}

		obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;
		obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
		obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
		obj.m_nIndices = counter;
		obj.m_vtxOffset = g_appDebugVertexBuffer->append( counter, vtxWc );
		obj.m_idxOffset = g_appDebugIndexBuffer->append( counter, idx );
	}

	delete [] vtxWc;
	delete [] idx;

}

__inline
void drawTriangle(XMFLOAT3& a, XMFLOAT3& b, XMFLOAT3& c, XMFLOAT4& color)
{
	VertexColorStruct vtx[3];
	{
		vtx[0].m_pos = a;
		vtx[0].m_normal = XMFLOAT3(0,0,0);
		vtx[0].m_color = color;
		vtx[1].m_pos = b;
		vtx[1].m_normal = XMFLOAT3(0,0,0);
		vtx[1].m_color = color;
		vtx[2].m_pos = c;
		vtx[2].m_normal = XMFLOAT3(0,0,0);
		vtx[2].m_color = color;
	}
	u32 idx[] = {0,1,2};

	RenderObject& obj = g_debugRenderObj.expandOne();
	new (&obj)RenderObject();
	obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
	obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
	obj.m_nIndices = 3;
	obj.m_vtxOffset = g_appDebugVertexBuffer->append( 3, vtx );
	obj.m_idxOffset = g_appDebugIndexBuffer->append( 3, idx );
}

__inline
void drawTriangle(XMFLOAT4& a, XMFLOAT4& b, XMFLOAT4& c, XMFLOAT4& color)
{
	XMFLOAT3 aa,bb,cc;
	aa = XMFLOAT3(a.x, a.y, a.z);
	bb = XMFLOAT3(b.x, b.y, b.z);
	cc = XMFLOAT3(c.x, c.y, c.z);
	drawTriangle( aa, bb, cc, color );
}

__inline
void drawTriangleList(XMFLOAT4* vtx, u32* idx, int nVtx, int nIdx, XMFLOAT4& color)
{
	VertexColorStruct* vtxWc = new VertexColorStruct[nVtx];
	for(int i=0; i<nVtx; i++)
	{
		vtxWc[i].m_pos = XMFLOAT3(vtx[i].x, vtx[i].y, vtx[i].z);
		vtxWc[i].m_color = color;
	}

	RenderObject& obj = g_debugRenderObj.expandOne();
	new (&obj)RenderObject();
	obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
	obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
	obj.m_nIndices = nIdx;
	obj.m_vtxOffset = g_appDebugVertexBuffer->append( nVtx, vtxWc );
	obj.m_idxOffset = g_appDebugIndexBuffer->append( nIdx, idx );

	delete [] vtxWc;
}

__inline
void drawTriangleList1(XMFLOAT4* vtx, u32* idx, int nVtx, int nIdx, XMFLOAT4* color)
{
	VertexColorStruct* vtxWc = new VertexColorStruct[nVtx];
	for(int i=0; i<nVtx; i++)
	{
		vtxWc[i].m_pos = XMFLOAT3(vtx[i].x, vtx[i].y, vtx[i].z);
		vtxWc[i].m_color = color[i];
	}

	RenderObject& obj = g_debugRenderObj.expandOne();
	new (&obj)RenderObject();
	obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
	obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
	obj.m_nIndices = nIdx;
	obj.m_vtxOffset = g_appDebugVertexBuffer->append( nVtx, vtxWc );
	obj.m_idxOffset = g_appDebugIndexBuffer->append( nIdx, idx );

	delete [] vtxWc;
}

__inline
XMMATRIX getWorldMatrix(const float4& translation, const Quaternion& quat)
{
	FXMVECTOR t = XMVectorSet(translation.x, translation.y, translation.z, 1);
	FXMVECTOR r = XMVectorSet(quat.x, quat.y, quat.z, quat.w);

	XMMATRIX rotM = XMMatrixRotationQuaternion( r );
	XMMATRIX trM = XMMatrixTranslationFromVector( t );
	XMMATRIX modelM = XMMatrixMultiply( rotM, trM );

	return XMMatrixTranspose( modelM );
}

__inline
void drawTriangleListTessellated(XMFLOAT4* vtx, XMFLOAT4* n, u32* idx, int nVtx, int nIdx, XMFLOAT4& color, const float4& translation, const Quaternion& quat, 
								 DeviceShaderDX11* vtxShader, DeviceShaderDX11* hullShader, DeviceShaderDX11* domainShader, DeviceShaderDX11* pixelShader)
{
	VertexColorStruct* vtxWc = new VertexColorStruct[nVtx];
	for(int i=0; i<nVtx; i++)
	{
		vtxWc[i].m_pos = XMFLOAT3( vtx[i].x, vtx[i].y, vtx[i].z );
		vtxWc[i].m_normal = XMFLOAT3( n[i].x, n[i].y, n[i].z );
		vtxWc[i].m_color = color;
	}

	RenderObject& obj = g_debugRenderObj.expandOne();
	new (&obj)RenderObject();
	obj.m_matrix.m_world = getWorldMatrix( translation, quat );
	obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST;
	obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
	obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
	obj.m_nIndices = nIdx;
	obj.m_vtxOffset = g_appDebugVertexBuffer->append( nVtx, vtxWc );
	obj.m_idxOffset = g_appDebugIndexBuffer->append( nIdx, idx );

	if( vtxShader )
		obj.m_vertexShader = *vtxShader;
	if( hullShader )
		obj.m_hullShader = *hullShader;
	if( domainShader )
		obj.m_domainShader = *domainShader;
	if( pixelShader )
		obj.m_pixelShader = *pixelShader;

	delete [] vtxWc;
}

__inline
void drawTriangleList(XMFLOAT4* vtx, XMFLOAT4* vtxNormal, u32* idx, int nVtx, int nIdx, XMFLOAT4& color)
{
	VertexColorStruct* vtxWc = new VertexColorStruct[nVtx];
	for(int i=0; i<nVtx; i++)
	{
		vtxWc[i].m_pos = XMFLOAT3(vtx[i].x, vtx[i].y, vtx[i].z);
		vtxWc[i].m_normal = XMFLOAT3( vtxNormal[i].x, vtxNormal[i].y, vtxNormal[i].z );
		vtxWc[i].m_color = color;
	}

	RenderObject& obj = g_debugRenderObj.expandOne();
	new (&obj)RenderObject();
	obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
	obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
	obj.m_nIndices = nIdx;
	obj.m_vtxOffset = g_appDebugVertexBuffer->append( nVtx, vtxWc );
	obj.m_idxOffset = g_appDebugIndexBuffer->append( nIdx, idx );

	delete [] vtxWc;
}


template<bool DEBUG_OBJ>
__inline
void drawTriangleListTransformed(XMFLOAT4* vtx, XMFLOAT4* vtxNormal, u32* idx, int nVtx, int nIdx, XMFLOAT4& color,
								 const float4& translation, const Quaternion& quat)
{
//	FXMVECTOR t = XMVectorSet(translation.x, translation.y, translation.z, 1);
//	FXMVECTOR r = XMVectorSet(quat.x, quat.y, quat.z, quat.w);

	VertexColorStruct* vtxWc = new VertexColorStruct[nVtx];
	for(int i=0; i<nVtx; i++)
	{
		vtxWc[i].m_pos = XMFLOAT3(vtx[i].x, vtx[i].y, vtx[i].z);
		vtxWc[i].m_normal = XMFLOAT3( vtxNormal[i].x, vtxNormal[i].y, vtxNormal[i].z );
		vtxWc[i].m_color = color;
	}

//	{
//		XMMATRIX rotM = XMMatrixRotationQuaternion( r );
//		XMMATRIX trM = XMMatrixTranslationFromVector( t );
//		XMMATRIX modelM = XMMatrixMultiply( rotM, trM );
//
//		obj.m_matrix.m_world = XMMatrixTranspose( modelM );
//	}

	if( DEBUG_OBJ )
	{
		RenderObject& obj = g_debugRenderObj.expandOne();
		new (&obj)RenderObject();
		obj.m_matrix.m_world = getWorldMatrix( translation, quat );
		obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
		obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
		obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
		obj.m_nIndices = nIdx;
		obj.m_vtxOffset = g_appDebugVertexBuffer->append( nVtx, vtxWc );
		obj.m_idxOffset = g_appDebugIndexBuffer->append( nIdx, idx );
	}
	else
	{
		RenderObject& obj = g_renderObj.expandOne();
		new (&obj)RenderObject();
		obj.m_matrix.m_world = getWorldMatrix( translation, quat );
		obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
		obj.m_vtxBuffer = g_appVertexBuffer->getBuffer();
		obj.m_idxBuffer = g_appIndexBuffer->getBuffer();
		obj.m_nIndices = nIdx;
		obj.m_vtxOffset = g_appVertexBuffer->append( nVtx, vtxWc );
		obj.m_idxOffset = g_appIndexBuffer->append( nIdx, idx );
	}

	delete [] vtxWc;
}

__inline
void drawAabb(const Aabb& a, const float4& color)
{
	XMFLOAT4 p[] = { XMFLOAT4(a.m_min.x, a.m_min.y, a.m_min.z,0), XMFLOAT4(a.m_min.x, a.m_min.y, a.m_max.z,0),
		XMFLOAT4(a.m_max.x, a.m_min.y, a.m_max.z,0), XMFLOAT4(a.m_max.x, a.m_min.y, a.m_min.z,0),
		
		XMFLOAT4(a.m_min.x, a.m_max.y, a.m_min.z,0), XMFLOAT4(a.m_min.x, a.m_max.y, a.m_max.z,0),
		XMFLOAT4(a.m_max.x, a.m_max.y, a.m_max.z,0), XMFLOAT4(a.m_max.x, a.m_max.y, a.m_min.z,0) };

	u32 idx[] = {0,1,1,2,2,3,3,0,
		4,5,5,6,6,7,7,4,
		0,4,1,5,2,6,3,7};

	XMFLOAT4 c = XMFLOAT4(color.x, color.y, color.z, color.w);

	drawLineList( p, idx, 8, 4*2*3, c );
}



extern DeviceShaderDX11 g_quadVertexShader;
extern ID3D11SamplerState* g_defaultSampler;
extern DeviceBufferDX11 g_constBuffer;



__inline
RenderObject createQuad1(const float4& orig, const float4& extent, const float4& minTexCrd, const float4& maxTexCrd, float cx=1.f, float cy=1.f, float cz=1.f, float cw=1.f)
{
	float4 vtx[4] = {make_float4(orig.x,orig.y,orig.z),
		make_float4(orig.x+extent.x,orig.y,orig.z),
		make_float4(orig.x,orig.y+extent.y,orig.z),
		make_float4(orig.x+extent.x,orig.y+extent.y,orig.z)};
	u32 idxT[] = {0,2,1,3,1,2};

	VertexColorStruct vtxWc[4];
	for(int i=0; i<4; i++)
	{
		vtxWc[i].m_pos = XMFLOAT3(vtx[i].x, vtx[i].y, vtx[i].z);
		vtxWc[i].m_color = XMFLOAT4(cx,cy,cz,cw);
	}

	vtxWc[0].m_texCrd = XMFLOAT2(minTexCrd.x,maxTexCrd.y);
	vtxWc[1].m_texCrd = XMFLOAT2(maxTexCrd.x,maxTexCrd.y);
	vtxWc[2].m_texCrd = XMFLOAT2(minTexCrd.x,minTexCrd.y);
	vtxWc[3].m_texCrd = XMFLOAT2(maxTexCrd.x,minTexCrd.y);

	RenderObject obj;
	{
		obj.m_topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
		obj.m_vtxBuffer = g_appDebugVertexBuffer->getBuffer();
		obj.m_idxBuffer = g_appDebugIndexBuffer->getBuffer();
		obj.m_nIndices = 6;
		obj.m_vtxOffset = g_appDebugVertexBuffer->append( 4, vtxWc );
		obj.m_idxOffset = g_appDebugIndexBuffer->append( 6, idxT );

		obj.m_vertexShader = g_defaultVertexShader;
		obj.m_pixelShader = g_defaultPixelShader;
	}
	return obj;
}

__inline
RenderObject createQuad(float s, float cx=1.f, float cy=1.f, float cz=1.f, float cw=1.f)
{
	return createQuad1( make_float4(-s,-s,0.f), make_float4(2.f*s,2.f*s,0.f), make_float4(0,0,0), make_float4(1,1,1), cx, cy, cz, cw );
}

__inline
void renderFullQuad(const DeviceDataBase* deviceData, DeviceShaderDX11* pixelShader, const float4& constValues)
{
	DeviceDataDX11* dd = (DeviceDataDX11*)deviceData;

	RenderObject quad = createQuad(1);
	RenderObject* obj = &quad;
	obj->m_matrix.m_gData = constValues;
	
	obj->m_pixelShader = *pixelShader;
	obj->m_vertexShader = g_quadVertexShader;

	dd->m_context->PSSetSamplers( 0, 1, &g_defaultSampler );

	{
		DeviceKernelDX11 kernel;
		KernelLauncherDX11 launcher( deviceData, kernel );
		launcher.setConst<ConstantBuffer>( g_constBuffer, &obj->m_matrix );
	}

	obj->render( dd, g_constBuffer.m_buffer );
}
