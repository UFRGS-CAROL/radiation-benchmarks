#include <Demos/UniformGridDefines.h>
#include <Demos/UniformGrid.h>


template<typename DEVICEBUFFER, typename DEVICEKERNEL>
UniformGrid<DEVICEBUFFER,DEVICEKERNEL>::UniformGrid( const DeviceDataBase* deviceData, const Aabb& space, float cellSize, bool shareGPUCPU )
: m_deviceData( deviceData )
{
	{
		float4 extent = space.getExtent();
		int nx = int(extent.x/cellSize)+1;
		int ny = int(extent.y/cellSize)+1;
		int nz = int(extent.z/cellSize)+1;

		m_gProps.m_gridScale = 1.f/(cellSize);
		m_gProps.m_nCells = make_int4(nx, ny, nz);
		m_gProps.m_max = space.m_max;
		m_gProps.m_min = space.m_min;
	}

	{
		int numCells = m_gProps.m_nCells.x*m_gProps.m_nCells.y*m_gProps.m_nCells.z;
		m_gridCounter.template allocate<int>( m_deviceData, numCells, (shareGPUCPU)? DeviceBufferBase::BUFFER_CPU_GPU: DeviceBufferBase::BUFFER_RAW );
		m_grid.template allocate<int>( m_deviceData, numCells*MAX_IDX_PER_GRID, (shareGPUCPU)? DeviceBufferBase::BUFFER_CPU_GPU: DeviceBufferBase::BUFFER );
		m_cBuffer.template allocate<GridProperties>( m_deviceData, 1, DeviceBufferBase::BUFFER_CONST );
	}

	{
		const char *option = "-I ./";

		typename DUtilsBase<DEVICEBUFFER, DEVICEKERNEL>::Builder builder( m_deviceData, "TestBed/Demos/UniformGridKernels", option, true );
		builder.createKernel("GridClearKernel", m_gridClearKernel );
		builder.createKernel("GridConstructionKernel", m_gridConstructionKernel );
	}

}

template<typename DEVICEBUFFER, typename DEVICEKERNEL>
UniformGrid<DEVICEBUFFER,DEVICEKERNEL>::~UniformGrid()
{
	m_gridCounter.deallocate( m_deviceData );
	m_grid.deallocate( m_deviceData );
	m_cBuffer.deallocate( m_deviceData );

	DUtilsBase<DEVICEBUFFER, DEVICEKERNEL>::Builder::deleteKernel( m_deviceData, m_gridClearKernel );
	DUtilsBase<DEVICEBUFFER, DEVICEKERNEL>::Builder::deleteKernel( m_deviceData, m_gridConstructionKernel );
}

template<typename DEVICEBUFFER, typename DEVICEKERNEL>
void UniformGrid<DEVICEBUFFER,DEVICEKERNEL>::clearAndBuild(const DEVICEBUFFER &pos, int numPos)
{
	m_gProps.m_maxParticles = numPos;

	{	//	clear grid count
		int size = m_gProps.m_nCells.x*m_gProps.m_nCells.y*m_gProps.m_nCells.z;
		typename DUtilsBase<DEVICEBUFFER, DEVICEKERNEL>::Launcher launcher( m_deviceData, m_gridClearKernel );
		launcher.pushBackRW( m_gridCounter );
		launcher.setConst( m_cBuffer, &m_gProps );
		launcher.launch1D( size );
	}

	{	//	grid construction
		typename DUtilsBase<DEVICEBUFFER, DEVICEKERNEL>::Launcher launcher( m_deviceData, m_gridConstructionKernel );
		launcher.pushBackR( pos );
		launcher.pushBackRW( m_grid );
		launcher.pushBackRW( m_gridCounter );
		launcher.setConst( m_cBuffer, &m_gProps );
		launcher.launch1D( numPos );
	}
}

