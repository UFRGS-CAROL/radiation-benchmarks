#ifndef D_UNIFORM_GRID_H
#define D_UNIFORM_GRID_H

#include <Common/Math/Math.h>
#include <Common/Geometry/Aabb.h>
#include <Common/DeviceUtils/DeviceUtils.h>


template<typename DEVICEBUFFER, typename DEVICEKERNEL>
class UniformGrid
{
	public:
		__inline
		UniformGrid( const DeviceDataBase* deviceData, const Aabb& space, float cellSize, bool shareGPUCPU = false );
		__inline
		~UniformGrid();

		__inline
		void clearAndBuild( const DEVICEBUFFER& pos, int numPos );

	public:
		struct GridProperties
		{
			float4 m_max;
			float4 m_min;
			int4 m_nCells;
			float m_gridScale;

			//	runtime data
			int m_maxParticles;
		};

		GridProperties m_gProps;
		const DeviceDataBase* m_deviceData;

		DEVICEBUFFER m_gridCounter;
		DEVICEBUFFER m_grid;
		DEVICEBUFFER m_cBuffer;

		DEVICEKERNEL m_gridClearKernel;
		DEVICEKERNEL m_gridConstructionKernel;

};

#include <Demos/UniformGrid.inl>


#endif
