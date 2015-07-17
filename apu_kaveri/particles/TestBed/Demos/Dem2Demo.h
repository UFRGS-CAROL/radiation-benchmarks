/*
		2011 Takahiro Harada
*/
#ifndef DEM_2_DEMO_H
#define DEM_2_DEMO_H

#include <Demo.h>
#include <Common/DeviceUtils/DeviceUtils.h>
#include <Demos/UniformGrid.h>

#define FINAL_STATE_ITERATIONS 1000

class Dem2Demo : public Demo
{
	public:
		COMPUTE_CL;
		static Demo* createFunc(const DeviceDataBase* deviceData, int numL_part, int numS_part, int numF_part)
		{
			return new Dem2Demo(deviceData, numL_part, numS_part, numF_part);
		}

		Dem2Demo(const DeviceDataBase* deviceData, int numL_part, int numS_part, int numF_part);
		~Dem2Demo();


		void init();

		void reset();

		void step(float dt);

		enum
		{
			USE_ZERO_COPY = 0,
		};

	public:
		int m_numSParticles;
		int m_numLParticles;
		int m_numFParticles;

		//	pos.w : radius
		//	vel.w : mass

		float4* m_posL;
		float4* m_velL;
		float4* m_forceL;
		float4* m_posS;
		float4* m_velS;
		float4* m_forceS;
		float4* m_forceSHost;

		DMapBuffer m_posMapped;
		DMapBuffer m_velMapped;
		DMapBuffer m_forceMapped;
		DMapBuffer m_gridMapped;
		DMapBuffer m_gridCounterMapped;


		DBuffer m_posD;
		DBuffer m_velD;
		DBuffer m_forceD;
		DBuffer m_forceDInt;
		DBuffer m_sBuffer;
		DKernel m_collisionKernel;
		DKernel m_integrateKernel;

		DBuffer m_constBuffer;

		UniformGrid<DBuffer, DKernel>* m_grid;

		bool m_gridIsDirty;


		typedef struct
		{
			float4 m_g;
			int m_numParticles;
			float m_dt;
			float m_scale;
			float m_e;

			int4 m_nCells;
			float4 m_spaceMin;
			float m_gridScale;
		}ConstBuffer;

		enum
		{
			NUM_PLANES = 4,
		};
		float4 m_planes[NUM_PLANES];

		float m_scale;
		
		int iter_count;
};


#endif
