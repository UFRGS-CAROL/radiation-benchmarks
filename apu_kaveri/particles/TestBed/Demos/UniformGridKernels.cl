typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;


///////////////////////////////////////
//	Vector
///////////////////////////////////////

#define make_float4 (float4)
#define make_int4 (int4)
#define make_float2 (float2)
#define make_int2 (int2)
#define max2 max
#define min2 min

///////////////////////////////////////
//	Vector
///////////////////////////////////////

__inline
float sqrtf(float a)
{
	return sqrt(a);
}

__inline
float4 cross3(float4 a, float4 b)
{
	return cross(a,b);
}

__inline
float dot3F4(float4 a, float4 b)
{
	float4 a1 = make_float4(a.xyz,0.f);
	float4 b1 = make_float4(b.xyz,0.f);
	return dot(a1, b1);
}

__inline
float length3(const float4 a)
{
	return sqrtf(dot3F4(a,a));
}

__inline
float dot4(const float4 a, const float4 b)
{
	return dot( a, b );
}

//	for height
__inline
float dot3w1(const float4 point, const float4 eqn)
{
	return dot3F4(point,eqn) + eqn.w;
}

__inline
float4 normalize3(const float4 a)
{
	float length = sqrtf(dot3F4(a, a));
	return 1.f/length * a;
}

__inline
float4 normalize4(const float4 a)
{
	float length = sqrtf(dot4(a, a));
	return 1.f/length * a;
}

__inline
float4 createEquation(const float4 a, const float4 b, const float4 c)
{
	float4 eqn;
	float4 ab = b-a;
	float4 ac = c-a;
	eqn = normalize3( cross3(ab, ac) );
	eqn.w = -dot3F4(eqn,a);
	return eqn;
}

#define GET_GROUP_IDX get_group_id(0)
#define GET_LOCAL_IDX get_local_id(0)
#define GET_GLOBAL_IDX get_global_id(0)

#include <TestBed/Demos/UniformGridDefines.h>
#include <TestBed/Demos/UniformGridFuncs.h>

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable


typedef struct
{
	float4 m_max;
	float4 m_min;
	int4 m_nCells;
	float m_gridScale;
	u32 m_maxParticles;
} ConstBuffer;


__kernel
void GridConstructionKernel( __global float4* gPosIn, __global int* gridG, __global int* gridCounterG,
							ConstBuffer cb )
{
	if( GET_GLOBAL_IDX >= cb.m_maxParticles ) return;

	float4 iPos = gPosIn[GET_GLOBAL_IDX];

	int4 gridCrd = ugConvertToGridCrd( iPos-cb.m_min, cb.m_gridScale );

	if( gridCrd.x < 0 || gridCrd.x >= cb.m_nCells.x 
		|| gridCrd.y < 0 || gridCrd.y >= cb.m_nCells.y
		|| gridCrd.z < 0 || gridCrd.z >= cb.m_nCells.z ) return;
	
	int gridIdx = ugGridCrdToGridIdx( gridCrd, cb.m_nCells.x, cb.m_nCells.y, cb.m_nCells.z );

	int count = atom_add(&gridCounterG[gridIdx], 1);

	if( count < MAX_IDX_PER_GRID )
	{
		gridG[ gridIdx*MAX_IDX_PER_GRID + count ] = GET_GLOBAL_IDX;
	}
}

__kernel
void GridClearKernel( __global int* gridCounterC, ConstBuffer cb )
{
	int4 m_nCells = cb.m_nCells;
	if( GET_GLOBAL_IDX >= m_nCells.x*m_nCells.y*m_nCells.z ) return;
	gridCounterC[ GET_GLOBAL_IDX ] = 0;
}
