

#define sizeofInt 4
#define make_float4 float4
#define make_float3 float3
#define make_int4 int4

typedef uint u32;
typedef uint u16;
typedef uint u8;

#define FLT_MAX         3.402823466e+38F

#define max2 max
#define min2 min



#define sqrtf sqrt
#define cross3(a,b) make_float4(cross((float3)a.xyz, (float3)b.xyz),0)
#define dot3F4(a, b)	dot(make_float4(a.xyz,0), make_float4(b.xyz,0))
#define dot4(a,b)		dot(a,b)
#define length3(a)		length(make_float4(a.xyz,0))
#define length4(a)		length(a)
#define dot3w1(point, eqn)	dot(make_float4(point.xyz,1), eqn)
#define normalize3(a)	normalize( make_float4(a.xyz,0) )
#define normalize4(a)	normalize(a)
float4 createEquation(float4 a,float4 b,float4 c)	{ float4 eqn=normalize3( cross3(b-a,c-a) ); eqn.w = -dot3F4(eqn,a); return eqn; }


//	Matrix
//#define Matrix3x3 float3x4
struct Matrix3x3
{
	float4 m_row[3];
};

Matrix3x3 mtZero()
{
	Matrix3x3 m;
	m.m_row[0] = make_float4(0,0,0,0);
	m.m_row[1] = make_float4(0,0,0,0);
	m.m_row[2] = make_float4(0,0,0,0);
	return m;
}

Matrix3x3 mtIdentity()
{
	Matrix3x3 m;
	m.m_row[0] = make_float4(1,0,0,0);
	m.m_row[1] = make_float4(0,1,0,0);
	m.m_row[2] = make_float4(0,0,1,0);
	return m;
}

Matrix3x3 mtTranspose(Matrix3x3 m)
{
	Matrix3x3 mtOut;
	mtOut.m_row[0] = make_float4(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x,0);
	mtOut.m_row[1] = make_float4(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y,0);
	mtOut.m_row[2] = make_float4(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z,0);
	return mtOut;
}

Matrix3x3 mtMul(Matrix3x3 a, Matrix3x3 b)
{
	Matrix3x3 transB;
	transB = mtTranspose( b );
	Matrix3x3 ans;
	for(int i=0; i<3; i++)
	{
		ans.m_row[i].x = dot3F4(a.m_row[i],transB.m_row[0]);
		ans.m_row[i].y = dot3F4(a.m_row[i],transB.m_row[1]);
		ans.m_row[i].z = dot3F4(a.m_row[i],transB.m_row[2]);
		ans.m_row[i].w = 0.f;
	}
	return ans;
}

float4 mtMul1(Matrix3x3 a, float4 b)
{
	float4 ans;
	ans.x = dot3F4( a.m_row[0], b );
	ans.y = dot3F4( a.m_row[1], b );
	ans.z = dot3F4( a.m_row[2], b );
	ans.w = 0.f;
	return ans;
}



//	Quaternion
#define Quaternion float4

Quaternion qtMul(Quaternion a, Quaternion b)
{
	Quaternion ans;
	ans = cross3( a, b );
	ans += a.w*b+b.w*a;
	ans.w = a.w*b.w - (a.x*b.x+a.y*b.y+a.z*b.z);
	return ans;
}

Quaternion qtNormalize(Quaternion qtIn)
{
	qtIn /= length4( qtIn );
	return qtIn;
}

Quaternion qtInvert(Quaternion q)
{
	Quaternion qInv = -q;
	qInv.w = q.w;
	return qInv;
//	return (Quaternion)(-q.xyz, q.w);
}

float4 qtRotate(Quaternion q, float4 vec)
{
	Quaternion qInv = q;
	float4 vcpy = vec;
	vcpy.w = 0.f;
	float4 qtOut = qtMul(qtMul(q,vcpy),qInv);
	return qtOut;
}

Matrix3x3 qtGetRotationMatrix(Quaternion quat)
{
	float4 quat2 = make_float4(quat.x*quat.x, quat.y*quat.y, quat.z*quat.z, 0.f);
	Matrix3x3 qtOut;

	qtOut.m_row[0].x=1-2*quat2.y-2*quat2.z;
	qtOut.m_row[0].y=2*quat.x*quat.y-2*quat.w*quat.z;
	qtOut.m_row[0].z=2*quat.x*quat.z+2*quat.w*quat.y;
	qtOut.m_row[0].w=0;

	qtOut.m_row[1].x=2*quat.x*quat.y+2*quat.w*quat.z;
	qtOut.m_row[1].y=1-2*quat2.x-2*quat2.z;
	qtOut.m_row[1].z=2*quat.y*quat.z-2*quat.w*quat.x;
	qtOut.m_row[1].w=0;

	qtOut.m_row[2].x=2*quat.x*quat.z-2*quat.w*quat.y;
	qtOut.m_row[2].y=2*quat.y*quat.z+2*quat.w*quat.x;
	qtOut.m_row[2].z=1-2*quat2.x-2*quat2.y;
	qtOut.m_row[2].w=0;

	return qtOut;
}

float4 transform(const float4 p, const float4 translation, const Quaternion orientation)
{
	return qtRotate( orientation, p ) + (translation);
}

float4 invTransform(const float4 p, const float4 translation, const Quaternion orientation)
{
	return qtRotate( qtInvert( orientation ), (p)-(translation) ); // use qtInvRotate
}


//	Sort Data
struct SortData
{
	u32 m_key;
	u32 m_value;
};



#define GET_GROUP_IDX gIdx.x
#define GET_LOCAL_IDX lIdx.x
#define GET_GLOBAL_IDX iIdx.x
#define GROUP_LDS_BARRIER GroupMemoryBarrierWithGroupSync()
