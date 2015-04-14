#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
typedef struct c{
	double real;
	double imag;
//	int stage;
} Complex;

typedef struct{
	int n;
	int stage;
	int unit_size;
}Cal_Unit;

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
__kernel void fft_unit(__global Cal_Unit *unit, __global Complex *factor, __global Complex *odata)
{

	int threadID=get_global_id(0);

	int start = threadID*unit->n;
	int end = start + unit->n;
	int size = unit->n;

	for( int jump = 1; jump < size; jump <<= 1 )
	{
		int num = jump << 1;

		for ( int i = 0; i < jump; i++ )
		{
			int shifted_start = start + i;
			for ( int pair = shifted_start; pair < end ; pair += num) 
			{
				int match = pair + jump;

				Complex twiddle = factor[(size/num)*(pair%num)];
				Complex temp = {twiddle.real*odata[match].real - twiddle.imag*odata[match].imag, twiddle.real*odata[match].imag + twiddle.imag*odata[match].real};
				odata[match].real = odata[pair].real - temp.real;
				odata[match].imag = odata[pair].imag - temp.imag;
				odata[pair].real = odata[pair].real + temp.real;
				odata[pair].imag = odata[pair].imag + temp.imag;

			}
		}
	}

}

