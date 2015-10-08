typedef struct c{
	double real;
	double imag;
} Complex;

typedef struct{
	int n;
	int stage;
	int unit_size;
}Cal_Unit;

__kernel void fft_unit(__global Cal_Unit *unit, __global Complex *factor, __global Complex *odata)
{

		int threadID=get_global_id(0);
//odata[threadID].real = threadID;

		int unit_start = threadID*unit->unit_size;
		int unit_end = unit_start + unit->unit_size;

		int unit_num = 1 << unit->stage;
		int jump = unit_num>>1;


		int pair;

			for(int i = unit_start; i < unit_end; i+= unit_num)
			{

				for(pair = i; pair <i+jump; pair++ )
				{
					int match = pair + jump;

					Complex twiddle = factor[(unit->n/unit_num)*(pair%unit_num)];
					Complex temp = {twiddle.real*odata[match].real - twiddle.imag*odata[match].imag, twiddle.real*odata[match].imag + twiddle.imag*odata[match].real};

					odata[match].real = odata[pair].real - temp.real;
					odata[match].imag = odata[pair].imag - temp.imag;
					odata[pair].real = odata[pair].real + temp.real;
					odata[pair].imag = odata[pair].imag + temp.imag;
					
				}
			}

}

