__kernel void scan(__global double *o0, __global double *o1, __global double *o2, __global double *o3,
			__global double *o4, __global double *o5, __global double *o6, __global double *o7,
			__global double *i0, __global double *i1, __global double *i2, __global double *i3,
			__global double *i4, __global double *i5, __global double *i6, __global double *i7,
			int size, int offset)  
{  
	int tx = get_global_id(0);

	if(tx < 256) {
		if(tx >= offset) 
			o0[tx] = i0[tx] + i0[tx - offset]; 
		
		else
			o0[tx] = i0[tx];
	}

	else if(tx < 512) {
		if(tx-256 >= offset)
			 o1[tx-256] = i1[tx-256] + i1[tx-256 - offset]; 

		else
			 o1[tx-256] = i1[tx-256];
	}

	else if(tx < 768) {
		if(tx-512 >= offset)
			 o2[tx-512] = i2[tx-512] + i2[tx-512 - offset]; 
	
		else
			 o2[tx-512] = i2[tx-512];
	}

	else if(tx < 1024) {
		if(tx-768 >= offset)
		         o3[tx-768] = i3[tx-768] + i3[tx-768 - offset]; 
	
		else
			 o3[tx-768] = i3[tx-768];
	}

	else if(tx < 1280) {
		if(tx-1024 >= offset)
			 o4[tx-1024] = i4[tx-1024] + i4[tx-1024 - offset]; 
	
		else
		 	 o4[tx-1024] = i4[tx-1024];
	}

	else if(tx < 1536) {
		if(tx-1280 >= offset)
			 o5[tx-1280] = i5[tx-1280] + i5[tx-1280 - offset]; 
	
		else
			 o5[tx-1280] = i5[tx-1280];
	}

	else if(tx < 1792) {
		if(tx-1536 >= offset) 
			 o6[tx-1536] = i6[tx-1536] + i6[tx-1536 - offset]; 
	
		else
			 o6[tx-1536] = i6[tx-1536];
	}

	else if(tx < 2048) {
		if(tx-1792 >= offset)
			 o7[tx-1792] = i7[tx-1792] + i7[tx-1792 - offset];
	
		else
			 o7[tx-1792] = i7[tx-1792];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);	


	if(tx < 256) {
		i0[tx] = o0[tx];
	}

	else if(tx < 512) {
	        i1[tx-256] = o1[tx-256];
	}

	else if(tx < 768) {
		i2[tx-512] = o2[tx-512];
	}

	else if(tx < 1024) {
		i3[tx-768] = o3[tx-768];
	}

	else if(tx < 1280) {
		i4[tx-1024] = o4[tx-1024];
	}

	else if(tx < 1536) {
		i5[tx-1280] = o5[tx-1280];
	}

	else if(tx < 1792) {
		i6[tx-1536] = o6[tx-1536];
	}

	else if(tx < 2048) {
		i7[tx-1792] = o7[tx-1792];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
}


