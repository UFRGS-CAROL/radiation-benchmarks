long float_xor(float a, float b)
{
	long result;
		
	__asm__ volatile (
		"xor %1, %2 \n\t"
		"mov %2, %0"
		: "=r" (result)
		: "r" (a), "r" (b)
		);

	return result;
}	
