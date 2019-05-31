#include "xor_asm.h"

int int_xor(int a, int b)
{
	int result;
		
	__asm__ volatile (
		"xor %1, %2 \n\t"
		"mov %2, %0"
		: "=r" (result)
		: "r" (a), "r" (b)
		);

	return result;
}	

long long_xor(long a, long b)
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

int float_xor(float a, float b)
{
	int result;
		
	__asm__ volatile (
		"xor %1, %2 \n\t"
		"mov %2, %0"
		: "=r" (result)
		: "r" (a), "r" (b)
		);

	return result;
}	

long double_xor(double a, double b)
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

void* ptr_xor(void* a, void* b)
{
	void* result;
		
	__asm__ volatile (
		"xor %1, %2 \n\t"
		"mov %2, %0"
		: "=r" (result)
		: "r" (a), "r" (b)
		);

	return result;
}
