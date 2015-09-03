/*
		2011 Takahiro Harada
*/
#ifndef RADIX_SORT_HOST_H
#define RADIX_SORT_HOST_H

#include <Common/Math/Math.h>

class RadixSortHost
{
	public:
		static void sort(SortData* src, SortData* workBuffer, int numElems);

		enum
		{
			BITS_PER_PASS = 8, 
			NUM_TABLES = (1<<BITS_PER_PASS),
		};
};

#endif

