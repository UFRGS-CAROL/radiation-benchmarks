/*
		2011 Takahiro Harada
*/
#include <Common/Math/RadixSortHost.h>

void RadixSortHost::sort(SortData* data, SortData* workBuffer, int numElems)
{
	int tables[NUM_TABLES];
	int counter[NUM_TABLES];

	SortData* src = data;
	SortData* dst = workBuffer;

	for(int startBit=0; startBit<32; startBit+=BITS_PER_PASS)
	{
		for(int i=0; i<NUM_TABLES; i++)
		{
			tables[i] = 0;
		}

		for(int i=0; i<numElems; i++)
		{
			int tableIdx = (src[i].m_key >> startBit) & (NUM_TABLES-1);
			tables[tableIdx]++;
		}

		//	prefix scan
		int sum = 0;
		for(int i=0; i<NUM_TABLES; i++)
		{
			int iData = tables[i];
			tables[i] = sum;
			sum += iData;
			counter[i] = 0;
		}

		//	distribute
		for(int i=0; i<numElems; i++)
		{
			int tableIdx = (src[i].m_key >> startBit) & (NUM_TABLES-1);
			
			dst[tables[tableIdx] + counter[tableIdx]] = src[i];
			counter[tableIdx] ++;
		}

		swap2( src, dst );
	}
}

