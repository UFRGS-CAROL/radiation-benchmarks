/*
		2011 Takahiro Harada
*/
#include <stdlib.h>
#include <stdio.h>

#include <assert.h>
#include <math.h>

#include <Demos/Dem2Demo.h>

#define ITERATIONS 100000
#define FINAL_STATE_ITERATIONS 1000

typedef Demo* CreateFunc(const DeviceDataBase* deviceData, int numL_part, int numS_part, int numF_part);

CreateFunc* createFuncs[] = {
	Dem2Demo::createFunc,
};


Demo* m_pDemo;
DeviceDataBase* deviceData = NULL;

void initDemo(int numL_part, int numS_part, int numF_part )
{	
	m_pDemo = NULL;
	m_pDemo = createFuncs[0]( deviceData, numL_part, numS_part, numF_part );

	CLASSERT( m_pDemo );
	m_pDemo->init();
	m_pDemo->reset();
}

int main(int argc, char *argv[])
{
	for (int iter = 0; iter < ITERATIONS; iter++)
	{
		//if (iter % 10 == 0)
			printf("iteration %d\n", iter);

		initDemo(100, 1024, 16);
		m_pDemo->resetDemo();
		for (int i = 0; i < FINAL_STATE_ITERATIONS; i++)
		{
			m_pDemo->stepDemo();
		}
	}

	return 0;
}
