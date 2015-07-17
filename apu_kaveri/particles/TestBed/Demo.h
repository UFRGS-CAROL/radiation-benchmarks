/*
		2011 Takahiro Harada
*/
#ifndef TEST_DEMO_H
#define TEST_DEMO_H

#include <Common/Math/Math.h>
#include <Common/Math/Array.h>
#include <Common/Base/ThreadPool.h>
#include <stdio.h>

#include <Common/DeviceUtils/DeviceDraw.h>

#define ADD_EXTENSION(deviceData, fullPath, fileName) \
	if( deviceData->m_type == DeviceData::TYPE_DX11) sprintf_s(fullPath, 256, "%s.hlsl", fileName);\
	else sprintf_s(fullPath, 256, "%s.cl", fileName);

class Demo
{
	public:
		__inline
		Demo(int nThreads = 0);

		__inline
		Demo( const DeviceDataBase* deviceData, DeviceUtils::DriverType driverType = DeviceUtils::DRIVER_HARDWARE, int nThreads = 0 );

		__inline
		virtual ~Demo();

		void stepDemo(){ m_stepCount++; step(m_dt); }

		void resetDemo() { m_stepCount=0; reset(); }

		virtual void init(){}

		virtual void step(float dt){ m_stepCount++; }

		virtual void render(){}

		virtual void renderPre(){}

		virtual void renderPost(){}

		virtual void reset(){}

		virtual void keyListener(unsigned char key){}

		virtual void keySpecialListener(unsigned char key){}

	public:
		const DeviceDataBase* m_deviceData;
		bool m_ddCreated;

		ThreadPool* m_threads;

		float m_dt;
		int m_stepCount;
		bool m_enableLighting;
		bool m_enablePostEffect;
		bool m_enableAlphaBlending;
		float4 m_backgroundColor;


		enum
		{
			MAX_LINES = 40,
			LINE_CAPACITY = 512,
		};

		int m_nTxtLines;
		char m_txtBuffer[MAX_LINES][LINE_CAPACITY];
};

Demo::Demo(int nThreads): 
m_dt(1.f/60.f), m_stepCount(0), m_enableLighting(true), m_enablePostEffect(false), m_enableAlphaBlending(false), m_nTxtLines(0)
{
	m_ddCreated = false;
	m_deviceData = 0;

	m_backgroundColor = make_float4(0.f);
}


Demo::Demo( const DeviceDataBase* deviceData, DeviceUtils::DriverType driverType, int nThreads )
: m_dt(1.f/60.f), m_stepCount(0)
{
	m_ddCreated = true;

	if( deviceData )
	{
		if( deviceData->m_type == DeviceData::TYPE_CL )
		{
			m_deviceData = deviceData;
			m_ddCreated = false;
		}
	}

	if( m_ddCreated )
	{
		m_deviceData = new DeviceData;
		DeviceUtils::initDevice( (DeviceData*)m_deviceData, driverType );
	}

	m_nTxtLines = 0;

	m_backgroundColor = make_float4(0.f);
}

Demo::~Demo()
{
	if( m_ddCreated )
	{
		DeviceUtils::releaseDevice( (DeviceData*)m_deviceData );
		delete m_deviceData;
	}

	if( m_threads )
	{
		delete m_threads;
	}
}

#endif
