/*
		2011 Takahiro Harada
*/
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <Common/Math/Math.h>

class ThreadPool
{
	public:
		struct Task
		{
			virtual u16 getType() = 0;
			virtual void run(int tIdx) = 0;
		};

		void resetThreadTimer();
		void start(bool resetTimestamp = true);
		void wait();
		void pushBack(Task* task);
		Task* pop();
};

#endif
