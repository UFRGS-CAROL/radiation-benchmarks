/*
 * CacheProfiler.h
 *
 *  Created on: Feb 6, 2019
 *      Author: carol
 */

#ifndef CACHEPROFILER_H_
#define CACHEPROFILER_H_

#include "cupti_profiler.h"
#include <vector>
#include <string>
#include "kernels.h"

class CacheProfiler {
private:
	  cupti_profiler::profiler* profiler;

	  std::vector<std::string> event_names;
	  std::vector<std::string> metric_names;
	  std::vector<std::string> event_data;
	  std::vector<std::string> metric_data;

public:
	CacheProfiler(std::string cache_memory, Board device);
	virtual ~CacheProfiler();

	void start();
	void stop();

	std::string get_data();
};

#endif /* CACHEPROFILER_H_ */
