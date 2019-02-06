/*
 * CacheProfiler.cpp
 *
 *  Created on: Feb 6, 2019
 *      Author: carol
 */

#include "CacheProfiler.h"
#include <sstream>

CacheProfiler::CacheProfiler(std::string cache_memory, Board device) {
	switch (device) {
	case K40:
		this->metric_names = {
			"l1_cache_global_hit_rate",
			"l1_cache_local_hit_rate",
			"tex_cache_hit_rate",
			"l2_l1_read_hit_rate",
			"l2_texture_read_hit_rate",
			"l2_utilization",
			"sm_efficiency",
			"achieved_occupancy",
			"l1_shared_utilization"
		};

		this->event_names = {
//			"l1_local_load_hit",
//			"l1_local_load_miss",
//			"l1_local_store_hit",
//			"l1_local_store_miss",
//			"l1_global_load_hit",
//			"l1_global_load_miss",
//			"l2_subp0_write_sector_misses",
//			"l2_subp1_write_sector_misses",
//			"l2_subp2_write_sector_misses",
//			"l2_subp3_write_sector_misses",
//			"l2_subp0_read_sector_misses",
//			"l2_subp1_read_sector_misses",
//			"l2_subp2_read_sector_misses",
//			"l2_subp3_read_sector_misses",
//			"l2_subp0_write_l1_sector_queries",
//			"l2_subp1_write_l1_sector_queries",
//			"l2_subp2_write_l1_sector_queries",
//			"l2_subp3_write_l1_sector_queries",
//			"l2_subp0_read_l1_sector_queries",
//			"l2_subp1_read_l1_sector_queries",
//			"l2_subp2_read_l1_sector_queries",
//			"l2_subp3_read_l1_sector_queries",
//			"l2_subp0_read_l1_hit_sectors",
//			"l2_subp1_read_l1_hit_sectors",
//			"l2_subp2_read_l1_hit_sectors",
//			"l2_subp3_read_l1_hit_sectors"
		};

		break;

		case TITANV:
		this->metric_names = {
			"global_hit_rate", //l1_cache_global_hit_rate for kepler
			"local_hit_rate",//l1_cache_local_hit_rate for kepler
			"tex_cache_hit_rate",
			"l2_global_load_bytes",//"l2_l1_read_hit_rate",
			"l2_local_load_bytes",//"l2_texture_read_hit_rate",
			"l2_utilization",
			"sm_efficiency",
			"achieved_occupancy",//maybe on kepler
			"shared_utilization",//l1_shared_utilization
		};

		this->event_names = {
			"l2_subp0_write_sector_misses",
			"l2_subp1_write_sector_misses",
			"l2_subp0_read_sector_misses",
			"l2_subp1_read_sector_misses",
		};
		break;
	}

	this->profiler =  new cupti_profiler::profiler(this->event_names, this->metric_names);
}
//set profiler object

CacheProfiler::~CacheProfiler() {
// TODO Auto-generated destructor stub
	if(this->profiler)
		delete this->profiler;
}

void CacheProfiler::start() {
	this->profiler->start();
}

void CacheProfiler::stop() {
	this->profiler->stop();
}

std::string CacheProfiler::get_data() {
	  std::stringstream ss;
//	  this->profiler->print_event_values<std::stringstream>(ss);
	  ss << std::endl;
	  this->profiler->print_metric_values<std::stringstream>(ss);

	  return ss.str();
}
