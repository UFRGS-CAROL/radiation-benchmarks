/*
 * HotspotExecute.h
 *
 *  Created on: 18/05/2019
 *      Author: fernando
 */

#ifndef HOTSPOTEXECUTE_H_
#define HOTSPOTEXECUTE_H_

#include "Parameters.h"
#include "Log.h"
#include "DataManagement.h"
#include "device_vector.h"

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

#define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline

/**
 * Hotspot needs to start values with float
 */
typedef float DefaultType;

struct HotspotExecute {
	HotspotExecute(Parameters& setup_parameters, Log& log);
	virtual ~HotspotExecute();
	void run();

private:
	Parameters& setup_params;
	Log& log;
	float flops;

	template<typename full, typename incomplete>
	void generic_execute(int blockCols, int blockRows, int borderCols,
			int borderRows);

	template<typename full, typename incomplete>
	int compute_tran_temp(rad::DeviceVector<full>& power_array,
			rad::DeviceVector<full>& temp_array_input,
			rad::DeviceVector<full>& temp_array_output,
			rad::DeviceVector<incomplete>& temp_array_output_incomplete,
			int col, int row,
			int sim_time, int num_iterations, int blockCols, int blockRows,
			int borderCols, int borderRows, cudaStream_t stream);
};

#endif /* HOTSPOTEXECUTE_H_ */
