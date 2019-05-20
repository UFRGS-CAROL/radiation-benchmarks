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

#define BLOCK_SIZE 16

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline

struct HotspotExecute {
	HotspotExecute(Parameters& setup_parameters);
	virtual ~HotspotExecute();
	void run();

private:
	Parameters& setup_params;
	Log log;

	template<typename full>
	void generic_execute(int blockCols, int blockRows, int borderCols,
			int borderRows);

	template<typename full>
	int compute_tran_temp(DeviceVector<full>& power_array,
			DeviceVector<full>& temp_array_input,
			DeviceVector<full>& temp_array_output, int col, int row,
			int sim_time, int num_iterations, int blockCols, int blockRows,
			int borderCols, int borderRows, cudaStream_t streamm,
			double& flops);
};

#endif /* HOTSPOTEXECUTE_H_ */
