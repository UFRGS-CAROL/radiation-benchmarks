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

struct HotspotExecute {
	Parameters setupParams;
	Log log;

	HotspotExecute(int argc, char** argv);
	virtual ~HotspotExecute();

	void run();
	void usage(int argc, char** argv);


private:
	template<typename full>
	void generic_execute(int size, double globaltime, double timestamp,
			int blockCols, int blockRows, int borderCols, int borderRows);

	template<typename full>
	int compute_tran_temp(DataManagement<full>& hotspot_data, int col, int row, int sim_time,
			int num_iterations, int blockCols, int blockRows, int borderCols,
			int borderRows, cudaStream_t streamm);
};


#endif /* HOTSPOTEXECUTE_H_ */
