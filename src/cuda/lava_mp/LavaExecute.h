/*
 * LavaExecute.h
 *
 *  Created on: 22/05/2019
 *      Author: fernando
 */

#ifndef LAVAEXECUTE_H_
#define LAVAEXECUTE_H_
#include "Parameters.h"
#include "Log.h"
#include "DataManagement.h"


class LavaExecute {

	Parameters& setup_parameters;
	Log& log;

	template<typename full, typename incomplete>
	void generic_execute();

public:


	LavaExecute(Parameters& setup_parameters, Log& log);
	virtual ~LavaExecute();

	void execute();


};



#endif /* LAVAEXECUTE_H_ */
