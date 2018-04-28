/*
 * copy_network.h
 *
 *  Created on: Mar 1, 2018
 *      Author: carol
 */

#ifndef COPY_NETWORK_H_
#define COPY_NETWORK_H_

void copy_network_content_to_buffer(network *mt_net, network_state *mt_state,
		network *buffer_nets, network_state *buffer_states, int thread_id,
		int mr_size, int start_layer);

#endif /* COPY_NETWORK_H_ */
