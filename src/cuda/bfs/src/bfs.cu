/***********************************************************************************
 Implementing Breadth first search on CUDA using algorithm given in HiPC'07
 paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

 Copyright (c) 2008 International Institute of Information Technology - Hyderabad.
 All rights reserved.

 Permission to use, copy, modify and distribute this software and its documentation for
 educational purpose is hereby granted without fee, provided that the above copyright
 notice and this permission notice appear in all copies of this software and that you do
 not sell the software.

 THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR
 OTHERWISE.

 Created by Pawan Harish.
 ************************************************************************************/
#include <cuda.h>

#include <iostream>

#include "kernel.h"
#include "cuda_utils.h"
#include "device_vector.h"

////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
int BFSGraph(rad::DeviceVector<Node>& d_graph_nodes,
		rad::DeviceVector<bool_t>& d_graph_mask,
		rad::DeviceVector<bool_t>& d_updating_graph_mask,
		rad::DeviceVector<bool_t>& d_graph_visited,
		rad::DeviceVector<int>& d_graph_edges, rad::DeviceVector<int>& d_cost,
		cudaStream_t& stream, int no_of_nodes) {

	//make a bool_t to check if the execution is over
	static rad::DeviceVector<bool_t> d_over(1);
	static std::vector<bool_t> stop(1);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
		num_of_blocks = (int) ceil(
				no_of_nodes / (double) MAX_THREADS_PER_BLOCK);
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
	}

	// setup execution parameters
	dim3 grid(num_of_blocks, 1, 1);
	dim3 threads(num_of_threads_per_block, 1, 1);

	int k = 0;
	//Call the Kernel untill all the elements of Frontier are not FALSE
	do {
		//if no thread changes this value then the loop stops
		stop[0] = FALSE;
		d_over = stop;
		Kernel<<<grid, threads, 0, stream>>>(d_graph_nodes.data(),
				d_graph_edges.data(), d_graph_mask.data(),
				d_updating_graph_mask.data(), d_graph_visited.data(),
				d_cost.data(), no_of_nodes);

		Kernel2<<<grid, threads, 0, stream>>>(d_graph_mask.data(),
				d_updating_graph_mask.data(), d_graph_visited.data(),
				d_over.data(), no_of_nodes);

		d_over.to_vector(stop);
		k++;
	} while (stop[0]);

	rad::checkFrameworkErrors(cudaPeekAtLastError());
	;

	return k;
}
