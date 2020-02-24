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
std::vector<int> BFSGraph(std::vector<Node>& h_graph_nodes,
		std::vector<bool_t>& h_graph_mask,
		std::vector<bool_t>& h_updating_graph_mask,
		std::vector<bool_t>& h_graph_visited, std::vector<int>& h_graph_edges,
		int no_of_nodes, int source) {

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
		num_of_blocks = (int) ceil(
				no_of_nodes / (double) MAX_THREADS_PER_BLOCK);
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
	}

	//Copy the Node list to device memory
	rad::DeviceVector<Node> d_graph_nodes = h_graph_nodes;

	//Copy the Edge List to device Memory
	rad::DeviceVector<int> d_graph_edges = h_graph_edges;

	//Copy the Mask to device memory
	rad::DeviceVector<bool_t> d_graph_mask = h_graph_mask;
	rad::DeviceVector<bool_t> d_updating_graph_mask = h_updating_graph_mask;

	//Copy the Visited nodes array to device memory
	rad::DeviceVector<bool_t> d_graph_visited = h_graph_visited;

	// allocate mem for the result on host side
	std::vector<int> h_cost(no_of_nodes);
	for (int i = 0; i < no_of_nodes; i++)
		h_cost[i] = -1;
	h_cost[source] = 0;

	// allocate device memory for result
	rad::DeviceVector<int> d_cost = h_cost;

	//make a bool_t to check if the execution is over
	bool_t *d_over;
	cudaMalloc((void**) &d_over, sizeof(bool_t));

	std::cout << ("Copied Everything to GPU memory\n");

	// setup execution parameters
	dim3 grid(num_of_blocks, 1, 1);
	dim3 threads(num_of_threads_per_block, 1, 1);

	int k = 0;
	std::cout << ("Start traversing the tree\n");
	bool_t stop;
	//Call the Kernel untill all the elements of Frontier are not FALSE
	do {
		//if no thread changes this value then the loop stops
		stop = FALSE;
		cudaMemcpy(d_over, &stop, sizeof(bool_t), cudaMemcpyHostToDevice);
		Kernel<<<grid, threads, 0>>>(d_graph_nodes.data(), d_graph_edges.data(),
				d_graph_mask.data(), d_updating_graph_mask.data(),
				d_graph_visited.data(), d_cost.data(), no_of_nodes);
		// check if kernel execution generated and error

		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		;

		Kernel2<<<grid, threads, 0>>>(d_graph_mask.data(),
				d_updating_graph_mask.data(), d_graph_visited.data(), d_over,
				no_of_nodes);
		// check if kernel execution generated and error
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		;

		cudaMemcpy(&stop, d_over, sizeof(bool_t), cudaMemcpyDeviceToHost);
		k++;
	} while (stop);

	std::cout << "Kernel Executed " << k << " times\n";

	rad::checkFrameworkErrors(cudaPeekAtLastError());
	;

	// copy result from device to host
	auto copy_time = rad::mysecond();
	d_cost.to_vector(h_cost);
	copy_time = rad::mysecond() - copy_time;
	std::cout << "Copy time " << copy_time << std::endl;
//	cudaMemcpy(h_cost.data(), d_cost.data(), sizeof(int) * no_of_nodes,
//			cudaMemcpyDeviceToHost);

	return h_cost;
}
