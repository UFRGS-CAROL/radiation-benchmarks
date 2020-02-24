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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include <vector>

#include "kernel.h"

#include "cuda_utils.h"
#include "device_vector.h"

void Usage(int argc, char**argv) {

	fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char** argv) {
	int no_of_nodes = 0;
	int edge_list_size = 0;
	FILE *fp;

	char *input_f;
	if (argc != 2) {
		Usage(argc, argv);
		exit(0);
	}

	input_f = argv[1];
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f, "r");
	if (!fp) {
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp, "%d", &no_of_nodes);

	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
		num_of_blocks = (int) ceil(
				no_of_nodes / (double) MAX_THREADS_PER_BLOCK);
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
	}

	// allocate host memory
	std::vector<Node> h_graph_nodes(no_of_nodes);
	std::vector<bool_t> h_graph_mask(no_of_nodes);
	std::vector<bool_t> h_updating_graph_mask(no_of_nodes);
	std::vector<bool_t> h_graph_visited(no_of_nodes);

	int start, edgeno;
	// initalize the memory
	for (unsigned int i = 0; i < no_of_nodes; i++) {
		fscanf(fp, "%d %d", &start, &edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i] = false;
		h_updating_graph_mask[i] = false;
		h_graph_visited[i] = false;
	}

	//read the source node from the file
	fscanf(fp, "%d", &source);
	source = 0;

	//set the source node as true in the mask
	h_graph_mask[source] = true;
	h_graph_visited[source] = true;

	fscanf(fp, "%d", &edge_list_size);

	int id, cost;
	std::vector<int> h_graph_edges(edge_list_size);
	for (int i = 0; i < edge_list_size; i++) {
		fscanf(fp, "%d", &id);
		fscanf(fp, "%d", &cost);
		h_graph_edges[i] = id;
	}

	if (fp)
		fclose(fp);

	printf("Read File\n");

	//Copy the Node list to device memory
	rad::DeviceVector<Node> d_graph_nodes = h_graph_nodes;
//	cudaMalloc((void**) &d_graph_nodes, sizeof(Node) * no_of_nodes);
//	cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes,
//			cudaMemcpyHostToDevice);

	//Copy the Edge List to device Memory
	rad::DeviceVector<int> d_graph_edges = h_graph_edges;
//	cudaMalloc((void**) &d_graph_edges, sizeof(int) * edge_list_size);
//	cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size,
//			cudaMemcpyHostToDevice);

	//Copy the Mask to device memory
	rad::DeviceVector<bool_t> d_graph_mask = h_graph_mask;
//	cudaMalloc((void**) &d_graph_mask, sizeof(bool_t) * no_of_nodes);
//	cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool_t) * no_of_nodes,
//			cudaMemcpyHostToDevice);

	rad::DeviceVector<bool_t> d_updating_graph_mask = h_updating_graph_mask;
//	cudaMalloc((void**) &d_updating_graph_mask, sizeof(bool_t) * no_of_nodes);
//	cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask,
//			sizeof(bool_t) * no_of_nodes, cudaMemcpyHostToDevice);

	//Copy the Visited nodes array to device memory
	rad::DeviceVector<bool_t> d_graph_visited = h_graph_visited;
//	cudaMalloc((void**) &d_graph_visited, sizeof(bool_t) * no_of_nodes);
//	cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(bool_t) * no_of_nodes,
//			cudaMemcpyHostToDevice);

	// allocate mem for the result on host side
	std::vector<int> h_cost(no_of_nodes);
	for (int i = 0; i < no_of_nodes; i++)
		h_cost[i] = -1;
	h_cost[source] = 0;

	// allocate device memory for result
	rad::DeviceVector<int> d_cost = h_cost;
//	cudaMalloc((void**) &d_cost, sizeof(int) * no_of_nodes);
//	cudaMemcpy(d_cost, h_cost, sizeof(int) * no_of_nodes,
//			cudaMemcpyHostToDevice);

	//make a bool_t to check if the execution is over
	bool_t *d_over;
	cudaMalloc((void**) &d_over, sizeof(bool_t));

	printf("Copied Everything to GPU memory\n");

	// setup execution parameters
	dim3 grid(num_of_blocks, 1, 1);
	dim3 threads(num_of_threads_per_block, 1, 1);

	int k = 0;
	printf("Start traversing the tree\n");
	bool_t stop;
	//Call the Kernel untill all the elements of Frontier are not false
	do {
		//if no thread changes this value then the loop stops
		stop = false;
		cudaMemcpy(d_over, &stop, sizeof(bool_t), cudaMemcpyHostToDevice);
		Kernel<<<grid, threads, 0>>>(d_graph_nodes.data(), d_graph_edges.data(), d_graph_mask.data(),
				d_updating_graph_mask.data(), d_graph_visited.data(), d_cost.data(), no_of_nodes);
		// check if kernel execution generated and error

		Kernel2<<<grid, threads, 0>>>(d_graph_mask.data(), d_updating_graph_mask.data(),
				d_graph_visited.data(), d_over, no_of_nodes);
		// check if kernel execution generated and error

		cudaMemcpy(&stop, d_over, sizeof(bool_t), cudaMemcpyDeviceToHost);
		k++;
	} while (stop);

	printf("Kernel Executed %d times\n", k);

	// copy result from device to host
	cudaMemcpy(h_cost.data(), d_cost.data(), sizeof(int) * no_of_nodes,
			cudaMemcpyDeviceToHost);

	//Store the result into a file
	FILE *fpo = fopen("result.txt", "w");
	for (int i = 0; i < no_of_nodes; i++)
		fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");

	// cleanup memory
//	free(h_graph_nodes);
//	free(h_graph_edges);
//	free(h_graph_mask);
//	free(h_updating_graph_mask);
//	free(h_graph_visited);
//	free(h_cost);
//	cudaFree(d_graph_nodes);
//	cudaFree(d_graph_edges);
//	cudaFree(d_graph_mask);
//	cudaFree(d_updating_graph_mask);
//	cudaFree(d_graph_visited);
//	cudaFree(d_cost);
}
