/*
 * kernel.h
 *
 *  Created on: 23/02/2020
 *      Author: fernando
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#define MAX_THREADS_PER_BLOCK 512

//Structure to hold a node information
struct Node {
	int starting;
	int no_of_edges;
};

__global__ void Kernel(Node* g_graph_nodes, int* g_graph_edges,
		bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited,
		int* g_cost, int no_of_nodes);

__global__ void Kernel2(bool* g_graph_mask, bool *g_updating_graph_mask,
		bool* g_graph_visited, bool *g_over, int no_of_nodes);


#endif /* KERNEL_H_ */
