/*
 * kernel.h
 *
 *  Created on: 23/02/2020
 *      Author: fernando
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#include <cstdint>

#define MAX_THREADS_PER_BLOCK 1024 //512

typedef uint8_t bool_t;

enum {
	FALSE = 0,
	TRUE
};

//Structure to hold a node information
struct Node {
	int starting;
	int no_of_edges;
};

__global__ void Kernel(Node* g_graph_nodes, int* g_graph_edges,
		bool_t* g_graph_mask, bool_t* g_updating_graph_mask, bool_t *g_graph_visited,
		int* g_cost, int no_of_nodes);

__global__ void Kernel2(bool_t* g_graph_mask, bool_t *g_updating_graph_mask,
		bool_t* g_graph_visited, bool_t *g_over, int no_of_nodes);


#endif /* KERNEL_H_ */
