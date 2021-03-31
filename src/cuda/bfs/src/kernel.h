/*
 * kernel.h
 *
 *  Created on: 23/02/2020
 *      Author: fernando
 */

#ifndef KERNEL_H_
#define KERNEL_H_

#include "common.h"
__global__ void Kernel(Node* g_graph_nodes, int* g_graph_edges,
		bool_t* g_graph_mask, bool_t* g_updating_graph_mask, bool_t *g_graph_visited,
		int* g_cost, int no_of_nodes);

__global__ void Kernel2(bool_t* g_graph_mask, bool_t *g_updating_graph_mask,
		bool_t* g_graph_visited, bool_t *g_over, int no_of_nodes);

__global__ void FullKernel(Node* g_graph_nodes, int* g_graph_edges,
		bool_t* g_graph_mask, bool_t* g_updating_graph_mask,
		bool_t *g_graph_visited, int* g_cost, int no_of_nodes, bool_t *g_over);

#endif /* KERNEL_H_ */
