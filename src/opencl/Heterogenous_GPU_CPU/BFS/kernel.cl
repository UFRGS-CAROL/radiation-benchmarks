/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#define _OPENCL_COMPILER_

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#include "support/common.h"

// OpenCL kernel ------------------------------------------------------------------------------------------
__kernel void BFS_gpu(__global Node *graph_nodes_av, __global Edge *graph_edges_av, __global long *cost,
    __global long *color, __global long *q1, __global long *q2, __global long *n_t,
    __global long *head, __global long *tail, __global long *threads_end, __global long *threads_run,
    __global long *overflow, __global long *iter, __local long *tail_bin, __local long *l_q2, __local long *shift,
    __local long *base, long LIMIT, const long CPU) {

    const long tid     = get_local_id(0);
    const long gtid    = get_global_id(0);
    const long MAXWG   = get_num_groups(0);
    const long WG_SIZE = get_local_size(0);

    long iter_local = atom_add(&iter[0], 0);

    long n_t_local = atom_add(n_t, 0);

    if(tid == 0) {
        // Reset queue
        *tail_bin = 0;
    }

    // Fetch frontier elements from the queue
    if(tid == 0)
        *base = atom_add(&head[0], WG_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);

    long my_base = *base;
    while(my_base < n_t_local) {
        if(my_base + tid < n_t_local && *overflow == 0) {
            // Visit a node from the current frontier
            long pid = q1[my_base + tid];
            //////////////// Visit node ///////////////////////////
            atom_xchg(&cost[pid], iter_local); // Node visited
            Node cur_node;
            cur_node.x = graph_nodes_av[pid].x;
            cur_node.y = graph_nodes_av[pid].y;
            // For each outgoing edge
            for(long i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
                long id        = graph_edges_av[i].x;
                long old_color = atom_max(&color[id], BLACK);
                if(old_color < BLACK) {
                    // Push to the queue
                    long tail_index = atom_add(tail_bin, 1);
                    if(tail_index >= W_QUEUE_SIZE) {
                        *overflow = 1;
                    } else
                        l_q2[tail_index] = id;
                }
            }
        }
        if(tid == 0)
            *base = atom_add(&head[0], WG_SIZE); // Fetch more frontier elements from the queue
        barrier(CLK_LOCAL_MEM_FENCE);
        my_base = *base;
    }
    /////////////////////////////////////////////////////////
    // Compute size of the output and allocate space in the global queue
    if(tid == 0) {
        *shift = atom_add(&tail[0], *tail_bin);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ///////////////////// CONCATENATE INTO GLOBAL MEMORY /////////////////////
    long local_shift = tid;
    while(local_shift < *tail_bin) {
        q2[*shift + local_shift] = l_q2[local_shift];
        // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
        local_shift += WG_SIZE;
    }
    //////////////////////////////////////////////////////////////////////////

    if(gtid == 0) {
        atom_add(&iter[0], 1);
    }
}
