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

#include "common.h"
#include <math.h>

inline long verify(std::atomic_long *h_cost, long num_of_nodes, const char *file_name) {
    // Compare to output file
    FILE *fpo = fopen(file_name, "r");
    if(!fpo) {
        printf("Error Reading output file\n");
        exit(EXIT_FAILURE);
    }
#if PRINT
    printf("Reading Output: %s\n", file_name);
#endif

    // the number of nodes in the output
    long num_of_nodes_o = 0;
    fscanf(fpo, "%ld", &num_of_nodes_o);
    if(num_of_nodes != num_of_nodes_o) {
        printf("Number of nodes does not match the expected value\n");
        exit(EXIT_FAILURE);
    }

    // cost of nodes in the output
    for(long i = 0; i < num_of_nodes_o; i++) {
        long j, cost;
        fscanf(fpo, "%ld %ld", &j, &cost);
        if(i != j || h_cost[i].load() != cost) {
            printf("Computed node %ld cost (%ld != %ld) does not match the expected value\n", i, h_cost[i].load(), cost);
            exit(EXIT_FAILURE);
        }
    }

    fclose(fpo);
    return 0;
}

inline long create_output(std::atomic_long *h_cost, long num_of_nodes) {
    // Compare to output file
    FILE *fpo = fopen("saida_grafo", "w");
    if(!fpo) {
        printf("Error Creating output file\n");
        exit(EXIT_FAILURE);
    }
	fprintf(fpo,"%ld\n",num_of_nodes);	
    // cost of nodes in the output
    for(long i = 0; i < num_of_nodes; i++) {
		// escreve i no arquivo e hcost 
	fprintf(fpo,"%ld %ld\n",i,h_cost[i].load());
   }
 
//printf("sai func \n");
    fclose(fpo);
    return 0;
}
