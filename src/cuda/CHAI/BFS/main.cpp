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

#include "support/cuda-setup.h"
#include "kernel.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"

#include <unistd.h>
#include <thread>
#include <assert.h>

//*****************************************  LOG  ***********************************//
#ifdef LOGS
#include "log_helper.h"
#endif
//************************************************************************************//

// Params ---------------------------------------------------------------------
struct Params {

	int device;
	int n_gpu_threads;
	int n_gpu_blocks;
	int n_threads;
	int n_warmup;
	int n_reps;
	const char *file_name;
	const char *comparison_file;
	int switching_limit;
	std::string gold_path;
	//for radiation test
	int mode;

	Params(int argc, char **argv) {
		device = 0;
		n_gpu_threads = 256;
		n_gpu_blocks = 8;
		n_threads = 2;
		n_warmup = 1;
		n_reps = 1;
		file_name = "input/NYR_input.dat";
		comparison_file = "output/NYR_bfs_BFS.out";
		switching_limit = 128;
		mode = -1;
		gold_path = "";

		int opt;
		while ((opt = getopt(argc, argv, "hd:i:g:t:w:r:f:c:l:m:p:")) >= 0) {
			switch (opt) {
			case 'h':
				usage();
				exit(0);
				break;
			case 'd':
				device = atoi(optarg);
				break;
			case 'i':
				n_gpu_threads = atoi(optarg);
				break;
			case 'g':
				n_gpu_blocks = atoi(optarg);
				break;
			case 't':
				n_threads = atoi(optarg);
				break;
			case 'w':
				n_warmup = atoi(optarg);
				break;
			case 'r':
				n_reps = atoi(optarg);
				break;
			case 'f':
				file_name = optarg;
				break;
			case 'c':
				comparison_file = optarg;
				break;
			case 'l':
				switching_limit = atoi(optarg);
				break;
			case 'm':
				mode = atoi(optarg);
				break;
			case 'p':
				gold_path = std::string(optarg);
				break;
			default:
				fprintf(stderr, "\nUnrecognized option!\n");
				usage();
				exit(0);
			}
		}
		assert(n_gpu_threads > 0 && "Invalid # of device threads!");
		assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
		assert(
				(mode == -1 || mode == 0 || mode == 1)
						&& "Invalid mode selection");
		if (mode == 0) { //generate case
			n_reps = 1;
		}
//        assert(n_threads > 0 && "Invalid # of host threads!");
	}

	void usage() {
		fprintf(stderr,
				"\nUsage:  ./bfs [options]"
						"\n"
						"\nGeneral options:"
						"\n    -h        help"
						"\n    -d <D>    CUDA device ID (default=0)"
						"\n    -i <I>    # of device threads per block (default=256)"
						"\n    -g <G>    # of device blocks (default=8)"
						"\n    -t <T>    # of host threads (default=2)"
						"\n    -w <W>    # of untimed warmup iterations (default=1)"
						"\n    -r <R>    # of timed repetition iterations (default=1)"
						"\n"
						"\nBenchmark-specific options:"
						"\n    -f <F>    name of input file with control points (default=input/NYR_input.dat)"
						"\n    -c <C>    comparison file (default=output/NYR_bfs_BFS.out)"
						"\n    -l <L>    switching limit (default=128)"
						"\n    -m <M>    mode -1 default value, 1 for radiation test "
						"normal execution, 0 generate gold, >=1 (number of iterations) radiation test"
						"\n	   -p <P>    it must contains gold ouptut path"
						"\n");
	}
};

// Input Data -----------------------------------------------------------------
void read_input_size(int &n_nodes, int &n_edges, const Params &p) {
	FILE *fp = fopen(p.file_name, "r");
	fscanf(fp, "%d", &n_nodes);
	fscanf(fp, "%d", &n_edges);
	if (fp)
		fclose(fp);
}

void read_gold_size(int &n_nodes_o, const Params &p) {
	FILE *fp = fopen(p.comparison_file, "r");
	fscanf(fp, "%d", &n_nodes_o);
	if (fp)
		fclose(fp);

}

Gold* read_gold(int *num_of_nodes, int *num_of_edges, std::string gold_path) {
	//std::cout << "Passou aqui\n\n\n";
	FILE *fpo = fopen(gold_path.c_str(), "r");
	if (!fpo) {
		printf("Error Reading output file\n");
		exit (EXIT_FAILURE);
	}

//	int num_of_nodes, num_of_edges;
	fscanf(fpo, "%d %d", num_of_nodes, num_of_edges);
//	if (h_nodes != nullptr) {
//		free(h_nodes);
//	}

	Gold *h_nodes = (Gold *) malloc(sizeof(Gold) * *num_of_nodes);
	if (h_nodes == NULL)
		exit (EXIT_FAILURE);

	for (int i = 0; i < *num_of_nodes; i++) {
		int j, cost;
		fscanf(fpo, "%d %d", &j, &cost);
		h_nodes[i].j = j;
		h_nodes[i].cost = cost;

	}

	fclose(fpo);
//	std::cout << "INSIDE READ GOLD " << h_nodes << "\n";
//	return std::pair<int, int>(num_of_nodes, num_of_edges);
	return h_nodes;
}

void read_input(int &source, Node *&h_nodes, Edge *&h_edges, const Params &p) {

	int start, edgeno;
	int n_nodes, n_edges;
	int id, cost;
	FILE *fp = fopen(p.file_name, "r");

	fscanf(fp, "%d", &n_nodes);
	fscanf(fp, "%d", &n_edges);
	fscanf(fp, "%d", &source);
	//printf("Number of nodes = %d\t", n_nodes);
	// printf("Number of edges = %d\t", n_edges);

	// initalize the memory: Nodes
	for (int i = 0; i < n_nodes; i++) {
		fscanf(fp, "%d %d", &start, &edgeno);
		h_nodes[i].x = start;
		h_nodes[i].y = edgeno;
	}
#if PRINT_ALL
	for(int i = 0; i < n_nodes; i++) {
		printf("%d, %d\n", h_nodes[i].x, h_nodes[i].y);
	}
#endif

	// initalize the memory: Edges
	for (int i = 0; i < n_edges; i++) {
		fscanf(fp, "%d", &id);
		fscanf(fp, "%d", &cost);
		h_edges[i].x = id;
		h_edges[i].y = -cost;
	}
	if (fp)
		fclose(fp);
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

	const Params p(argc, argv);
	CUDASetup setcuda(p.device);
	Timer timer;
	cudaError_t cudaStatus;

	int it_cpu = 0;
	int it_gpu = 0;
	int err = 0;

#ifdef LOGS
	set_iter_interval_print(10);
	char test_info[500];
	snprintf(test_info, 500, "-i %d -g %d -t %d -f %s -l %d\n",p.n_gpu_threads, p.n_gpu_blocks,p.n_threads, p.file_name,p.switching_limit);
	start_log_file("cudaBreadthFirstSearch", test_info);
	//printf("Com LOG\n");
#endif

	// Allocate
	int n_nodes, n_edges;
//	int n_nodes_o;
//************************* Allocando Memoria para o Gold **********************
	Gold *gold = nullptr;
	if (p.mode == 0) {
//		read_gold_size(n_nodes_o, p);
		read_input_size(n_nodes, n_edges, p);
		gold = (Gold *) malloc(sizeof(Gold) * n_nodes);

	} else {
//		std::pair<int, int> sizes =
		gold = read_gold(&n_nodes, &n_edges, p.gold_path);
		update_timestamp();

//		std::cout << gold << " " << p.mode << "\n";
//
//		n_nodes = sizes.first;
//		n_edges = sizes.second;
	}
//******************************************************************************

	timer.start("Allocation");
	Node * h_nodes = (Node *) malloc(sizeof(Node) * n_nodes);
	Node * d_nodes;
	cudaStatus = cudaMalloc((void**) &d_nodes, sizeof(Node) * n_nodes);
	Edge * h_edges = (Edge *) malloc(sizeof(Edge) * n_edges);
	Edge * d_edges;
	cudaStatus = cudaMalloc((void**) &d_edges, sizeof(Edge) * n_edges);
	std::atomic_int *h_color = (std::atomic_int *) malloc(
			sizeof(std::atomic_int) * n_nodes);
	int * d_color;
	cudaStatus = cudaMalloc((void**) &d_color, sizeof(int) * n_nodes);
	std::atomic_int *h_cost = (std::atomic_int *) malloc(
			sizeof(std::atomic_int) * n_nodes);
	int * d_cost;
	cudaStatus = cudaMalloc((void**) &d_cost, sizeof(int) * n_nodes);
	int * h_q1 = (int *) malloc(n_nodes * sizeof(int));
	int * d_q1;
	cudaStatus = cudaMalloc((void**) &d_q1, sizeof(int) * n_nodes);
	int * h_q2 = (int *) malloc(n_nodes * sizeof(int));
	int * d_q2;
	cudaStatus = cudaMalloc((void**) &d_q2, sizeof(int) * n_nodes);
	std::atomic_int h_head[1];
	int * d_head;
	cudaStatus = cudaMalloc((void**) &d_head, sizeof(int));
	std::atomic_int h_tail[1];
	update_timestamp();
	int * d_tail;
	cudaStatus = cudaMalloc((void**) &d_tail, sizeof(int));
	std::atomic_int h_threads_end[1];
	int * d_threads_end;
	cudaStatus = cudaMalloc((void**) &d_threads_end, sizeof(int));
	std::atomic_int h_threads_run[1];
	int * d_threads_run;
	cudaStatus = cudaMalloc((void**) &d_threads_run, sizeof(int));
	int h_num_t[1];
	int * d_num_t;
	cudaStatus = cudaMalloc((void**) &d_num_t, sizeof(int));
	int h_overflow[1];
	int * d_overflow;
	cudaStatus = cudaMalloc((void**) &d_overflow, sizeof(int));
	std::atomic_int h_iter[1];
	int * d_iter;
	cudaStatus = cudaMalloc((void**) &d_iter, sizeof(int));
	ALLOC_ERR(h_nodes, h_edges, h_color, h_cost, h_q1, h_q2);
	CUDA_ERR();
	cudaDeviceSynchronize();
	timer.stop("Allocation");

	// Initialize
	timer.start("Initialization");
	const int max_gpu_threads = setcuda.max_gpu_threads();
	int source;
	read_input(source, h_nodes, h_edges, p);
	update_timestamp();
//// **********************  Lendo O gold *********************************
//	read_gold(gold, p);
//// **********************************************************************
	for (int i = 0; i < n_nodes; i++) {
		h_cost[i].store(INF);
	}
	h_cost[source].store(0);
	for (int i = 0; i < n_nodes; i++) {
		h_color[i].store(WHITE);
	}
	h_tail[0].store(0);
	h_head[0].store(0);
	h_threads_end[0].store(0);
	h_threads_run[0].store(0);
	h_q1[0] = source;
	h_iter[0].store(0);
	h_overflow[0] = 0;
	cudaDeviceSynchronize();
	timer.stop("Initialization");
	// timer.print("Initialization", 1);

	// Copy to device
	timer.start("Copy To Device");
	cudaStatus = cudaMemcpy(d_nodes, h_nodes, sizeof(Node) * n_nodes,
			cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_edges, h_edges, sizeof(Edge) * n_edges,
			cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	CUDA_ERR();
	timer.stop("Copy To Device");

	for (int rep = 0; rep < p.n_reps; rep++) {
		//printf("Rep %d \n",rep);
		// Reset
		update_timestamp();
		for (int i = 0; i < n_nodes; i++) {
			h_cost[i].store(INF);
		}
		h_cost[source].store(0);
		for (int i = 0; i < n_nodes; i++) {
			h_color[i].store(WHITE);
		}

		it_cpu = 0;
		it_gpu = 0;

		h_tail[0].store(0);
		h_head[0].store(0);
		h_threads_end[0].store(0);
		h_threads_run[0].store(0);
		h_q1[0] = source;
		h_iter[0].store(0);
		h_overflow[0] = 0;

		// if(rep >= p.n_warmup)
		timer.start("Kernel");
#ifdef LOGS
		start_iteration();
#endif
		// Run first iteration in master CPU thread
		h_num_t[0] = 1;
		int pid;
		int index_i, index_o;
		for (index_i = 0; index_i < h_num_t[0]; index_i++) {
			pid = h_q1[index_i];
			h_color[pid].store(BLACK);
			for (int i = h_nodes[pid].x; i < (h_nodes[pid].y + h_nodes[pid].x);
					i++) {
				int id = h_edges[i].x;
				h_color[id].store(BLACK);
				index_o = h_tail[0].fetch_add(1);
				h_q2[index_o] = id;
			}
		}
		h_num_t[0] = h_tail[0].load();
		h_tail[0].store(0);
		h_threads_run[0].fetch_add(1);
		h_iter[0].fetch_add(1);
		//if(rep >= p.n_warmup)
		timer.stop("Kernel");

		// Pointers to input and output queues
		int * h_qin = h_q2;
		int * h_qout = h_q1;
		int * d_qin = d_q2;
		int * d_qout = d_q1;

		const int CPU_EXEC = (p.n_threads > 0) ? 1 : 0;
		const int GPU_EXEC =
				(p.n_gpu_blocks > 0 && p.n_gpu_threads > 0) ? 1 : 0;

		// Run subsequent iterations on CPU or GPU until number of input queue elements is 0
		while (*h_num_t != 0) {

			if ((*h_num_t < p.switching_limit || GPU_EXEC == 0)
					&& CPU_EXEC == 1) { // If the number of input queue elements is lower than switching_limit
				update_timestamp();
				it_cpu = it_cpu + 1;
				//if(rep >= p.n_warmup)
				timer.start("Kernel");

				// Continue until switching_limit condition is not satisfied
				while ((*h_num_t != 0)
						&& (*h_num_t < p.switching_limit || GPU_EXEC == 0)
						&& CPU_EXEC == 1) {

					// Swap queues
					if (h_iter[0] % 2 == 0) {
						h_qin = h_q1;
						h_qout = h_q2;
					} else {
						h_qin = h_q2;
						h_qout = h_q1;
					}

					std::thread main_thread(run_cpu_threads, h_nodes, h_edges,
							h_cost, h_color, h_qin, h_qout, h_num_t, h_head,
							h_tail, h_threads_end, h_threads_run, h_iter,
							p.n_threads, p.switching_limit, GPU_EXEC);
					main_thread.join();

					h_num_t[0] = h_tail[0].load(); // Number of elements in output queue
					h_tail[0].store(0);
					h_head[0].store(0);
				}

				//if(rep >= p.n_warmup)
				timer.stop("Kernel");

			} else if ((*h_num_t >= p.switching_limit || CPU_EXEC == 0)
					&& GPU_EXEC == 1) { // If the number of input queue elements is higher than or equal to switching_limit
				update_timestamp();
				it_gpu = it_gpu + 1;
				//if(rep >= p.n_warmup)
				timer.start("Copy To Device");
				cudaStatus = cudaMemcpy(d_cost, h_cost, sizeof(int) * n_nodes,
						cudaMemcpyHostToDevice);
				cudaStatus = cudaMemcpy(d_color, h_color, sizeof(int) * n_nodes,
						cudaMemcpyHostToDevice);
				cudaStatus = cudaMemcpy(d_threads_run, h_threads_run,
						sizeof(int), cudaMemcpyHostToDevice);
				cudaStatus = cudaMemcpy(d_threads_end, h_threads_end,
						sizeof(int), cudaMemcpyHostToDevice);
				cudaStatus = cudaMemcpy(d_overflow, h_overflow, sizeof(int),
						cudaMemcpyHostToDevice);
				cudaStatus = cudaMemcpy(d_q1, h_q1, sizeof(int) * n_nodes,
						cudaMemcpyHostToDevice);
				cudaStatus = cudaMemcpy(d_q2, h_q2, sizeof(int) * n_nodes,
						cudaMemcpyHostToDevice);
				cudaStatus = cudaMemcpy(d_iter, h_iter, sizeof(int),
						cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				CUDA_ERR();
				//if(rep >= p.n_warmup)
				timer.stop("Copy To Device");

				// Continue until switching_limit condition is not satisfied
				while ((*h_num_t != 0)
						&& (*h_num_t >= p.switching_limit || CPU_EXEC == 0)
						&& GPU_EXEC == 1) {

					// Swap queues
					if (h_iter[0] % 2 == 0) {
						d_qin = d_q1;
						d_qout = d_q2;
					} else {
						d_qin = d_q2;
						d_qout = d_q1;
					}

					//if(rep >= p.n_warmup)
					timer.start("Copy To Device");
					cudaStatus = cudaMemcpy(d_num_t, h_num_t, sizeof(int),
							cudaMemcpyHostToDevice);
					cudaStatus = cudaMemcpy(d_tail, h_tail, sizeof(int),
							cudaMemcpyHostToDevice);
					cudaStatus = cudaMemcpy(d_head, h_head, sizeof(int),
							cudaMemcpyHostToDevice);
					cudaDeviceSynchronize();
					CUDA_ERR();
					//if(rep >= p.n_warmup)
					timer.stop("Copy To Device");

					if (rep >= p.n_warmup)
						timer.start("Kernel");
					assert(
							p.n_gpu_threads <= max_gpu_threads
									&& "The thread block size is greater than the maximum thread block size that can be used on this device");
					cudaStatus = call_BFS_gpu(p.n_gpu_blocks, p.n_gpu_threads,
							d_nodes, d_edges, d_cost, d_color, d_qin, d_qout,
							d_num_t, d_head, d_tail, d_threads_end,
							d_threads_run, d_overflow, d_iter,
							p.switching_limit, CPU_EXEC,
							sizeof(int) * (W_QUEUE_SIZE + 3));
					cudaDeviceSynchronize();
					CUDA_ERR();
					// if(rep >= p.n_warmup)
					timer.stop("Kernel");

					//if(rep >= p.n_warmup)
					timer.start("Copy Back and Merge");
					cudaStatus = cudaMemcpy(h_tail, d_tail, sizeof(int),
							cudaMemcpyDeviceToHost);
					cudaStatus = cudaMemcpy(h_iter, d_iter, sizeof(int),
							cudaMemcpyDeviceToHost);
					cudaDeviceSynchronize();
					CUDA_ERR();
					//if(rep >= p.n_warmup)
					timer.stop("Copy Back and Merge");

					h_num_t[0] = h_tail[0].load(); // Number of elements in output queue
					h_tail[0].store(0);
					h_head[0].store(0);
				}

				//if(rep >= p.n_warmup)
				timer.start("Copy Back and Merge");
				cudaStatus = cudaMemcpy(h_cost, d_cost, sizeof(int) * n_nodes,
						cudaMemcpyDeviceToHost);
				cudaStatus = cudaMemcpy(h_color, d_color, sizeof(int) * n_nodes,
						cudaMemcpyDeviceToHost);
				cudaStatus = cudaMemcpy(h_threads_run, d_threads_run,
						sizeof(int), cudaMemcpyDeviceToHost);
				cudaStatus = cudaMemcpy(h_threads_end, d_threads_end,
						sizeof(int), cudaMemcpyDeviceToHost);
				cudaStatus = cudaMemcpy(h_overflow, d_overflow, sizeof(int),
						cudaMemcpyDeviceToHost);
				cudaStatus = cudaMemcpy(h_q1, d_q1, sizeof(int) * n_nodes,
						cudaMemcpyDeviceToHost);
				cudaStatus = cudaMemcpy(h_q2, d_q2, sizeof(int) * n_nodes,
						cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				CUDA_ERR();
				// if(rep >= p.n_warmup)
				timer.stop("Copy Back and Merge");
			}

		}
#ifdef LOGS
		end_iteration();
#endif

		//printf("IT CPU:%d\t",it_cpu);
		//printf("IT GPU:%d\n",it_gpu);

//	err=new_verify(h_cost, n_nodes, p.comparison_file,it_cpu,it_gpu);
		if (p.mode == 1){
			update_timestamp();
			err = newest_verify(h_cost, n_nodes, n_nodes, gold, it_cpu, it_gpu);

		if (err > 0) {
			printf("Errors: %d\n", err);
			update_timestamp();
			read_input(source, h_nodes, h_edges, p);
			if(gold){
				free(gold);
			}
			update_timestamp();
			gold = read_gold(&n_nodes, &n_edges, p.gold_path);

		} else {
			printf(".");
			fflush(stdout);
		}
    }
#ifdef LOGS
		log_error_count(err);
#endif
		// Ler a entrada novamente
		//read_input(source, h_nodes, h_edges, p);
		//read_gold(gold,p);

	} // end of iteration

#ifdef LOGS
	end_log_file();
#endif

	// Generate case
	if (p.mode == 0) {
		//TODO: Put generate function here
		create_output(h_cost, n_nodes, n_edges, p.gold_path);
	}
	//timer.print("Allocation", 1);
	//timer.print("Copy To Device", p.n_reps);
	//timer.print("Kernel", p.n_reps);
	//timer.print("Copy Back and Merge", p.n_reps);

	// Verify answer
	// verify(h_cost, n_nodes, p.comparison_file);

	// Free memory
	timer.start("Deallocation");
	free(h_nodes);
	free(h_edges);
	free(h_color);
	free(h_cost);
	free(h_q1);
	free(h_q2);
	free(gold);
	cudaStatus = cudaFree(d_nodes);
	cudaStatus = cudaFree(d_edges);
	cudaStatus = cudaFree(d_cost);
	cudaStatus = cudaFree(d_color);
	cudaStatus = cudaFree(d_q1);
	cudaStatus = cudaFree(d_q2);
	cudaStatus = cudaFree(d_num_t);
	cudaStatus = cudaFree(d_head);
	cudaStatus = cudaFree(d_tail);
	cudaStatus = cudaFree(d_threads_end);
	cudaStatus = cudaFree(d_threads_run);
	cudaStatus = cudaFree(d_overflow);
	cudaStatus = cudaFree(d_iter);

	CUDA_ERR();
	cudaDeviceSynchronize();
	timer.stop("Deallocation");
	//timer.print("Deallocation", 1);

	// Release timers
	timer.release("Allocation");
	timer.release("Initialization");
	timer.release("Copy To Device");
	timer.release("Kernel");
	timer.release("Copy Back and Merge");
	timer.release("Deallocation");

	printf("Test Passed\n");
	return 0;
}
