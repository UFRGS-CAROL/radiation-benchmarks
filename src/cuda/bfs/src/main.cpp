#include <string>
#include <fstream>
#include <iostream>
#include <tuple>
#include <algorithm>
#include <omp.h>
#include "common.h"
#include "Parameters.h"

int BFSGraph(rad::DeviceVector<Node>& d_graph_nodes,
		rad::DeviceVector<bool_t>& d_graph_mask,
		rad::DeviceVector<bool_t>& d_updating_graph_mask,
		rad::DeviceVector<bool_t>& d_graph_visited,
		rad::DeviceVector<int>& d_graph_edges, rad::DeviceVector<int>& d_cost,
		cudaStream_t& stream, int no_of_nodes);

void Usage(int argc, char**argv) {

	fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);

}

std::tuple<int, int> read_input_file(std::vector<Node>& h_graph_nodes,
		std::vector<bool_t>& h_graph_mask,
		std::vector<bool_t>& h_updating_graph_mask,
		std::vector<bool_t>& h_graph_visited, std::vector<int>& h_graph_edges,
		std::string& input_f) {

	std::cout << "Reading File " << input_f << std::endl;
	//Read in Graph from a file
	std::ifstream fp(input_f);

	int source = 0;
	int no_of_nodes = 0;
	int edge_list_size = 0;
	if (fp.good()) {

		fp >> no_of_nodes;

		// allocate host memory
		h_graph_nodes.resize(no_of_nodes);
		h_graph_mask.resize(no_of_nodes);
		h_updating_graph_mask.resize(no_of_nodes);
		h_graph_visited.resize(no_of_nodes);

		// initalize the memory
		for (auto i = 0; i < no_of_nodes; i++) {
			int start, edgeno;
			fp >> start >> edgeno;
			h_graph_nodes[i].starting = start;
			h_graph_nodes[i].no_of_edges = edgeno;
			h_graph_mask[i] = FALSE;
			h_updating_graph_mask[i] = FALSE;
			h_graph_visited[i] = FALSE;
		}

		//read the source node from the file
		fp >> source;
		source = 0;

		//set the source node as TRUE in the mask
		h_graph_mask[source] = TRUE;
		h_graph_visited[source] = TRUE;

		fp >> edge_list_size;
		h_graph_edges.resize(edge_list_size);
		for (int i = 0; i < edge_list_size; i++) {
			int id, cost;

			fp >> id >> cost;
			h_graph_edges[i] = id;
		}

		fp.close();

		std::cout << ("Read File\n");
	} else {
		throw_line("Error Reading graph file " + input_f);
	}

	return {no_of_nodes, source};
}

void compare_output(std::vector<std::vector<int>>& output,
		std::vector<int>& gold) {
	auto stream_number = output.size();
	auto no_of_nodes = gold.size();
	auto num_of_threads = 1;
#pragma omp parallel
	{
		num_of_threads = omp_get_num_threads();
	}
	auto stream_slice = stream_number / num_of_threads;
	static std::vector<bool> equal_array(num_of_threads);

	std::cout << "NUM THR " << num_of_threads << " STREAM SLICE "
			<< stream_slice << std::endl;

#pragma omp parallel default(shared)
	{
		auto tid = omp_get_thread_num();
		equal_array[tid] = false;

		auto start_slice = tid * stream_slice;
		auto end_slice = start_slice + stream_slice;
		for (auto st = start_slice; st < end_slice; st++) {
			equal_array[tid] = equal_array[tid] || (output[st] != gold);
		}
	}

	auto falses = std::count(equal_array.begin(), equal_array.end(), false);
	if (falses != equal_array.size()) {
#pragma omp parallel for default(shared)
		for (auto stream = 0; stream < stream_number; stream++) {
			for (auto node = 0; node < no_of_nodes; node++) {
				auto g = gold[node];
				auto f = output[stream][node];
				if (g != f) {
					std::cout << "PAU\n";
				}

			}
		}
	}

}

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	Parameters parameters(argc, argv);

	if (parameters.verbose) {
		std::cout << parameters << std::endl;
	}

	// allocate host memory
	std::vector<Node> h_graph_nodes;
	std::vector<bool_t> h_graph_mask;
	std::vector<bool_t> h_updating_graph_mask;
	std::vector<bool_t> h_graph_visited;
	std::vector<int> h_graph_edges;

	int no_of_nodes, source;
	std::tie(no_of_nodes, source) = read_input_file(h_graph_nodes, h_graph_mask,
			h_updating_graph_mask, h_graph_visited, h_graph_edges,
			parameters.input);

	std::vector<int> gold_cost;
	if (!parameters.generate) {
		gold_cost.resize(no_of_nodes);
		std::ifstream gold_f(parameters.gold, std::ios::binary);
		if (gold_f.good()) {
			gold_f.read(reinterpret_cast<char*>(gold_cost.data()),
					sizeof(int) * no_of_nodes);
		} else {
			throw_line("Could not read " + parameters.gold);
		}

	}

	// allocate mem for the result on host side
	auto streams_in_parallel = parameters.sm_count * WARPS_PER_SM;
	std::vector<cudaStream_t> streams(streams_in_parallel);
	std::vector<std::vector<int>> h_cost(streams_in_parallel,
			std::vector<int>(no_of_nodes, -1));

	for (auto stream = 0; stream < streams_in_parallel; stream++) {
		h_cost[stream][source] = 0;
		rad::checkFrameworkErrors(cudaStreamCreate(&streams[stream]));
	}

	//Copy the Node list to device memory
	DevMat<Node> h_d_graph_nodes(streams_in_parallel, h_graph_nodes);

	//Copy the Edge List to device Memory
	DevMat<int> h_d_graph_edges(streams_in_parallel, h_graph_edges);

	//Copy the Mask to device memory
	DevMat<bool_t> h_d_graph_mask(streams_in_parallel);
	DevMat<bool_t> h_d_updating_graph_mask(streams_in_parallel);

	//Copy the Visited nodes array to device memory
	DevMat<bool_t> h_d_graph_visited(streams_in_parallel);

	// allocate device memory for result
	DevMat<int> h_d_cost(streams_in_parallel);

	//saving arrays to re-set memory
	const rad::DeviceVector<bool_t> d_save_graph_mask = h_graph_mask;
	const rad::DeviceVector<bool_t> d_save_updating_graph_mask =
			h_updating_graph_mask;
	const rad::DeviceVector<bool_t> d_save_graph_visited = h_graph_visited;
	const rad::DeviceVector<int> d_save_cost = h_cost[0];
	std::vector<int> k_times(streams_in_parallel);

	for (size_t iteration = 0; iteration < parameters.iterations; iteration++) {
		// Copied Everything to GPU memory

		auto set_time = rad::mysecond();

		for (int i = 0; i < streams_in_parallel; i++) {
			h_d_cost[i] = d_save_cost;
			h_d_graph_mask[i] = d_save_graph_mask;
			h_d_updating_graph_mask[i] = d_save_updating_graph_mask;
			h_d_graph_visited[i] = d_save_graph_visited;
		}

		set_time = rad::mysecond() - set_time;

		//Start traversing the tree

		auto kernel_time = rad::mysecond();
		for (int i = 0; i < streams_in_parallel; i++) {
			k_times[i] = BFSGraph(h_d_graph_nodes[i], h_d_graph_mask[i],
					h_d_updating_graph_mask[i], h_d_graph_visited[i],
					h_d_graph_edges[i], h_d_cost[i], streams[i], no_of_nodes);
		}

		for (auto& stream : streams) {
			rad::checkFrameworkErrors(cudaStreamSynchronize(stream));
		}
		kernel_time = rad::mysecond() - kernel_time;

		auto copy_time = rad::mysecond();
		for (int i = 0; i < streams_in_parallel; i++) {
			h_d_cost[i].to_vector(h_cost[i]);
		}
		copy_time = rad::mysecond() - copy_time;

		auto compare_time = rad::mysecond();
		compare_output(h_cost, gold_cost);
		compare_time = rad::mysecond() - compare_time;

		if (parameters.verbose) {
			auto wasted_time = copy_time + set_time + compare_time;
			auto overall_time = wasted_time + kernel_time;
			std::cout << "Iteration " << iteration << " - Overall time "
					<< overall_time;
			std::cout << " Set time " << set_time;
			std::cout << " Kernel time " << kernel_time;
			std::cout << " Compare time " << compare_time;
			std::cout << " Copy time " << copy_time << std::endl;
			std::cout << "Wasted time " << wasted_time << " ("
					<< int(wasted_time / overall_time * 100.0f) << "%)"
					<< std::endl;
		}
	}

	// copy result from device to host

	//Store the result into a file
	if (parameters.generate) {
		gold_cost.resize(no_of_nodes);
		std::ofstream gold_f(parameters.gold, std::ios::binary);
		if (gold_f.good()) {
			gold_f.write(reinterpret_cast<char*>(h_cost[0].data()),
					sizeof(int) * no_of_nodes);
		} else {
			throw_line("Could not write " + parameters.gold);
		}
	}

	std::cout << "Result of stream 0 stored in " << parameters.gold
			<< std::endl;

	for (auto& stream : streams) {
		rad::checkFrameworkErrors(cudaStreamDestroy(stream));
	}

}
