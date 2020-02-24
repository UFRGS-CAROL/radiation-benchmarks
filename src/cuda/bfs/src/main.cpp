#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <tuple>

#include "common.h"
#include "device_vector.h"

void BFSGraph(rad::DeviceVector<Node>& d_graph_nodes,
		rad::DeviceVector<bool_t>& d_graph_mask,
		rad::DeviceVector<bool_t>& d_updating_graph_mask,
		rad::DeviceVector<bool_t>& d_graph_visited,
		rad::DeviceVector<int>& d_graph_edges, rad::DeviceVector<int>& d_cost,
		int no_of_nodes);

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
		std::cout << "Error Reading graph file " << input_f << std::endl;
	}

	return {no_of_nodes, source};
}

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	if (argc != 2) {
		Usage(argc, argv);
		exit(0);
	}

	// allocate host memory
	std::vector<Node> h_graph_nodes;
	std::vector<bool_t> h_graph_mask;
	std::vector<bool_t> h_updating_graph_mask;
	std::vector<bool_t> h_graph_visited;
	std::vector<int> h_graph_edges;

	std::string input_f = argv[1];
	std::string output_f = "result.txt";
	int no_of_nodes, source;
	std::tie(no_of_nodes, source) = read_input_file(h_graph_nodes, h_graph_mask,
			h_updating_graph_mask, h_graph_visited, h_graph_edges, input_f);

	// allocate mem for the result on host side
	std::vector<int> h_cost(no_of_nodes, -1);
	h_cost[source] = 0;

	//Copy the Node list to device memory
	rad::DeviceVector<Node> d_graph_nodes = h_graph_nodes;

	//Copy the Edge List to device Memory
	rad::DeviceVector<int> d_graph_edges = h_graph_edges;

	//Copy the Mask to device memory
	rad::DeviceVector<bool_t> d_graph_mask = h_graph_mask;
	rad::DeviceVector<bool_t> d_updating_graph_mask = h_updating_graph_mask;

	//Copy the Visited nodes array to device memory
	rad::DeviceVector<bool_t> d_graph_visited = h_graph_visited;

	// allocate device memory for result
	rad::DeviceVector<int> d_cost = h_cost;

	//saving arrays to re-set memory
	const auto d_save_graph_mask = d_graph_mask;
	const auto d_save_updating_graph_mask = d_updating_graph_mask;
	const auto d_save_graph_visited = d_graph_visited;
	const auto d_save_cost = d_cost;

	for (size_t iteration = 0; iteration < 1000; iteration++) {
		auto set_time = rad::mysecond();
		d_cost = d_save_cost;
		d_graph_mask = d_save_graph_mask;
		d_updating_graph_mask = d_save_updating_graph_mask;
		d_graph_visited = d_save_graph_visited;
		set_time = rad::mysecond() - set_time;

		auto kernel_time = rad::mysecond();
		BFSGraph(d_graph_nodes, d_graph_mask, d_updating_graph_mask,
				d_graph_visited, d_graph_edges, d_cost, no_of_nodes);

		kernel_time = rad::mysecond() - kernel_time;

		auto copy_time = rad::mysecond();
		d_cost.to_vector(h_cost);
		copy_time = rad::mysecond() - copy_time;

		std::cout << "Set time " << set_time << std::endl;
		std::cout << "Kernel time " << kernel_time << std::endl;
		std::cout << "Copy time " << copy_time << std::endl;

	}

	// copy result from device to host

	//Store the result into a file
	std::ofstream fo(output_f);
	for (int i = 0; i < no_of_nodes; i++)
		fo << i << ") cost:" << h_cost[i] << std::endl;
	fo.close();
	std::cout << "Result stored in " << output_f << std::endl;
}
