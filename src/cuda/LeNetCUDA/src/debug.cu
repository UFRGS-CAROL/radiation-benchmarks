/*
 * debug.cu
 *
 *  Created on: Jun 20, 2017
 *      Author: carol
 *
 *      this file is only for debugging
 *      classes
 */
#include <iostream>
#include <vector>
#include <cstdio>



// cpu implementation of dot product
float dot(const float* v1, const float* v2, int N) {
	float s = 0;
	for (int i = 0; i != N; ++i) {
		s += v1[i] * v2[i];
	}
	return s;
}

void print_matrix(float *m, size_t h, size_t w) {
	printf("matxix\n");
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			printf("%f ", m[i * w + j]);
		}
		printf("\n");
	}

}

void cuda_gridsize(dim3 *threads, dim3 *blocks, size_t x, size_t y, size_t z) {

	long blocks_x = ceil(float(x) / float(BLOCK_SIZE));
	long threads_x = ceil(float(x) / float(blocks_x));
	long blocks_y = ceil(float(y) / float(BLOCK_SIZE));
	long threads_y = ceil(float(y) / float(blocks_y));
	long blocks_z = ceil(float(z) / float(BLOCK_SIZE));
	long threads_z = ceil(float(z) / float(blocks_z));

	*blocks = dim3(blocks_x, blocks_y, blocks_z);
	*threads = dim3(threads_x, threads_y, threads_z);

	printf("b_x %d b_y %d b_z %d\nt_x %d t_y %d t_z %d\n", blocks->x, blocks->y,
			blocks->z, threads->x, threads->y, threads->z);

}

void test_dot_product() {
	const size_t ARRAY_SIZE = 3333000;	//1024 * 1024; //1Mi elements
//	const int BLOCKS = 64;	//512;
//	const int THREADS_PER_BLOCK = BLOCK_SIZE;//256; // total threads = 512 x 256 = 128ki threads;
	const size_t SIZE = ARRAY_SIZE * sizeof(float);
	float *dev_v1;
	float *dev_v2; // vector 2
	float* dev_out; // result array, final result is at position 0;
	cudaMallocManaged(&dev_v1, SIZE);
	cudaMallocManaged(&dev_v2, SIZE);

	// host storage
	std::vector<float> host_v1(ARRAY_SIZE);
	std::vector<float> host_v2(ARRAY_SIZE);

	dim3 blocks, threads;
	cuda_gridsize(&threads, &blocks, ARRAY_SIZE, 1, 1);
//	const int BLOCKS = blocks.x;
	cudaMallocManaged(&dev_out, sizeof(float));

	fill<<<blocks, threads>>>(dev_v1, 2, ARRAY_SIZE);
	cudaMemcpy(host_v1.data(), dev_v1, SIZE, cudaMemcpyDeviceToHost);

	// initialize vector 2 with kernel; much faster than using for loops on the cpu
	fill<<<blocks, threads>>>(dev_v2, 1, ARRAY_SIZE);
	cudaMemcpy(host_v2.data(), dev_v2, SIZE, cudaMemcpyDeviceToHost);

	full_dot<BLOCK_SIZE><<<blocks, threads>>>(dev_v1, dev_v2, dev_out, ARRAY_SIZE);
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	cudaDeviceSynchronize();

	std::cout << "GPU: " << *dev_out << std::endl;
	std::cout << "CPU: " << dot(host_v1.data(), host_v2.data(), ARRAY_SIZE)
			<< std::endl;
	cudaFree(dev_v1);
	cudaFree(dev_v2);
	cudaFree(dev_out);
}

void forward_maxpool_layer_gpu() {
//
////
//	size_t out_width = 2;
//	size_t out_height = 2;
//	size_t out_depth = 1;
//	size_t in_height = 8;
//	size_t in_width = 8;
//	size_t bytes = sizeof(float);
//
//	float *input, *output, *max_loc;
//	cudaMalloc(&input, bytes * in_height * in_width);
//	cudaMalloc(&output, bytes * out_depth * out_height * out_width);
//	cudaMalloc(&max_loc, bytes * in_height * in_width);
//
//	dim3 blocks, threads;
//	cuda_gridsize(&threads, &blocks, in_width, in_height, out_depth);
//
//	//fill first
//	fill<<<1, in_height * in_width>>>(input);
//
//	float host_input[in_height * in_width];
//	cudaMemcpy(host_input, input, bytes * in_height * in_width,
//			cudaMemcpyDeviceToHost);
//	print_matrix(host_input, in_height, in_width);
//
//	forward_maxpool_layer_kernel<<<blocks, threads>>>(input, max_loc, output,
//			out_width, out_height, out_depth, in_height, in_width);
//
//	float host_out[out_width * out_height * out_depth];
//
//	cudaMemcpy(host_out, output, bytes * out_depth * out_height * out_width,
//			cudaMemcpyDeviceToHost);
//
//	print_matrix(host_out, out_height, out_width);
//
//	cudaError_t ret = cudaDeviceSynchronize();
//	CUDA_CHECK_RETURN(ret);
//
//	cudaFree(input);
//	cudaFree(output);
//	cudaFree(max_loc);
}

int main(int argc, char **argv) {

//	std::string opt(argv[1]);
	test_dot_product();
//	if (opt == "maxpool") {
//		forward_maxpool_layer_gpu();
//	} else if (opt == "device_vector") {
//
//	}

	return 0;
}

