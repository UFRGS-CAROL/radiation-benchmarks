/*
 * debug.cu
 *
 *  Created on: Jun 20, 2017
 *      Author: carol
 *
 *      this file is only for debugging
 *      classes
 */

__global__ void fill(float *input) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	input[x] = x;
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

void forward_maxpool_layer_gpu() {

//
	size_t out_width = 2;
	size_t out_height = 2;
	size_t out_depth = 1;
	size_t in_height = 8;
	size_t in_width = 8;
	size_t bytes = sizeof(float);

	float *input, *output, *max_loc;
	cudaMalloc(&input, bytes * in_height * in_width);
	cudaMalloc(&output, bytes * out_depth * out_height * out_width);
	cudaMalloc(&max_loc, bytes * in_height * in_width);

	dim3 blocks, threads;
	cuda_gridsize(&threads, &blocks, in_width, in_height, out_depth);

	//fill first
	fill<<<1, in_height * in_width>>>(input);

	float host_input[in_height * in_width];
	cudaMemcpy(host_input, input, bytes * in_height * in_width,
			cudaMemcpyDeviceToHost);
	print_matrix(host_input, in_height, in_width);

	forward_maxpool_layer_kernel<<<blocks, threads>>>(input, max_loc, output,
			out_width, out_height, out_depth, in_height, in_width);

	float host_out[out_width * out_height * out_depth];

	cudaMemcpy(host_out, output, bytes * out_depth * out_height * out_width,
			cudaMemcpyDeviceToHost);

	print_matrix(host_out, out_height, out_width);

	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);

	cudaFree(input);
	cudaFree(output);
	cudaFree(max_loc);
}

int main(int argc, char **argv) {

	std::string opt(argv[1]);

	if(opt == "maxpool"){
		forward_maxpool_layer_gpu();
	}else if (opt == "device_vector"){

	}
	return 0;
}

