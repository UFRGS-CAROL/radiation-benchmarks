//====================================================================================================100
//		UPDATE
//====================================================================================================100

//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a  
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts   
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments

//====================================================================================================100
//	DEFINE / INCLUDE
//====================================================================================================100

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

#include "define.c"
#include "extract_kernel.cu"
#include "prepare_kernel.cu"
#include "reduce_kernel.cu"
#include "srad_kernel.cu"
#include "srad2_kernel.cu"
#include "compress_kernel.cu"
#include "graphics.c"
#include "resize.c"
#include "timer.c"

#include <string>
#include <iostream>
#include "device.c"				// (in library path specified to compiler)	needed by for device functions

#ifdef LOGS
#include "log_helper.h"
#endif

#define MAX_ERROR_THRESHOLD 0.00001

void compute_srad(long Ne, dim3 blocks2, int no, int mul, int blocks_x,
		int mem_size_single, fp meanROI, fp meanROI2, fp varROI,
		fp q0sqr, dim3& blocks, dim3& threads, fp*& d_I,
		fp*& d_sums, fp*& d_sums2, fp& total, fp& total2, long & NeROI,
		fp& lambda, int& Nr, int& Nc, int*& d_iN, int*& d_iS, int*& d_jE,
		int*& d_jW, fp*& d_dN, fp*& d_dS, fp*& d_dW,
		fp*& d_dE, fp*& d_c) {
	//================================================================================80
	// 	SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//================================================================================80
	extract<<<blocks, threads>>>(Ne, d_I);
	checkCUDAError("extract");
	//		time7 = get_time();
	//================================================================================80
	// 	COMPUTATION
	//================================================================================80
	// printf("iterations: ");
	// printf("%d ", iter);
	// fflush(NULL);
	// execute square kernel
	prepare<<<blocks, threads>>>(Ne, d_I, d_sums, d_sums2);
	checkCUDAError("prepare");
	// performs subsequent reductions of sums
	blocks2.x = blocks.x; // original number of blocks
	blocks2.y = blocks.y;
	no = Ne; // original number of sum elements
	mul = 1; // original multiplier
	while (blocks2.x != 0) {

		checkCUDAError("before reduce");

		// run kernel
		reduce<<<blocks2, threads>>>(Ne, no, mul, d_sums, d_sums2);

		checkCUDAError("reduce");

		// update execution parameters
		no = blocks2.x;					// get current number of elements
		if (blocks2.x == 1) {
			blocks2.x = 0;
		} else {
			mul = mul * NUMBER_THREADS;				// update the increment
			blocks_x = blocks2.x / threads.x;			// number of blocks
			if (blocks2.x % threads.x != 0) {// compensate for division remainder above by adding one grid
				blocks_x = blocks_x + 1;
			}
			blocks2.x = blocks_x;
			blocks2.y = 1;
		}

		checkCUDAError("after reduce");

	}
	checkCUDAError("before copy sum");
	// copy total sums to device
	mem_size_single = sizeof(fp) * 1;
	cudaMemcpy(&total, d_sums, mem_size_single, cudaMemcpyDeviceToHost);
	cudaMemcpy(&total2, d_sums2, mem_size_single, cudaMemcpyDeviceToHost);
	checkCUDAError("copy sum");
	// calculate statistics
	meanROI = total / fp(NeROI);
	meanROI2 = meanROI * meanROI; //
	varROI = (total2 / fp(NeROI)) - meanROI2;
	q0sqr = varROI / meanROI2; // gets standard deviation of ROI
	// execute srad kernel
	srad<<<blocks, threads>>>(lambda, // SRAD coefficient
			Nr, // # of rows in input image
			Nc, // # of columns in input image
			Ne, // # of elements in input image
			d_iN, // indices of North surrounding pixels
			d_iS, // indices of South surrounding pixels
			d_jE, // indices of East surrounding pixels
			d_jW, // indices of West surrounding pixels
			d_dN, // North derivative
			d_dS, // South derivative
			d_dW, // West derivative
			d_dE, // East derivative
			q0sqr, // standard deviation of ROI
			d_c, // diffusion coefficient
			d_I); // output image
	checkCUDAError("srad");
	// execute srad2 kernel
	srad2<<<blocks, threads>>>(lambda, // SRAD coefficient
			Nr, // # of rows in input image
			Nc, // # of columns in input image
			Ne, // # of elements in input image
			d_iN, // indices of North surrounding pixels
			d_iS, // indices of South surrounding pixels
			d_jE, // indices of East surrounding pixels
			d_jW, // indices of West surrounding pixels
			d_dN, // North derivative
			d_dS, // South derivative
			d_dW, // West derivative
			d_dE, // East derivative
			d_c, // diffusion coefficient
			d_I); // output image
	checkCUDAError("srad2");
	// printf("\n");
	//		time8 = get_time();
	//================================================================================80
	// 	SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//================================================================================80
	compress<<<blocks, threads>>>(Ne, d_I);
	checkCUDAError("compress");
}

void compare_and_log(PGMImage *gold, PGMImage *found) {
	int error_count = 0;
	for (int i = 0; i < gold->height; i++) {
		for (int j = 0; j < gold->width; j++) {
			fp g = gold->data[i * gold->width + j];
			fp f = found->data[i * gold->width + j];
			fp diff = fabs(g - f);
			if (diff > MAX_ERROR_THRESHOLD) {
				std::string error_detail = "position[" + std::to_string(i)
						+ "][" + std::to_string(j) + "] expected: "
						+ std::to_string(g) + "read: " + std::to_string(f);
				error_count++;
#ifdef LOGS
				log_error_detail((char*)error_detail.c_str());

#else
				std::cout << error_detail << "\n";
#endif
			}
		}
	}

#ifdef LOGS
	log_error_count(error_count);
#endif
}

void inline start_benchmark(char *gold_path, int iterations, fp lambda) {
#ifdef LOGS
	std::string header = std::string("gold: ") + std::string(gold_path)
					+ " iterations: " + std::to_string(iterations) +
					" lambda: " + std::to_string(lambda);
	start_log_file("cudaSRADV1", (char*)header.c_str());

#endif
}

void inline end_benchmark() {
#ifdef LOGS
	end_log_file();
#endif
}

void inline start_iteration_call() {
#ifdef LOGS
	start_iteration();
#endif
}

void inline end_iteration_call() {
#ifdef LOGS
	end_iteration();
#endif
}

void save_gold(PGMImage *img, char *gold_path) {
	int size = img->height * img->width;

	FILE* fout = fopen(gold_path, "wb");
	if (fout) {
		fwrite(&img->height, sizeof(unsigned), 1, fout);
		fwrite(&img->width, sizeof(unsigned), 1, fout);
		fwrite(img->data, sizeof(fp), size, fout);

	} else {
		std::cout << gold_path << " directory not found\n";
	}
}

void load_gold(PGMImage *img, char *gold_path) {
	FILE* fin = fopen(gold_path, "rb");
	if (fin) {
		fread(&img->height, sizeof(unsigned), 1, fin);
		fread(&img->width, sizeof(unsigned), 1, fin);

		int size = img->height * img->width;

		malloc_img_data(img);
		fread(img->data, sizeof(fp), size, fin);

	} else {
		std::cout << gold_path << " directory not found\n";
	}

}

//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100

int main(int argc, char *argv[]) {

	//================================================================================80
	// 	VARIABLES
	//================================================================================80

	// inputs image, input paramenters
	fp* image_ori;										// originalinput image
	int image_ori_rows;
	int image_ori_cols;
	long image_ori_elem;

	// inputs image, input paramenters
	fp* image;													// input image
	int Nr, Nc;								// IMAGE nbr of rows/cols/elements
	long Ne;

	// algorithm parameters
	int niter;												// nbr of iterations
	fp lambda;												// update step size

	// size of IMAGE
	int r1, r2, c1, c2;					// row/col coordinates of uniform ROI
	long NeROI;											// ROI nbr of elements

	// surrounding pixel indicies
	int *iN, *iS, *jE, *jW;

	// counters
	int iter;   // primary loop
	long i, j;    // image row/col

	// memory sizes
	int mem_size_i;
	int mem_size_j;
	int mem_size_single;

	//================================================================================80
	// 	GPU VARIABLES
	//================================================================================80

	// CUDA kernel execution parameters
	dim3 threads;
	int blocks_x;
	dim3 blocks;
	dim3 blocks2;
	dim3 blocks3;

	// memory sizes
	int mem_size;										// matrix memory size

	// HOST
	int no;
	int mul;
	fp total;
	fp total2;
	fp meanROI;
	fp meanROI2;
	fp varROI;
	fp q0sqr;

	// DEVICE
	fp* d_sums;													// partial sum
	fp* d_sums2;
	int* d_iN;
	int* d_iS;
	int* d_jE;
	int* d_jW;
	fp* d_dN;
	fp* d_dS;
	fp* d_dW;
	fp* d_dE;
	fp* d_I;											// input IMAGE on DEVICE
	fp* d_c;

//	time1 = get_time();

	//================================================================================80
	// 	GET INPUT PARAMETERS
	//================================================================================80
	//parameters
	char *input_path;
	char *gold_path;
	// 0 for generate and 1 for radiation test
	int mode = 0;
	if (argc != 6) {
		printf("ERROR: wrong number of arguments\n");
		printf("usage: %s "
				"<n iterations> <lambda> <image input path> "
				"<mode: 0 for generate and 1 for radiation test> <gold path>\n",
				argv[0]);
		return 0;
	} else {
		niter = atoi(argv[1]);
		lambda = atof(argv[2]);
		input_path = argv[3];
		mode = atoi(argv[4]);
		gold_path = argv[5];

//		Nr = atoi(argv[3]);					// it is 502 in the original image
//		Nc = atoi(argv[4]);					// it is 458 in the original image
	}

	//================================================================================80
	// 	Starting bench
	//================================================================================80
	if (mode == 1)
		start_benchmark(gold_path, niter, lambda);

	//================================================================================80
	// 	READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	//================================================================================80
	std::cout << "Reading image\n";

	// read image

//	image_ori = (fp*) malloc(sizeof(fp) * image_ori_elem);

	PGMImage img;
//	read_graphics("image.pgm", image_ori, image_ori_rows, image_ori_cols, 1);
	read_pgm_image(input_path, &img, 1);
	print_image(&img);
	image_ori = img.data;
	Nr = img.height;
	Nc = img.width;
	image_ori_rows = Nr;
	image_ori_cols = Nc;
	image_ori_elem = Nr * Nc;

	//================================================================================80
	// 	read gold
	//================================================================================80
	PGMImage gold_img;
	if (mode == 1) {
//		read_pgm_image(gold_path, &gold_img, 1);
		load_gold(&gold_img, gold_path);
	}

	//================================================================================80
	// 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	//================================================================================80
	std::cout << "Resizing the image\n";
	Ne = Nr * Nc;

	image = (fp*) malloc(sizeof(fp) * Ne);
	fp *image_output = (fp*) malloc(sizeof(fp) * Ne);

	resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

	//================================================================================80
	// 	SETUP
	//================================================================================80
	std::cout << "Setup on GPU\n";
	r1 = 0;											// top row index of ROI
	r2 = Nr - 1;									// bottom row index of ROI
	c1 = 0;											// left column index of ROI
	c2 = Nc - 1;									// right column index of ROI

	// ROI image size
	NeROI = (r2 - r1 + 1) * (c2 - c1 + 1);// number of elements in ROI, ROI size

	// allocate variables for surrounding pixels
	mem_size_i = sizeof(int) * Nr;											//
	iN = (int *) malloc(mem_size_i);				// north surrounding element
	iS = (int *) malloc(mem_size_i);				// south surrounding element
	mem_size_j = sizeof(int) * Nc;											//
	jW = (int *) malloc(mem_size_j);				// west surrounding element
	jE = (int *) malloc(mem_size_j);				// east surrounding element

	// N/S/W/E indices of surrounding pixels (every element of IMAGE)
	for (i = 0; i < Nr; i++) {
		iN[i] = i - 1;						// holds index of IMAGE row above
		iS[i] = i + 1;						// holds index of IMAGE row below
	}
	for (j = 0; j < Nc; j++) {
		jW[j] = j - 1;				// holds index of IMAGE column on the left
		jE[j] = j + 1;				// holds index of IMAGE column on the right
	}

	// N/S/W/E boundary conditions, fix surrounding indices outside boundary of image
	iN[0] = 0;						// changes IMAGE top row index from -1 to 0
	iS[Nr - 1] = Nr - 1;	// changes IMAGE bottom row index from Nr to Nr-1
	jW[0] = 0;				// changes IMAGE leftmost column index from -1 to 0
	jE[Nc - 1] = Nc - 1;// changes IMAGE rightmost column index from Nc to Nc-1

	//================================================================================80
	// 	GPU SETUP
	//================================================================================80

	// allocate memory for entire IMAGE on DEVICE
	mem_size = sizeof(fp) * Ne;	// get the size of float representation of input IMAGE
	cudaMalloc((void **) &d_I, mem_size);									//

	// allocate memory for coordinates on DEVICE
	cudaMalloc((void **) &d_iN, mem_size_i);								//
	cudaMemcpy(d_iN, iN, mem_size_i, cudaMemcpyHostToDevice);				//
	cudaMalloc((void **) &d_iS, mem_size_i);								//
	cudaMemcpy(d_iS, iS, mem_size_i, cudaMemcpyHostToDevice);				//
	cudaMalloc((void **) &d_jE, mem_size_j);								//
	cudaMemcpy(d_jE, jE, mem_size_j, cudaMemcpyHostToDevice);				//
	cudaMalloc((void **) &d_jW, mem_size_j);								//
	cudaMemcpy(d_jW, jW, mem_size_j, cudaMemcpyHostToDevice);			//

	// allocate memory for partial sums on DEVICE
	cudaMalloc((void **) &d_sums, mem_size);								//
	cudaMalloc((void **) &d_sums2, mem_size);								//

	// allocate memory for derivatives
	cudaMalloc((void **) &d_dN, mem_size);									//
	cudaMalloc((void **) &d_dS, mem_size);									//
	cudaMalloc((void **) &d_dW, mem_size);									//
	cudaMalloc((void **) &d_dE, mem_size);									//

	// allocate memory for coefficient on DEVICE
	cudaMalloc((void **) &d_c, mem_size);									//

	checkCUDAError("setup");

	//================================================================================80
	// 	KERNEL EXECUTION PARAMETERS
	//================================================================================80

	// all kernels operating on entire matrix
	threads.x = NUMBER_THREADS;		// define the number of threads in the block
	threads.y = 1;
	blocks_x = Ne / threads.x;
	if (Ne % threads.x != 0) {// compensate for division remainder above by adding one grid
		blocks_x = blocks_x + 1;
	}
	blocks.x = blocks_x;			// define the number of blocks in the grid
	blocks.y = 1;
	std::cout << "Executing iterations\n";
	// execute main loop
	for (iter = 0; iter < niter; iter++) {// do for the number of iterations input parameter
		//================================================================================80
		// 	COPY INPUT TO CPU
		//================================================================================80
		cudaMemcpy(d_I, image, mem_size, cudaMemcpyHostToDevice);

		double time = get_time();
		//for logs
		if (mode == 1)
			start_iteration_call();
		compute_srad(Ne, blocks2, no, mul, blocks_x, mem_size_single, meanROI,
				meanROI2, varROI, q0sqr, blocks, threads, d_I, d_sums, d_sums2,
				total, total2, NeROI, lambda, Nr, Nc, d_iN, d_iS, d_jE, d_jW,
				d_dN, d_dS, d_dW, d_dE, d_c);

		//for logs
		if (mode == 1)
			end_iteration_call();

		std::cout << "Time for iteration " << get_time() - time << "\n";
		//================================================================================80
		// 	COPY RESULTS BACK TO CPU
		//================================================================================80
		cudaMemcpy(image_output, d_I, mem_size, cudaMemcpyDeviceToHost);

		checkCUDAError("copy back");

		time = get_time();
		//compare
		if (mode == 1) {
			PGMImage found = make_pgm_img(image_output, Nr, Nc,
					gold_img.max_gray_value, gold_img.magic_number);
			compare_and_log(&gold_img, &found);
		}

		std::cout << "Time for comparing " << get_time() - time << "\n";
	}


	//================================================================================80
	// WRITE IMAGE AFTER PROCESSING
	//================================================================================80
	if (mode == 0) {
//		write_graphics(gold_path, image, Nr, Nc, 1, 255);
		PGMImage found = make_pgm_img(image, Nr, Nc, gold_img.max_gray_value,
				gold_img.magic_number);
		save_gold(&found, gold_path);
	}
	//================================================================================80
	//	DEALLOCATE
	//================================================================================80
	std::cout << "Finishing and deallocate\n";
	free(image_ori);
	free(image);
	free(image_output);
	free(iN);
	free(iS);
	free(jW);
	free(jE);

	cudaFree(d_I);
	cudaFree(d_c);
	cudaFree(d_iN);
	cudaFree(d_iS);
	cudaFree(d_jE);
	cudaFree(d_jW);
	cudaFree(d_dN);
	cudaFree(d_dS);
	cudaFree(d_dE);
	cudaFree(d_dW);
	cudaFree(d_sums);
	cudaFree(d_sums2);

	if (mode == 1) {
		free_img(&gold_img);
	}
	if (mode == 1)
		end_benchmark();
}

//====================================================================================================100
//	END OF FILE
//====================================================================================================100
