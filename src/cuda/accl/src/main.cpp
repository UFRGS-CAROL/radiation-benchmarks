/*
 * -- NUPAR: A Benchmark Suite for Modern GPU Architectures
 *    NUPAR - 2 December 2014
 *    Fanny Nina-Paravecino
 *    Northeastern University
 *    NUCAR Research Laboratory
 *
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:
 *
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:
 * This  product  includes  software  developed  at  the Northeastern U.
 *
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.
 *
 * -- Disclaimer:
 *
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------
 */
/*
 * Include files
 */
//#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <omp.h>

#define MAX_LABELS 262144
#define BUF_SIZE 256

#include "Parameters.h"
#include "log_helper.h"
#include "utils.h"

#include "cuda_utils.h"
#include "generic_log.h"

#include "image.h"
#include "misc.h"
#include "accl.h"

class errorHandler {
};
//using namespace std;
/*
 * ---------------------------------------------------------------------
 * Prototypes
 * ---------------------------------------------------------------------
 */
double getWallTime();
double getCpuTime();
void pgmRead(std::ifstream &file, char *buf);
image<uchar> *loadPGM(const char *name);
image<int> *imageUcharToInt(image<uchar> *input);
void savePGM(image<rgb> *im, const char *name);
void acclSerial(image<int> *imInt, int *spans, int *components, const int rows,
		const int cols, image<rgb> *output);
/*
 * RGB generation colors randomly
 */
rgb randomRgb() {
	rgb c;

	c.r = (uchar) rand();
	c.g = (uchar) rand();
	c.b = (uchar) rand();
	return c;
}

///*
// * getWallTime: Compute timing of execution including I/O
// */
//double getWallTime() {
//	struct timeval time;
//	if (gettimeofday(&time, NULL)) {
//		printf("Error getting time\n");
//		return 0;
//	}
//	return (double) time.tv_sec + (double) time.tv_usec * .000001;
//}
//
///*
// * getCpuTime: Compute timing of execution using Clocks function from C++
// */
//double getCpuTime() {
//	return (double) clock() / CLOCKS_PER_SEC;
//}

/*
 * pgmRead: read a pgm image file
 * Parameters:
 * - file:  std::ifstream
 *          path of the pgm image file
 * - buf:   char*
 *          buffer where information will be allocated
 */
void pgmRead(std::ifstream &file, char *buf) {
	char doc[BUF_SIZE];
	char c;

	file >> c;
	while (c == '#') {
		file.getline(doc, BUF_SIZE);
		file >> c;
	}
	file.putback(c);

	file.width(BUF_SIZE);
	file >> buf;
	file.ignore();
}

/*
 * loadPGM: load pgm file and return it in a image<uchar> structure
 * Parameters:
 * - name:  const char*
 *          path of the pgm image file
 * Return:
 * - image<uchar>: image loaded in an uchar structure
 */
image<uchar> *loadPGM(const std::string& name) {
	char buf[BUF_SIZE];

	/*
	 * read header
	 */
	std::ifstream file(name, std::ios::in | std::ios::binary);
	pgmRead(file, buf);
	if (strncmp(buf, "P5", 2))
		throw errorHandler();

	pgmRead(file, buf);
	int width = atoi(buf);
	pgmRead(file, buf);
	int height = atoi(buf);

	pgmRead(file, buf);
	if (atoi(buf) > UCHAR_MAX)
		throw errorHandler();

	/* read data */
	image<uchar> *im = new image<uchar>(width, height);
	file.read((char *) imPtr(im, 0, 0), width * height * sizeof(uchar));
	return im;
}

/*
 * savePGM: save pgm file
 * Parameters:
 * - im:    image<rgb>
 *          image in rgb colors to save the final output image
 * - name:  const char*
 *          path for the image output file
 */
void savePGM(image<rgb> *im, const char *name) {
	int width = im->width();
	int height = im->height();
	std::ofstream file(name, std::ios::out | std::ios::binary);

	file << "P6\n" << width << " " << height << "\n" << UCHAR_MAX << "\n";
	file.write((char *) imPtr(im, 0, 0), width * height * sizeof(rgb));
}

/*
 * imageUcharToInt: convert image from uchar to integer
 * Parameters:
 * - input: image<uchar>
 *          image in uchar to convert to integer values
 * Return:
 * - image<int>: image with integer values
 */
image<int> *imageUcharToInt(image<uchar> *input) {
	int width = input->width();
	int height = input->height();
	image<int> *output = new image<int>(width, height, false);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			imRef(output, x, y) = imRef(input, x, y);
		}
	}
	return output;
}

//double mysecond() {
//	struct timeval tp;
//	struct timezone tzp;
//	int i = gettimeofday(&tp, &tzp);
//	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
//}
template<typename real_t>
bool read_from_file(std::string& path, std::vector<real_t>& array) {
	std::ifstream input(path, std::ios::binary);
	if (input.good()) {
		input.read(reinterpret_cast<char*>(array.data()),
				array.size() * sizeof(real_t));
		input.close();
		return false;
	}
	return true;
}

template<typename real_t>
bool write_to_file(std::string& path, std::vector<real_t>& array) {
	std::ofstream output(path, std::ios::binary);
	if (output.good()) {
		output.write(reinterpret_cast<char*>(array.data()),
				array.size() * sizeof(real_t));
		output.close();

		return false;
	}
	return true;
}

template<typename int_t>
void readGold(std::vector<int_t>& gold_spans,
		std::vector<int_t>& gold_components, const std::string& fpath) {

	std::ifstream input(fpath, std::ios::binary);
	if (input.good()) {
		input.read(reinterpret_cast<char*>(gold_spans.data()),
				gold_spans.size() * sizeof(int_t));
		input.read(reinterpret_cast<char*>(gold_components.data()),
				gold_components.size() * sizeof(int_t));

		input.close();
	} else {
		throw_line("Could not read file " + fpath);
	}
}

//void usage() {
//	std::cout
//			<< "Usage: ./accl <N frames in the image> <(HyperQ) Frames per Stream> <Input image path> <GOLD path> <#iteractions>"
//			<< std::endl;
//}

int main(int argc, char** argv) {
//	if (argc < 6) {
//		usage();
//		exit(0);
//	}

	Parameters parameters(argc, argv);

	//	"frames:%d, framesPerStream:%d"
	std::string test_info = "frames:" + std::to_string(parameters.nFrames);
	test_info += ", framesPerStream:"
			+ std::to_string(parameters.nFramesPerStream);
	std::string test_name = "cudaCCL";

	rad::Log log(test_name, test_info);

	if (parameters.verbose) {
		std::cout << parameters << std::endl;
		std::cout << log << std::endl;
		std::cout << "Accelerated Connected Component Labeling" << std::endl;
		std::cout << "========================================" << std::endl;
		std::cout << "Loading input image..." << std::endl;
	}

	image<uchar> *input = loadPGM(parameters.input);
	const int width = input->width();
	const int height = input->height();

	/*
	 * Declaration of Variables
	 */
	image<int> *imInt = new image<int>(width, height);
	image<rgb> *output1 = new image<rgb>(width, height);
	image<rgb> *output2 = new image<rgb>(width, height);
	imInt = imageUcharToInt(input);

	auto nFrames = parameters.nFrames;
	auto nFramsPerStream = parameters.nFramesPerStream;

	const int rows = nFrames * 512;
	const int cols = 512;
	const int imageSize = rows * cols;
	int *image = new int[imageSize];
	memcpy(image, imInt->data, rows * cols * sizeof(int));

	/*
	 * Buffers
	 */
	const int colsSpans = ((cols + 2 - 1) / 2) * 2; /*ceil(cols/2)*2*/
	const int spansSize = colsSpans * rows;
	const int componentsSize = (colsSpans / 2) * rows;
	std::vector<int> spans(spansSize);
	std::vector<int> components(componentsSize);
	std::vector<int> gold_spans(spansSize);
	std::vector<int> gold_components(componentsSize);

	readGold(gold_spans, gold_components, parameters.gold);

	for (size_t loop1 = 0; loop1 < parameters.iterations; loop1++) {

		/*
		 * Initialize
		 */
		std::fill(spans.begin(), spans.end(), -1);
		std::fill(components.begin(), components.end(), -1);

		/*
		 * CUDA
		 */
		std::cout << "Passou " << spans.data() << " " << components.data()
				<< " " << image << " " << nFrames << " " << nFramsPerStream
				<< " " << rows << " " << cols << std::endl;
		double ktime = 0.0;
		ktime = acclCuda(spans.data(), components.data(), image, nFrames,
				nFramsPerStream, rows, cols, 1);
		//printf("acclCuda time: %.5f", ktime);

		ktime /= 1000;

		// output validation
		int kernel_errors = 0;

#pragma omp parallel for
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < colsSpans; j++) {
				int index = i + j * rows;
#pragma omp critical
				if (spans[index] != gold_spans[index]) {
					char error_detail[150];
					snprintf(error_detail, 150,
							"t: [spans], p: [%d][%d], r: %d, e: %d", i, j,
							spans[index], gold_spans[index]);
					printf("%s\n", error_detail);
					log_error_detail(error_detail);
					kernel_errors++;
				}
			}
		}
//		for (int k = 0; k < spansSize; k++) {
//#pragma omp critical
//			if (spans[k] != gold_spans[k]) {
//				char error_detail[150];
//				snprintf(error_detail, 150, "t: [spans], p: [%d], r: %d, e: %d",
//						k, spans[k], gold_spans[k]);
//				printf("%s\n", error_detail);
//				log_error_detail(error_detail);
//				kernel_errors++;
//			}
//		}
		// output validation
		const int component_width = (colsSpans / 2);
#pragma omp parallel for
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < component_width; j++) {
				int index = i + j * rows;
#pragma omp critical
				if (components[index] != gold_components[index]) {
					char error_detail[150];
					snprintf(error_detail, 150,
							"t: [components], p: [%d][%d], r: %d, e: %d", i, j,
							components[index], gold_components[index]);
					printf("%s\n", error_detail);
					log_error_detail(error_detail);
					kernel_errors++;
				}
			}
		}

//		for (int k = 0; k < componentsSize; k++) {
//			if (components[k] != gold_components[k])
//#pragma omp critical
//					{
//				char error_detail[150];
//				snprintf(error_detail, 150,
//						"t: [components], p: [%d], r: %d, e: %d", k,
//						components[k], gold_components[k]);
//				printf("%s\n", error_detail);
//				log_error_detail(error_detail);
//				kernel_errors++;
//			}
//		}
		log_error_count(kernel_errors);

		std::cout << "." << std::endl;
	}

//	end_log_file();

	std::cout << "Image Segmentation ended" << std::endl;
	return 0;
}
