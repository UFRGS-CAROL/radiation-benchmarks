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
#include <memory>

//#include <omp.h>

//#define MAX_LABELS 262144
#define BUF_SIZE 256

#include "Parameters.h"
#include "utils.h"
#include "cuda_utils.h"
#include "generic_log.h"
#include "image.h"
#include "misc.h"
#include "accl.h"

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
std::shared_ptr<image<uchar>> loadPGM(const std::string& name) {
	char buf[BUF_SIZE];

	/*
	 * read header
	 */
	std::ifstream file(name, std::ios::in | std::ios::binary);
	pgmRead(file, buf);
	if (strncmp(buf, "P5", 2) != 0)
		throw_line("P5");

	pgmRead(file, buf);
	int width = atoi(buf);
	pgmRead(file, buf);
	int height = atoi(buf);

	pgmRead(file, buf);
	if (atoi(buf) > UCHAR_MAX)
		throw_line("CHAR MAX");

	/* read data */
	std::shared_ptr<image<uchar>> im = std::make_shared<image<uchar>>(width,
			height);
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
void savePGM(std::shared_ptr<image<rgb>> im, const std::string& name) {
	int width = im->width();
	int height = im->height();
	std::ofstream file(name, std::ios::binary);

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
std::shared_ptr<image<int>> imageUcharToInt(
		const std::shared_ptr<image<uchar>>& input) {
	int width = input->width();
	int height = input->height();
	std::shared_ptr<image<int>> output = std::make_shared<image<int>>(width,
			height, false);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			imRef(output, x, y) = imRef(input, x, y);
		}
	}
	return output;
}

template<typename int_t>
void writeGold(std::vector<int_t>& gold_spans,
		std::vector<int_t>& gold_components, const std::string& fpath) {

	std::ofstream output(fpath, std::ios::binary);
	if (output.good()) {
		output.write(reinterpret_cast<char*>(gold_spans.data()),
				gold_spans.size() * sizeof(int_t));
		output.write(reinterpret_cast<char*>(gold_components.data()),
				gold_components.size() * sizeof(int_t));

		output.close();

	} else {
		throw_line("Could not write file " + fpath);
	}

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


int main(int argc, char** argv) {
	Parameters parameters(argc, argv);

	//	"frames:%d, framesPerStream:%d"
	std::string test_info = "frames:" + std::to_string(parameters.nFrames);
	test_info += ", framesPerStream:"
			+ std::to_string(parameters.nFramesPerStream) + get_multi_compiler_header();

	std::string test_name = "cudaCCL";

	rad::Log log(test_name, test_info);

	if (parameters.verbose) {
		std::cout << parameters << std::endl;
		std::cout << log << std::endl;
		std::cout << "Accelerated Connected Component Labeling" << std::endl;
		std::cout << "========================================" << std::endl;
		std::cout << "Loading input image..." << std::endl;
	}

	std::shared_ptr<image<uchar>> input = loadPGM(parameters.input);
	const int width = input->width();
	const int height = input->height();

	/*
	 * Declaration of Variables
	 */
//	image<int> *imInt = new image<int>(width, height);
//	image<rgb> *output1 = new image<rgb>(width, height);
//	image<rgb> *output2 = new image<rgb>(width, height);
	std::shared_ptr<image<int>> imInt = imageUcharToInt(input); //std::make_shared<image<int>>(width, height);
	std::shared_ptr<image<rgb>> output1 = std::make_shared<image<rgb>>(width,
			height);
	std::shared_ptr<image<rgb>> output2 = std::make_shared<image<rgb>>(width,
			height);
//	imInt = imageUcharToInt(input);

	int nFrames = parameters.nFrames;
	int nFramsPerStream = parameters.nFramesPerStream;

	const int rows = nFrames * 512;
	const int cols = 512;
	const int imageSize = rows * cols;
	std::vector<int> image(imageSize);

	//	memcpy(image, imInt->data, rows * cols * sizeof(int));
	std::copy(imInt->data, imInt->data + rows * cols, image.begin());

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

	if (!parameters.generate) {
		readGold(gold_spans, gold_components, parameters.gold);
		if (parameters.debug) {
			gold_components[22] = 22;
			gold_spans[24] = 33;
		}
	}

	std::fill(spans.begin(), spans.end(), -1);
	std::fill(components.begin(), components.end(), -1);

	rad::DeviceVector<int> devSpans = spans;
	rad::DeviceVector<int> devComponents = components;
	rad::DeviceVector<int> devImage = image;
	const rad::DeviceVector<int> dev_spans_set(spans);
	const rad::DeviceVector<int> dev_components_set(components);

	uint nStreams = nFrames / nFramsPerStream;

	std::vector < cudaStream_t > streams(nStreams);
	for (auto& stream : streams) {
		rad::checkFrameworkErrors(cudaStreamCreate(&(stream)));
	}
	for (size_t loop1 = 0; loop1 < parameters.iterations; loop1++) {

		/*
		 * Initialize
		 */
		auto set_time = rad::mysecond();
		devSpans = dev_spans_set;
		devComponents = dev_components_set;
		set_time = rad::mysecond() - set_time;

		/*
		 * CUDA
		 */

		//std::vector<int>& out, std::vector<int>& components,
		//		const std::vector<int>& in
		auto ktime = acclCuda(devSpans, devComponents, devImage, nFrames,
				nFramsPerStream, rows, cols, 1, log, streams);
		//printf("acclCuda time: %.5f", ktime);

		auto copy_time = rad::mysecond();
		devComponents.to_vector(components);
		devSpans.to_vector(spans);
		copy_time = rad::mysecond() - copy_time;

		auto comparisson_time = rad::mysecond();
		// output validation
		int kernel_errors = 0;
		if (!parameters.generate) {
#pragma omp parallel for
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < colsSpans; j++) {
					int index = i + j * rows;
					if (spans[index] != gold_spans[index]) {
						//"t: [spans], p: [%d][%d], r: %d, e: %d
						auto gc = std::to_string(gold_spans[index]);
						auto fc = std::to_string(spans[index]);
						auto istr = std::to_string(i);
						auto jstr = std::to_string(j);
						std::string error_detail;
						error_detail = "t: [spans], p: [" + istr + "][" + jstr
								+ "], ";
						error_detail += "r: " + fc + ", e: " + gc;

#pragma omp critical
						{
							if (parameters.verbose && kernel_errors <= 10) {
								std::cout << error_detail << std::endl;
							}
							log.log_error_detail(error_detail);
							kernel_errors++;
						}

					}
				}
			}

			// output validation
			const int component_width = (colsSpans / 2);
#pragma omp parallel for
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < component_width; j++) {
					int index = i + j * rows;
					if (components[index] != gold_components[index]) {
						auto gc = std::to_string(gold_components[index]);
						auto fc = std::to_string(components[index]);
						auto istr = std::to_string(i);
						auto jstr = std::to_string(j);
						std::string error_detail;
						error_detail = "t: [components], p: [" + istr + "]["
								+ jstr + "], ";
						error_detail += "r: " + fc + ", e: " + gc;

#pragma omp critical
						{
							//					log_error_detail(error_detail);
							if (parameters.verbose && kernel_errors <= 10) {
								std::cout << error_detail << std::endl;
							}
							log.log_error_detail(error_detail);
							kernel_errors++;
						}
					}
				}
			}
		}
		comparisson_time = rad::mysecond() - comparisson_time;
		log.update_errors();


		if (parameters.verbose) {
			std::cout << "Iteration " << loop1 << " - kernel time: " << ktime
					<< " errors: " << kernel_errors << std::endl;
			auto wasted_time = copy_time + set_time + comparisson_time;
			auto overall_time = ktime + wasted_time;
			std::cout << "Overall time: " << overall_time << " wasted time: "
					<< wasted_time << " - "
					<< int((wasted_time / overall_time) * 100.0) << "%"
					<< std::endl;

		} else {
			std::cout << "." << std::endl;

		}
	}

	if (parameters.generate) {
		writeGold(spans, components, parameters.gold);
	}

	for (auto& stream : streams) {
		rad::checkFrameworkErrors(cudaStreamDestroy(stream));
	}
	std::cout << "Image Segmentation ended" << std::endl;
	return 0;
}
