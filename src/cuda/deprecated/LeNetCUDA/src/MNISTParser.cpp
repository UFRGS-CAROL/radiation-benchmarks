/*
 * MNISTParser.cpp
 *
 *  Created on: 05/06/2017
 *      Author: fernando
 */

#include "MNISTParser.h"
#include <iostream>
#include <fstream>

#include <cassert>
#include <ctime>
//:
//		test_img_fname(data_path + "/t10k-images-idx3-ubyte"), test_lbl_fname(
//				data_path + "/t10k-labels-idx1-ubyte"), train_img_fname(
//				data_path + "/train-images-idx3-ubyte"), train_lbl_fname(
//				data_path + "/train-labels-idx1-ubyte")
MNISTParser::MNISTParser(std::string img_fname, std::string lbl_fname) {
	this->test_img_fname = img_fname;
	this->test_lbl_fname = lbl_fname;
	this->train_img_fname = img_fname;
	this->train_lbl_fname = lbl_fname;

}

std::string MNISTParser::get_test_img_fname() {
	return this->test_img_fname;
}

std::string MNISTParser::get_test_lbl_fname() {
	return this->test_lbl_fname;
}

//std::vector<Sample*> MNISTParser::load_testing() {
void MNISTParser::load_testing() {
	this->test_sample = load(test_img_fname, test_lbl_fname);
	this->is_train = false;
}

//std::vector<Sample*> MNISTParser::load_training() {
void MNISTParser::load_training() {
	this->train_sample = load(train_img_fname, train_lbl_fname);
	this->is_train = true;
}

void MNISTParser::test() {
	srand((int) time(0));
	size_t i = (int) (rand());
	std::cout << i << std::endl;
	std::cout << (int) test_sample[i]->label << std::endl;
	//test_sample[i]->image->display();

//	size_t j = (int) (rand() * 60000);
	std::cout << (int) (train_sample[i]->label) << std::endl;
	//train_sample[i]->image->display();

}

std::vector<Sample*> MNISTParser::load(std::string fimage, std::string flabel) {
	std::ifstream in;
	in.open(fimage, std::ios::binary | std::ios::in);
	if (!in.is_open()) {
		std::cout << "file opened failed." << std::endl;
	}

	std::uint32_t magic = 0;
	std::uint32_t number = 0;
	std::uint32_t rows = 0;
	std::uint32_t cols = 0;

	in.read((char*) &magic, sizeof(uint32_t));
	in.read((char*) &number, sizeof(uint32_t));
	in.read((char*) &rows, sizeof(uint32_t));
	in.read((char*) &cols, sizeof(uint32_t));

	assert(swapEndien_32(magic) == 2051);
	std::cout << "number:" << swapEndien_32(number) << std::endl;
	assert(swapEndien_32(rows) == 28);
	assert(swapEndien_32(cols) == 28);

	std::vector < std::float_t > row;
	std::vector < std::vector<float_t> > img;
	std::vector<Img> images;

	uint8_t pixel = 0;
	size_t col_index = 0;
	size_t row_index = 0;
	while (!in.eof()) {
		in.read((char*) &pixel, sizeof(uint8_t));
		col_index++;
		row.push_back((float_t) pixel);
		if (col_index == 28) {
			img.push_back(row);

			row.clear();
			col_index = 0;

			row_index++;
			if (row_index == 28) {
				Img i = new Image(28, img);
				i->upto_32();
				//i->display();
				images.push_back(i);
				img.clear();
				row_index = 0;
			}
		}
	}

	in.close();

	assert(images.size() == swapEndien_32(number));

	//label
	in.open(flabel, std::ios::binary | std::ios::in);
	if (!in.is_open()) {
		std::cout << "failed opened label file";
	}

	in.read((char*) &magic, sizeof(uint32_t));
	in.read((char*) &number, sizeof(uint32_t));

	assert(2049 == swapEndien_32(magic));
	assert(swapEndien_32(number) == images.size());

	std::vector < uint8_t > labels;

	uint8_t label;
	while (!in.eof()) {
		in.read((char*) &label, sizeof(uint8_t));
		//std::cout << (int)label << std::endl;
		labels.push_back(label);
	}

	std::vector<Sample*> samples;
	for (size_t i = 0; i < swapEndien_32(number); i++) {
		samples.push_back(new Sample(labels[i], images[i]->extend()));
	}

	std::cout << "Loading complete" << std::endl;
	in.close();
	return samples;
}

// reverse endien for uint32_t
std::uint32_t MNISTParser::swapEndien_32(std::uint32_t value) {
	return ((value & 0x000000FF) << 24) | ((value & 0x0000FF00) << 8)
			| ((value & 0x00FF0000) >> 8) | ((value & 0xFF000000) >> 24);
}

std::ostream& operator<<(std::ostream& oss, const MNISTParser& t) {
	oss << "Img fname " << t.test_img_fname << std::endl << "Label fname "
			<< t.test_lbl_fname;
	return oss;
}

Sample* MNISTParser::get_sample(int i) {
	if (this->is_train)
		return this->train_sample[i];
	return this->test_sample[i];
}
