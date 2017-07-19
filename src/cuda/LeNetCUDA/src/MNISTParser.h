/*
 * MNISTParser.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef MNISTPARSER_H_
#define MNISTPARSER_H_

#include <string>
#include <vector>
#include <cstdint>
#include "Util.h"

class MNISTParser {
public:
	MNISTParser(std::string img_fname, std::string lbl_fname);

	std::string get_test_img_fname();
	std::string get_test_lbl_fname();

	void load_testing();

	void load_training();

	void test();

	void add_new_sample(Sample *s);
	Sample* get_sample(int i);



private:
	std::vector<Sample*> load(std::string fimage, std::string flabel);
	// reverse endien for uint32_t
	std::uint32_t swapEndien_32(std::uint32_t value);

	// filename for mnist data set
	std::string test_img_fname;
	std::string test_lbl_fname;
	std::string train_img_fname;
	std::string train_lbl_fname;

	friend std::ostream& operator<<(std::ostream& oss, const MNISTParser& t);

	// vector for store test and train samples
	std::vector<Sample*> test_sample;
	std::vector<Sample*> train_sample;

	bool is_train;
};

#endif /* MNISTPARSER_H_ */
