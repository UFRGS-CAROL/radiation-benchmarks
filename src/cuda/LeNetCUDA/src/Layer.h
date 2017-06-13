/*
 * Layer.h
 *
 *  Created on: May 26, 2017
 *      Author: carol
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include "Util.h"

class Layer {
private:
	virtual void forward_gpu() = 0;
	virtual void forward_cpu() = 0;

public:
	Layer(size_t in_width, size_t in_height, size_t in_depth, size_t out_width,
			size_t out_height, size_t out_depth, float_t alpha, float_t lambda);

	virtual void init_weight() = 0;
	virtual void back_prop() = 0;
	virtual void save_layer(FILE *of) = 0;
	virtual void load_layer(FILE *in) = 0;

	template<typename T> void write_layer_vec(std::vector<T> v, FILE *of) {
		this->write_layer_var<size_t>(v.size(), of);
		fwrite(v.data(), sizeof(T), v.size(), of);
	}

	template<typename T> void write_layer_var(T var, FILE *of) {
		fwrite(&var, sizeof(T), 1, of);
	}

	template<typename T> std::vector<T> load_layer_vec(FILE *in) {
		size_t siz = this->load_layer_var<size_t>(in);

		std::vector<T> v(siz);
		fread(v.data(), sizeof(T), siz, in);
		return v;
	}

	template<typename T> T load_layer_var(FILE *in) {
		T buf;
		fread(&buf, sizeof(T), 1, in);
		return buf;
	}

	void save_base_layer(FILE *of);
	void load_base_layer(FILE *in);

	void forward();

	float_t sigmod(float_t in);

	float_t df_sigmod(float_t f_x);

	size_t fan_in();

	size_t fan_out();

	size_t in_width_;
	size_t in_height_;
	size_t in_depth_;

	size_t out_width_;
	size_t out_height_;
	size_t out_depth_;

	vec_host W_;
	vec_host b_;

	vec_host deltaW_;

	vec_host input_;
	vec_host output_;

	Layer* next;

	float_t alpha_; // learning rate
	float_t lambda_; // momentum
	vec_host g_; // err terms

	/*output*/
	float_t err;
	int exp_y;
	vec_host exp_y_vec;

#ifdef GPU
	vec_t_gpu input_buf;
	vec_t_gpu weight_buf;
	vec_t_gpu output_buf;
	vec_t_gpu b_buf;

	float *get_raw_vector(vec_t_gpu th);
#endif
};

//} /* namespace convnet */

#endif /* LAYER_H_ */
