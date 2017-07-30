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

#ifdef GPU
#include "DeviceVector.h"
#endif

class Layer {
public:
	Layer(size_t in_width, size_t in_height, size_t in_depth, size_t out_width,
			size_t out_height, size_t out_depth, float_t alpha, float_t lambda);

	virtual void init_weight() = 0;
	virtual void back_prop() = 0;
	virtual void forward() = 0;

	virtual void save_layer(FILE *of) = 0;
	virtual void load_layer(FILE *in) = 0;
//	float* get_next_input_data_ptr();

#ifdef GPU
	template<typename T> void write_layer_vec(DeviceVector<T> v, FILE *of) {
		this->write_layer_var<size_t>(v.size(), of);
		fwrite(v.data(), sizeof(T), v.size(), of);

//		cudaError_t ret = cudaDeviceSynchronize();
//		CUDA_CHECK_RETURN(ret);
	}

	template<typename T>  DeviceVector<T> load_layer_vec(FILE *in) {
		size_t siz = this->load_layer_var<size_t>(in);

		DeviceVector<T> v(siz);
		fread(v.data(), sizeof(T), siz, in);

//		cudaError_t ret = cudaDeviceSynchronize();
//		CUDA_CHECK_RETURN(ret);
		return v;
	}
#else
	template<typename T> void write_layer_vec(std::vector<T> v, FILE *of) {
		this->write_layer_var<size_t>(v.size(), of);
		fwrite(v.data(), sizeof(T), v.size(), of);
	}

	template<typename T> std::vector<T> load_layer_vec(FILE *in) {
		size_t siz = this->load_layer_var<size_t>(in);

		std::vector<T> v(siz);
		fread(v.data(), sizeof(T), siz, in);
		return v;
	}
#endif

	template<typename T> void write_layer_var(T var, FILE *of) {
		fwrite(&var, sizeof(T), 1, of);
	}

	template<typename T> T load_layer_var(FILE *in) {
		T buf;
		fread(&buf, sizeof(T), 1, in);
		return buf;
	}

	void save_base_layer(FILE *of);
	void load_base_layer(FILE *in);

	float_t sigmod(float_t in);

	float_t df_sigmod(float_t f_x);

	size_t fan_in();

	size_t fan_out();

	float_t getWeightsSum();
	float_t getSquaredWeightsSum();

	virtual void set_sum_LeNet_squared_weights(float_t sum_Lenet_squared_weights);
	virtual void set_sum_LeNet_weights(float_t sum_Lenet_weights);
	
	//weights values debug
	void print_layer_weights(int layer_num);


	virtual void back_prop_L1();
	virtual void back_prop_L2();

	size_t in_width_;
	size_t in_height_;
	size_t in_depth_;

	size_t out_width_;
	size_t out_height_;
	size_t out_depth_;

	Layer* next;

	float_t alpha_; // learning rate
	float_t lambda_; // momentum
	/*output*/
	float_t err;
	int exp_y;


	//it is necessary for GPU implementation
#ifdef GPU
	DeviceVector<float_t> W_;
	DeviceVector<float_t> b_;
	DeviceVector<float_t> deltaW_;
	DeviceVector<float_t> input_;
	DeviceVector<float_t> output_;
	DeviceVector<float_t> g_; // err terms
	DeviceVector<float_t> exp_y_vec;

#else
	vec_host W_;
	vec_host b_;
	vec_host deltaW_;
	vec_host input_;
	vec_host output_;
	vec_host g_; // err terms
	vec_host exp_y_vec;
#endif

};

#endif /* LAYER_H_ */
