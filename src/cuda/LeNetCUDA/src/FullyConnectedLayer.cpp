/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 05/06/2017
 *      Author: fernando
 */

#include "FullyConnectedLayer.h"
#include "Util.h"

#ifdef GPU
#include "FullyConnectedLayerKernel.h"

void FullyConnectedLayer::forward() {
	this->v_output = this->W_;

	float *output_ = this->output_.data();
	float *input_ = this->input_.data();
	float *b_ = this->b_.data();
	float *W_ = this->W_.data();
//	float *v_output = this->v_output.data();
	int out_depth_ = this->out_depth_;
	int in_depth_ = this->in_depth_;
	int input_size = this->input_.size();

	call_forward_fully_connected(output_, input_, b_, W_, out_depth_, in_depth_,input_size);

}

DeviceVector<float>  FullyConnectedLayer::get_W(size_t index) {
	DeviceVector<float> v(in_depth_);
	for (size_t i = 0; i < in_depth_; i++) {
		v[i] = (W_[index * in_depth_ + i]);
	}
	return v;
}

/*
 for the activation sigmod,
 weight init as [-4 * (6 / sqrt(fan_in + fan_out)), +4 *(6 / sqrt(fan_in + fan_out))]:
 see also:http://deeplearning.net/tutorial/references.html#xavier10
 */
void FullyConnectedLayer::init_weight() {
	/*
	 uniform_rand(W_.begin(), W_.end(),
	 -4 * 6 / std::sqrtf((float)(fan_in() + fan_out())),
	 4 * 6 / std::sqrtf((float)(fan_in() + fan_out())));
	 uniform_rand(b_.begin(), b_.end(),
	 -4 * 6 / std::sqrtf((float)(fan_in() + fan_out())),
	 4 * 6 / std::sqrtf((float)(fan_in() + fan_out())));
	 */
	vec_host temp_W_, temp_b_;
	temp_W_.resize(in_depth_ * out_depth_);
	temp_b_.resize(out_depth_);
	uniform_rand(temp_W_.begin(), temp_W_.end(), -2, 2);
	uniform_rand(temp_b_.begin(), temp_b_.end(), -2, 2);

	this->W_ = temp_W_;
	this->b_ = temp_b_;

//	this->v_output.resize(this->in_depth_ * this->out_depth_);
}

DeviceVector<float> FullyConnectedLayer::get_W_step(size_t in) {
	DeviceVector<float> r(out_depth_);
	for (size_t i = in; i < out_depth_ * in_depth_; i += in_depth_) {
		int it = i / in_depth_;
		r[it] = (W_[i]);
	}
	return r;
}


#else

vec_host FullyConnectedLayer::get_W(size_t index) {
	vec_host v(in_depth_);
	for (size_t i = 0; i < in_depth_; i++) {
		v[i] = (W_[index * in_depth_ + i]);
	}
	return v;
}



void FullyConnectedLayer::forward() {
	for (size_t out = 0; out < out_depth_; out++) {
		output_[out] = sigmod(dot(input_, get_W(out)) + b_[out]);
	}
}


/*
 for the activation sigmod,
 weight init as [-4 * (6 / sqrt(fan_in + fan_out)), +4 *(6 / sqrt(fan_in + fan_out))]:
 see also:http://deeplearning.net/tutorial/references.html#xavier10
 */
void FullyConnectedLayer::init_weight() {
	/*
	 uniform_rand(W_.begin(), W_.end(),
	 -4 * 6 / std::sqrtf((float)(fan_in() + fan_out())),
	 4 * 6 / std::sqrtf((float)(fan_in() + fan_out())));
	 uniform_rand(b_.begin(), b_.end(),
	 -4 * 6 / std::sqrtf((float)(fan_in() + fan_out())),
	 4 * 6 / std::sqrtf((float)(fan_in() + fan_out())));
	 */
	uniform_rand(W_.begin(), W_.end(), -2, 2);
	uniform_rand(b_.begin(), b_.end(), -2, 2);

}

vec_host FullyConnectedLayer::get_W_step(size_t in) {
	vec_host r;
	for (size_t i = in; i < out_depth_ * in_depth_; i += in_depth_) {
		r.push_back(W_[i]);
	}
	return r;
}


#endif


void FullyConnectedLayer::back_prop() {
	/*
	 Compute the err terms;
	 */
	for (size_t in = 0; in < in_depth_; in++) {
		g_[in] = df_sigmod(input_[in]) * dot(this->next->g_, get_W_step(in));
	}


	/*
	 Update weights.
	 */
	for (size_t out = 0; out < out_depth_; out++) {
		for (size_t in = 0; in < in_depth_; in++) {
			auto delta = alpha_/*learning rate*/
			* input_[in] * this->next->g_[out]/*err terms*/
			/*+ lambda_ weight decay*/
			+ lambda_ * deltaW_[out * in_depth_ + in];

			W_[out * in_depth_ + in] += delta;
			deltaW_[out * in_depth_ + in] = delta;
		}
		b_[out] += alpha_ * this->next->g_[out];
	}
}




void FullyConnectedLayer::save_layer(FILE *of) {
	this->save_base_layer(of);
}

void FullyConnectedLayer::load_layer(FILE *in) {
	this->load_base_layer(in);
}

FullyConnectedLayer::FullyConnectedLayer(size_t in_depth, size_t out_depth) :
		Layer(1, 1, in_depth, 1, 1, out_depth, 0.3, 0.01) {
	output_.resize(out_depth_);
	W_.resize(in_depth_ * out_depth_);
	deltaW_.resize(in_depth_ * out_depth_);
	b_.resize(out_depth_);
	g_.resize(in_depth_);


	this->init_weight();

}


//void FullyConnectedLayer::back_prop() {
//	float *input_ = this->input_.data();
//	float *g_ = this->g_.data();
//	float *g_next = this->next->g_.data();
//	int g_next_size = this->next->g_.size();
//	float *deltaW_ = this->deltaW_.data();
//	float *W_ = this->W_.data();
//	float *b_ = this->b_.data();
//	float *r_output = this->v_output.data();
//	float alpha_ = this->alpha_;
//	float lambda_ = this->lambda_;
//	int in_depth_ = this->in_depth_;
//	int out_depth_ = this->out_depth_;
//
//	call_backpropagation_fully_connected(input_, g_, g_next,
//			deltaW_, W_, b_, r_output,
//			alpha_, lambda_, in_depth_, out_depth_, g_next_size);
//
//			printf("---------\n");
//
//			printf("deltaW_gpu = [");
//			for (int i = 0; i < this->deltaW_.size(); i++) {
//				printf("%f, ", this->deltaW_[i]);
//			}
//			printf("]\n");
//
//			printf("W_gpu = [");
//			for (int i = 0; i < this->W_.size(); i++) {
//				printf("%f, ", this->W_[i]);
//			}
//			printf("]\n");
//
//			printf("b_gpu = [ ");
//			for (int i = 0; i < this->b_.size(); i++) {
//				printf("%f, ", this->b_[i]);
//			}
//			printf("]\n");
////			exit(-1);
//}
