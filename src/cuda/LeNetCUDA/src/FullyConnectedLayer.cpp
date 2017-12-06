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

void FullyConnectedLayer::gradient_checker(DeviceVector<float>& original_b, DeviceVector<float_t>& original_w, DeviceVector<float_t>& original_input) {
	this->input_.pop();
	this->g_.pop();
	this->next->g_.pop();
	this->deltaW_.pop();
	this->W_.pop();
	this->b_.pop();
	this->v_output.pop();
	int input_size = this->input_.size();

	int in_depth_ = this->in_depth_;
	int out_depth_ = this->out_depth_;

	//create 2 vectors which are copies of original_w
	DeviceVector<float_t> J_plus(original_w);
	DeviceVector<float_t> J_minus(original_w);

	//only allocate a std::vector with original_w.size
	std::vector<float_t> grad_approx(original_w.size());

	//put theta vector in host
	original_w.pop();
	for (int i = 0; i < original_w.size(); i++) {
		//-----------------------
		//theta minus calculation
		DeviceVector<float_t> theta_plus(original_w);

		theta_plus.pop();

		theta_plus[i] = original_w[i] + EPSILON;

		theta_plus.push();

		call_forward_fully_connected(J_plus.data(), original_input.data(), original_b.data(),
				theta_plus.data(), out_depth_, in_depth_, input_size);

		//-----------------------
		//theta minus calculation
		DeviceVector<float_t> theta_minus(original_w);
		theta_minus.pop();

		theta_minus[i] = original_w[i] - EPSILON;

		theta_minus.push();

		call_forward_fully_connected(J_minus.data(), riginal_input.data(), original_b.data(),
				theta_minus.data(), out_depth_, in_depth_, input_size);

		J_plus.pop();
		J_minus.pop();
		grad_approx[i] = (J_plus[i] - J_minus[i]) / (2 * EPSILON);
	}

	this->deltaW_.pop();
	std::vector<float_t> grad_diff(original_w.size());

	//calc norms

	double norm_approx = this->vector_norm<std::vector<float_t> >(grad_approx);

	double norm_grad = this->vector_norm<DeviceVector<float_t> >(this->deltaW_);

	this->g_.pop();
	for (int i = 0; i < grad_diff.size(); i++) {
		grad_diff[i] = this->deltaW_[i] - grad_approx[i];
//		std::cout << "delta " << this->deltaW_[i] << " approx " << grad_approx[i] << "\n";
	}

	double numerator = this->vector_norm<std::vector<float_t> >(grad_diff);
	double difference = numerator / (norm_grad + norm_approx);
	if (difference > MAX_ERROR_ALLOWED) {
		std::cout
		<< "There is a mistake in the backward propagation! difference ="
		<< difference << "\n";
	} else {
		std::cout
		<< "Your backward propagation works perfectly fine! difference ="
		<< difference << "\n";
	}
}

//#ifdef TRAINGPU
void FullyConnectedLayer::back_prop() {
	float *input_ = this->input_.data();
	float *g_ = this->g_.data();
	float *g_next = this->next->g_.data();
	int g_next_size = this->next->g_.size();
	float *deltaW_ = this->deltaW_.data();
	float *W_ = this->W_.data();
	float *b_ = this->b_.data();
	float *r_output = this->v_output.data();

	float alpha_ = this->alpha_;
	float lambda_ = this->lambda_;
	int in_depth_ = this->in_depth_;
	int out_depth_ = this->out_depth_;

	DeviceVector<float_t> copy_b_(this->b_);
	DeviceVector<float_t> copy_W_(this->W_);
	DeviceVector<float_t> copy_input(this->input_);

	call_backpropagation_fully_connected(input_, g_, g_next,
			deltaW_, W_, b_, r_output,
			alpha_, lambda_, in_depth_, out_depth_, g_next_size);

	gradient_checker(copy_b_, copy_W_, copy_input);
}
//#endif //TRAINGPU

#else

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

//FULL CPU BACKPROPAGATION
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

#endif

//#ifndef TRAINGPU
//void FullyConnectedLayer::back_prop() {
//	/*
//	 Compute the err terms;
//	 */
//#ifdef NOTUNIFIEDMEMORY
//	this->W_.pop();
//	this->g_.pop();
//	this->input_.pop();
//	this->next->g_.pop();
//	this->input_.pop();
//	this->deltaW_.pop();
//	this->b_.pop();
//#endif
//	for (size_t in = 0; in < in_depth_; in++) {
//		g_[in] = df_sigmod(input_[in]) * dot(this->next->g_, get_W_step(in));
//	}
//
//	/*
//	 Update weights.
//	 */
//	for (size_t out = 0; out < out_depth_; out++) {
//		for (size_t in = 0; in < in_depth_; in++) {
//			auto delta = alpha_/*learning rate*/
//			* input_[in] * this->next->g_[out]/*err terms*/
//			/*+ lambda_ weight decay*/
//			+ lambda_ * deltaW_[out * in_depth_ + in];
//
//			W_[out * in_depth_ + in] += delta;
//			deltaW_[out * in_depth_ + in] = delta;
//		}
//		b_[out] += alpha_ * this->next->g_[out];
//	}
//#ifdef NOTUNIFIEDMEMORY
//	this->g_.push();
//	this->b_.push();
//	this->W_.push();
//#endif
//}
//
//#endif //TRAINGPU

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
	this->layer_type = "fullyconnected";
}

vec_host FullyConnectedLayer::get_W(size_t index) {
	vec_host v(in_depth_);

	for (size_t i = 0; i < in_depth_; i++) {
		v[i] = (W_[index * in_depth_ + i]);
	}
	return v;
}

vec_host FullyConnectedLayer::get_W_step(size_t in) {
	vec_host r;
	for (size_t i = in; i < out_depth_ * in_depth_; i += in_depth_) {
		r.push_back(W_[i]);
	}
	return r;
}

//DeviceVector<float> FullyConnectedLayer::get_W(size_t index) {
//	DeviceVector<float> v(in_depth_);
//#ifdef NOTUNIFIEDMEMORY
//	v.pop();
//#endif
//	for (size_t i = 0; i < in_depth_; i++) {
//		v[i] = (W_[index * in_depth_ + i]);
//	}
//	return v;
//}

//DeviceVector<float> FullyConnectedLayer::get_W_step(size_t in) {
//	DeviceVector<float> r(out_depth_);
//#ifdef NOTUNIFIEDMEMORY
//	r.pop();
//#endif
//	for (size_t i = in; i < out_depth_ * in_depth_; i += in_depth_) {
//		int it = i / in_depth_;
//		r[it] = (W_[i]);
//	}
//	return r;
//}
