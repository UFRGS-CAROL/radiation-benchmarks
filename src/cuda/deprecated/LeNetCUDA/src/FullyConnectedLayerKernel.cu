/*
 * FullyConnectedLayerKernel.cu
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#include "FullyConnectedLayer.h"
#include "cudaUtil.h"

__device__ float sigmod_gpu_fully(float in) {
	return 1.0 / (1.0 + exp(-in));
}

__device__ float df_sigmod_gpu_fully(float f_x) {
	return f_x * (1.0 - f_x);
}

__device__ float dot_gpu_fully(float *x, int x_size, float *w) {
//	assert(x.size() == w.size());
	double sum = 0;
#pragma unroll
	for (int i = 0; i < x_size; i++) {
		sum += x[i] * w[i];
	}
	return sum;
}

__global__ void forward_gpu_kernel(float *output_, float *input_, float *b_,
		float *W_, int out_depth_, int in_depth_, int input_size) {

	int out = blockIdx.x * blockDim.x + threadIdx.x;

	if (out > out_depth_)
		return;

//	 original for was like this for (size_t out = 0; out < out_depth_; out++)
	float *v = &W_[out * in_depth_];
	float dot = dot_gpu_fully(input_, input_size, v);

	output_[out] = sigmod_gpu_fully(dot + b_[out]);
}

void call_forward_fully_connected(float *output_,
		float *input_, float *b_, float *W_, int out_depth_, int in_depth_,
		int input_size) {

	dim3 blocks, threads;
	cuda_gridsize(&threads, &blocks, out_depth_);
	forward_gpu_kernel<<<blocks, threads>>>(output_, input_, b_, W_, out_depth_,
			in_depth_, input_size);
	CudaCheckError();
}

__device__ void get_W_step(float *r_output, float *W_, int in, int out_depth_,
		int in_depth_) {
//	vec_host r;
//	for (size_t i = in; i < out_depth_ * in_depth_; i += in_depth_) {
//		r.push_back(W_[i]);
//	}
//	return r;
	for (int i = in, j = 0; i < out_depth_ * in_depth_; i += in_depth_, j++) {
		r_output[j] = W_[i];
	}
}

__global__ void backpropagation_gpu_err_terms(float *g_, float *g_next,
		float *input_, float *r_output, float *W_, int out_depth_,
		int in_depth_, int g_next_size) {
	/*
	 Compute the err terms;
	 */
	int in = blockIdx.x * blockDim.x + threadIdx.x;
	if (in_depth_ < in)
		return;

//	for (size_t in = 0; in < in_depth_; in++) {
	int r_index = out_depth_ * in;
	get_W_step(&r_output[r_index], W_, in, out_depth_, in_depth_);
	float dot_result = dot_gpu_fully(g_next, g_next_size, &r_output[r_index]);
	g_[in] = df_sigmod_gpu_fully(input_[in]) * dot_result;
//	}

}

__global__ void backpropagation_gpu_update_weights(float *input_, float *g_next,
		float *deltaW_, float *W_, float *b_, float alpha_, float lambda_,
		int in_depth_, int out_depth_) {
	/*
	 Update weights.
	 */
	int out = blockIdx.x * blockDim.x + threadIdx.x;
	if (out > out_depth_)
		return;

//	for (size_t out = 0; out < out_depth_; out++) {
	for (int in = 0; in < in_depth_; in++) {
		auto delta = alpha_/*learning rate*/
		* input_[in] * g_next[out]/*err terms*/
		/*+ lambda_ weight decay*/
		+ lambda_ * deltaW_[out * in_depth_ + in];
		W_[out * in_depth_ + in] += delta;
		deltaW_[out * in_depth_ + in] = delta;
	}
	__syncthreads();
	atomicAdd(&b_[out], alpha_ * g_next[out]);
//	b_[out] += alpha_ * g_next[out];
//	}
}

void call_backpropagation_fully_connected(float *input_,
		float *g_, float *g_next, float *deltaW_, float *W_, float *b_,
		float *r_output, float alpha_, float lambda_, int in_depth_,
		int out_depth_, int g_next_size) {

	dim3 blocks, threads;
	cuda_gridsize(&threads, &blocks, in_depth_);
	backpropagation_gpu_err_terms<<<blocks, threads>>>(g_, g_next, input_,
			r_output, W_, out_depth_, in_depth_, g_next_size);
	CudaCheckError();

	cuda_gridsize(&threads, &blocks, out_depth_);

	backpropagation_gpu_update_weights<<<blocks, threads>>>(input_, g_next,
			deltaW_, W_, b_, alpha_, lambda_, in_depth_, out_depth_);
	CudaCheckError();
}

void FullyConnectedLayer::forward() {
	this->v_output = this->W_;

	float *output_ = this->output_.d_data();
	float *input_ = this->input_.d_data();
	float *b_ = this->b_.d_data();
	float *W_ = this->W_.d_data();
//	float *v_output = this->v_output.d_data();
	int out_depth_ = this->out_depth_;
	int in_depth_ = this->in_depth_;
	int input_size = this->input_.size();

	call_forward_fully_connected(output_, input_, b_, W_, out_depth_,
			in_depth_, input_size);
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

void FullyConnectedLayer::gradient_checker(DeviceVector<float>& original_b,
		DeviceVector<float_t>& original_w,
		DeviceVector<float_t>& original_input) {
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
	for (size_t i = 0; i < original_w.size(); i++) {
		//-----------------------
		//theta minus calculation
		DeviceVector<float_t> theta_plus(original_w);

		theta_plus.pop();

		theta_plus[i] = original_w[i] + EPSILON;

		theta_plus.push();

		call_forward_fully_connected(J_plus.d_data(), original_input.d_data(),
				original_b.d_data(), theta_plus.d_data(), out_depth_, in_depth_,
				input_size);

		//-----------------------
		//theta minus calculation
		DeviceVector<float_t> theta_minus(original_w);
		theta_minus.pop();

		theta_minus[i] = original_w[i] - EPSILON;

		theta_minus.push();

		call_forward_fully_connected(J_minus.d_data(), original_input.d_data(),
				original_b.d_data(), theta_minus.d_data(), out_depth_,
				in_depth_, input_size);

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
	for (size_t i = 0; i < grad_diff.size(); i++) {
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

void FullyConnectedLayer::back_prop() {
	float *input_ = this->input_.d_data();
	float *g_ = this->g_.d_data();
	float *g_next = this->next->g_.d_data();
	int g_next_size = this->next->g_.size();
	float *deltaW_ = this->deltaW_.d_data();
	float *W_ = this->W_.d_data();
	float *b_ = this->b_.d_data();
	float *r_output = this->v_output.d_data();

	float alpha_ = this->alpha_;
	float lambda_ = this->lambda_;
	int in_depth_ = this->in_depth_;
	int out_depth_ = this->out_depth_;

	DeviceVector<float_t> copy_b_(this->b_);
	DeviceVector<float_t> copy_W_(this->W_);
	DeviceVector<float_t> copy_input(this->input_);

	call_backpropagation_fully_connected(input_, g_, g_next, deltaW_, W_,
			b_, r_output, alpha_, lambda_, in_depth_, out_depth_, g_next_size);

//	gradient_checker(copy_b_, copy_W_, copy_input);
}
