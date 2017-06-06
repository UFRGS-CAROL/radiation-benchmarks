/*
 * ConvolutionalLayer.cpp
 *
 *  Created on: Jun 5, 2017
 *      Author: carol
 */

#include "ConvolutionalLayer.h"


ConvolutionalLayer::ConvolutionalLayer(size_t in_width, size_t in_height,
		size_t in_depth, size_t kernel_size, size_t out_depth) :
		Layer(in_width, in_height, in_depth, in_width - kernel_size + 1,
				in_height - kernel_size + 1, out_depth, 0.3, 0.01), kernel_size_(
				kernel_size) {

	this->W_.resize(
			kernel_size * kernel_size * this->in_depth_ * this->out_depth_);
	this->deltaW_.resize(
			kernel_size * kernel_size * this->in_depth_ * this->out_depth_);
	this->b_.resize(out_depth * this->out_width_ * this->out_height_);
	this->output_.resize(out_depth * this->out_width_ * this->out_height_);
	init_weight();
	this->init_cuda();
}

void ConvolutionalLayer::init_weight() {
	uniform_rand(W_.begin(), W_.end(), -1, 1);
	uniform_rand(b_.begin(), b_.end(), -1, 1);
}

void ConvolutionalLayer::init_cuda() {
//      Vamos fazer meio que igual a darknet
//      alloca tudo antes de come��ar tanto na GPU quanto na CPU e depois d�� free
//      em tudo
	// Allocate memory on the device
//      cl::Buffer input_buf(context, CL_MEM_READ_ONLY,
//              in_width_ * in_height_ * in_depth_ * sizeof(cl_float));
//      cl::Buffer weight_buf(context, CL_MEM_READ_ONLY,
//              kernel_size_ * kernel_size_ * in_depth_ * out_depth_
//                      * sizeof(cl_float));
//      cl::Buffer b_buf(context, CL_MEM_READ_ONLY,
//              out_depth_ * out_width_ * out_height_ * sizeof(cl_float));
//      cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY,
//              out_width_ * out_height_ * out_depth_ * sizeof(cl_float));
//      this->input_buf =  thrust::device_malloc<float>(in_width_ * in_height_ * in_depth_ * sizeof(float));
//      this->weight_buf = thrust::device_malloc<float>(kernel_size_ * kernel_size_ * in_depth_ * out_depth_* sizeof(float));
//      this->b_buf = thrust::device_malloc<float>(out_depth_ * out_width_ * out_height_ * sizeof(float));
//      this->output_buf = thrust::device_malloc<float>(out_width_ * out_height_ * out_depth_ * sizeof(cl_float));

}

void ConvolutionalLayer::forward() {
	forward_cpu();
}

void ConvolutionalLayer::forward_cpu() {
	std::fill(output_.begin(), output_.end(), 0);
	for (size_t out = 0; out < out_depth_; out++) { /* for each output feature map */
		for (size_t in = 0; in < in_depth_; in++) { /* for each input feature map */
			for (size_t h_ = 0; h_ < out_height_; h_++) {
				for (size_t w_ = 0; w_ < out_width_; w_++) {
					output_[getOutIndex(out, h_, w_)] += conv(
							getInforKernel(in, h_, w_), getW_(in, out));
				}
			}
		}
		/* use activate function to get output */
		for (size_t h_ = 0; h_ < out_height_; h_++) {
			for (size_t w_ = 0; w_ < out_width_; w_++) {
				output_[getOutIndex(out, h_, w_)] = sigmod(
						output_[getOutIndex(out, h_, w_)]
								+ /*eh?*/b_[getb_(out, h_, w_)]);
			}
		}
	}
}

void ConvolutionalLayer::forward_gpu() {

	try {
//          // Allocate memory on the device
//          cl::Buffer input_buf(context, CL_MEM_READ_ONLY,
//                  in_width_ * in_height_ * in_depth_ * sizeof(cl_float));
//          cl::Buffer weight_buf(context, CL_MEM_READ_ONLY,
//                  kernel_size_ * kernel_size_ * in_depth_ * out_depth_
//                          * sizeof(cl_float));
//          cl::Buffer b_buf(context, CL_MEM_READ_ONLY,
//                  out_depth_ * out_width_ * out_height_ * sizeof(cl_float));
//          cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY,
//                  out_width_ * out_height_ * out_depth_ * sizeof(cl_float));

//          std::string kernel_name = "forward_parallel";
//          cl::Kernel kernel(program, kernel_name.c_str());
//          kernel.setArg < cl::Memory > (0, input_buf);
//          kernel.setArg < cl::Memory > (1, weight_buf);
//          kernel.setArg < cl::Memory > (2, b_buf);
//          kernel.setArg < cl::Memory > (3, output_buf);
//          kernel.setArg<int>(4, in_width_);
//          kernel.setArg<int>(5, in_height_);
//          kernel.setArg<int>(6, in_depth_);
//          kernel.setArg<int>(7, out_width_);
//          kernel.setArg<int>(8, out_height_);
//          kernel.setArg<int>(9, out_depth_);
//          kernel.setArg<int>(10, kernel_size_);

		// transfer source data from the host to the device
//          queue.enqueueWriteBuffer(input_buf, CL_TRUE, 0,
//                  in_width_ * in_height_ * in_depth_ * sizeof(cl_float),
//                  &input_[0]);
		//using Thust we can only assign
		this->input_buf = this->input_;

//          queue.enqueueWriteBuffer(weight_buf, CL_TRUE, 0,
//                  kernel_size_ * kernel_size_ * in_depth_ * out_depth_
//                          * sizeof(cl_float), &W_[0]);

		//PEDRO check if it is necessary to transfer weight again
		this->weight_buf = this->W_;

//          queue.enqueueWriteBuffer(b_buf, CL_TRUE, 0,
//                  out_depth_ * out_width_ * out_height_ * sizeof(cl_float),
//                  &b_[0]);

		this->b_buf = this->b_;

		// execute the code on the device
		//PEDRO CHECK IT
		float *i_buf = thrust::raw_pointer_cast(this->input_buf.data());
		float *w_buf = thrust::raw_pointer_cast(this->weight_buf.data());
		float *b_buf = thrust::raw_pointer_cast(this->b_buf.data());
		float *o_buf = thrust::raw_pointer_cast(this->output_buf.data());

		call_foward_parallel(i_buf, w_buf, b_buf, o_buf, this->in_width_,
				this->in_height_, this->in_depth_, this->out_width_,
				this->out_height_, this->out_depth_, this->kernel_size_);

//          int grpWidth = 20;
//          cl::NDRange global(
//                  jc::closestMultiple(out_depth_ * out_width_, grpWidth),
//                  jc::closestMultiple(out_height_, grpWidth));
//          cl::NDRange local(grpWidth, grpWidth);
//          cl_ulong t = jc::runAndTimeKernel(kernel, queue, global, local);
//          queue.enqueueReadBuffer(output_buf, CL_TRUE, 0,
//                  out_width_ * out_height_ * out_depth_ * sizeof(cl_float),
//                  &output_[0]);

		// transfer destination data from the device to the host
		//CHECK IT
		this->output_ = this->output_buf;

//      } catch (cl::Error& e) {
//          std::cerr << e.what() << ": " << jc::readable_status(e.err());
//          //return 3;
	} catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		//return 2;
	} catch (...) {
		std::cerr << "Unexpected error. Aborting!\n" << std::endl;
		//return 1;
	}

}

void ConvolutionalLayer::forward_batch(int batch_size) {

	try {
		/*
		 Allocate memory on the device
		 cl::Buffer input_buf(context, CL_MEM_READ_ONLY,
		 in_width_ * in_height_ * in_depth_ * sizeof(cl_float));
		 cl::Buffer weight_buf(context, CL_MEM_READ_ONLY,
		 kernel_size_ * kernel_size_ * in_depth_ * out_depth_
		 * sizeof(cl_float));
		 cl::Buffer b_buf(context, CL_MEM_READ_ONLY,
		 out_depth_ * out_width_ * out_height_ * sizeof(cl_float));
		 cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY,
		 out_width_ * out_height_ * out_depth_ * sizeof(cl_float));

		 cl::Buffer input_batch_buf(context, CL_MEM_READ_ONLY,
		 batch_size * in_width_ * in_height_ * in_depth_
		 * sizeof(cl_float));
		 cl::Buffer weight_buf(context, CL_MEM_READ_ONLY,
		 kernel_size_ * kernel_size_ * in_depth_ * out_depth_
		 * sizeof(cl_float));
		 cl::Buffer b_buf(context, CL_MEM_READ_ONLY,
		 out_depth_ * out_width_ * out_height_ * sizeof(cl_float));
		 cl::Buffer output_batch_buf(context, CL_MEM_WRITE_ONLY,
		 batch_size * out_width_ * out_height_ * out_depth_
		 * sizeof(cl_float));

		 #ifdef BATCH_MORE
		 std::string kernel_name = "forward_batch_more";
		 #else
		 std::string kernel_name = "forward_batch";
		 #endif
		 cl::Kernel kernel(program, kernel_name.c_str());
		 kernel.setArg < cl::Memory > (0, input_batch_buf);
		 kernel.setArg < cl::Memory > (1, weight_buf);
		 kernel.setArg < cl::Memory > (2, b_buf);
		 kernel.setArg < cl::Memory > (3, output_batch_buf);
		 kernel.setArg<int>(4, in_width_);
		 kernel.setArg<int>(5, in_height_);
		 kernel.setArg<int>(6, in_depth_);
		 kernel.setArg<int>(7, out_width_);
		 kernel.setArg<int>(8, out_height_);
		 kernel.setArg<int>(9, out_depth_);
		 kernel.setArg<int>(10, kernel_size_);
		 kernel.setArg<int>(11, batch_size);

		 queue.enqueueWriteBuffer(input_batch_buf, CL_TRUE, 0,
		 batch_size * in_width_ * in_height_ * in_depth_
		 * sizeof(cl_float), &input_batch_[0]);
		 queue.enqueueWriteBuffer(weight_buf, CL_TRUE, 0,
		 kernel_size_ * kernel_size_ * in_depth_ * out_depth_
		 * sizeof(cl_float), &W_[0]);
		 queue.enqueueWriteBuffer(b_buf, CL_TRUE, 0,
		 out_depth_ * out_width_ * out_height_ * sizeof(cl_float),
		 &b_[0]);
		 */

		// transfer source data from the host to the device
		//PEDRO CHECK it
		this->input_buf = this->input_batch_;
		this->weight_buf = this->W_;
		this->b_buf = this->b_;

		// execute the code on the device
		int grpWidth = 20;

		//PEDRO TEM QUE VER COMO CHAMAR O KERNEL AQUI

		int global_width = closestMultiple(out_depth_ * out_width_, grpWidth);
#ifdef BATCH_MORE
		int global_height = closestMultiple(batch_size/out_height_, grpWidth);
#else
		int global_height = closestMultiple(batch_size * out_height_, grpWidth);
#endif
//          cl::NDRange global(global_width, global_height);
//          cl::NDRange local(grpWidth, grpWidth);

#ifndef PROFILING
//          jc::runAndTimeKernel(kernel, queue, global, local);
#else
		int iteration = 100;
		int input_data_size = (batch_size*in_width_*in_height_*in_depth_
				+ kernel_size_*kernel_size_*in_depth_*out_depth_
				+ batch_size*out_depth_ * out_width_* out_height_)*sizeof(cl_float);
		int output_data_size = batch_size*out_width_*out_height_*out_depth_*sizeof(cl_float);
#ifdef BATCH_MORE
		printf(" **** In ConvolutionalLayer::forward_batch_more ****\n");
//          int memory_access_per_thread = (in_depth_*kernel_size_*kernel_size_*(1+THREAD_TASKS) + THREAD_TASKS)*sizeof(float);
		int memory_access_per_thread = (in_depth_*kernel_size_*kernel_size_*2 + 1)*sizeof(float);
		int operations = in_depth_*kernel_size_*kernel_size_*9
		+ in_depth_*kernel_size_*kernel_size_*15 + 20;
		printf("    Batch size: %d, Tasks of each thread: %d\n    INPUT depth: %d, height: %d, width: %d\n    OUTPUT depth: %d, height: %d, width: %d\n",
				batch_size, 1, in_depth_, in_height_, in_width_, out_depth_, out_height_, out_width_);
#else
		printf(" **** In ConvolutionalLayer::forward_batch ****\n");
		int memory_access_per_thread = (in_depth_ * 2 * kernel_size_*kernel_size_ + 1 + 1)*sizeof(float);
		int operations = 22 + 26 * in_depth_*kernel_size_*kernel_size_;
		printf("    Batch size: %d\n    INPUT depth: %d, height: %d, width: %d\n    OUTPUT depth: %d, height: %d, width: %d\n",
				batch_size, in_depth_, in_height_, in_width_, out_depth_, out_height_, out_width_);
#endif

		printf("    ==Running with>>> %d <<<Iterations==\n", iteration);

		cl_ulong t = 0; // time in nanosecond, 1e-9 second
		for (int i = 0; i < iteration; i++) {
			t += runAndTimeKernel(kernel, queue, global, local);
		}
		const float each_lasts = float(t) / iteration; // nano seconds
		std::cout << "    Time consumed for each iteration: " << each_lasts / 1e6 << " ms" << std::endl;
		std::cout << "    Time consumed for each batch: " << each_lasts / batch_size / 1e6 << " ms" << std::endl;
		float cpI = float(operations) / memory_access_per_thread;
		float peak_bandwidth = 25.6;// Memory Bandwidth: 25.6 GB/s
#ifdef BATCH_MORE
		float throughPut = memory_access_per_thread * batch_size*out_depth_*out_width_*out_height_ / each_lasts; // GB/s
		long long int all_ops = operations*out_depth_*out_width_*out_height_*batch_size;
#else
		float throughPut = memory_access_per_thread * batch_size*out_depth_*out_width_*out_height_ / each_lasts; // GB/s
		long long int all_ops = operations*out_depth_*out_width_*out_height_*batch_size;
#endif
		printf("    Input Buffer size: %.2g MB, Output Buffer size: %.2g MB\n", input_data_size / 1e6, output_data_size / 1e6);
		printf("    CI: %.2g, ThoughPut: %.3g GB/s, Ops/Time= %.3g GFLOPS, CI*Bandwidth= %.3g GFLOPS\n",
				cpI, throughPut, all_ops/each_lasts, cpI*peak_bandwidth);
#endif
		output_batch_.resize(
				batch_size * out_depth_ * out_width_ * out_height_);
		// transfer destination data from the device to the host
//          queue.enqueueReadBuffer(output_batch_buf, CL_TRUE, 0,
//                  batch_size * out_width_ * out_height_ * out_depth_
//                          * sizeof(cl_float), &output_batch_[0]);

		this->output_batch_ = this->output_bach_buf;

//      } catch (cl::Error& e) {
//          std::cerr << e.what() << ": " << jc::readable_status(e.err());
//          //return 3;
	} catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		//return 2;
	} catch (...) {
		std::cerr << "Unexpected error. Aborting!\n" << std::endl;
		//return 1;
	}

}

void ConvolutionalLayer::back_prop() {
	g_.clear();
	g_.resize(in_width_ * in_height_ * in_depth_);
	/*update err terms of this layer.*/
	for (size_t out = 0; out < out_depth_; out++) {
		for (size_t in = 0; in < in_depth_; in++) {
			for (size_t w_ = 0; w_ < out_width_; w_++) {
				for (size_t h_ = 0; h_ < out_height_; h_++) {
					for (size_t y_ = 0; y_ < kernel_size_; y_++) {
						for (size_t x_ = 0; x_ < kernel_size_; x_++) {
							auto ff = in * in_width_ * in_height_
									+ (h_ + y_) * in_width_ + (x_ + w_);
							g_[ff] +=
							/*next layer err terms*/
							this->next->g_[out * out_width_ * out_height_
									+ h_ * out_width_ + w_]
									*
									/*weight*/
									W_[in * out_depth_ * kernel_size_
											* kernel_size_
											+ out * kernel_size_ * kernel_size_
											+ kernel_size_
													* (kernel_size_ - y_ - 1)
											+ (kernel_size_ - 1 - x_)] *
									/*df of input*/
									df_sigmod(input_[ff]);
						}
					}
				}
			}
		}
	}

	/*update weight*/
	for (size_t out = 0; out < out_depth_; out++) {
		for (size_t in = 0; in < in_depth_; in++) {
			for (size_t h_ = 0; h_ < out_height_; h_++) {
				for (size_t w_ = 0; w_ < out_height_; w_++) {
					auto tt = getb_(out, h_, w_);
					for (size_t y_ = 0; y_ < kernel_size_; y_++) {
						for (size_t x_ = 0; x_ < kernel_size_; x_++) {
							/*find update pixel*/
							auto target = in * out_depth_ * kernel_size_
									* kernel_size_
									+ out * kernel_size_ * kernel_size_
									+ kernel_size_ * (kernel_size_ - y_ - 1)
									+ (kernel_size_ - 1 - x_);
							/*cal delta*/
							auto delta =
							/*learning rate*/
							alpha_
									*
									/*input*/
									input_[in * in_width_ * in_height_
											+ (h_ + y_) * in_width_ + (x_ + w_)]
									*
									/*next layer err terms*/
									this->next->g_[tt]
							/*weight momentum*/
							+ lambda_ * deltaW_[target];

							W_[target] += delta;
							/*update momentum*/
							deltaW_[target] = delta;
						}
					}
					b_[tt] += alpha_ * this->next->g_[tt];
				}
			}
		}
	}
}

inline size_t ConvolutionalLayer::getOutIndex(size_t out, size_t h_,
		size_t w_) {
	return out * out_height_ * out_width_ + h_ * out_width_ + w_;
}

inline vec_t ConvolutionalLayer::getInforKernel(size_t in, size_t h_,
		size_t w_) {
	vec_t r;
	for (size_t y = 0; y < kernel_size_; y++) {
		for (size_t x = 0; x < kernel_size_; x++) {
			r.push_back(
					input_[in * (in_width_ * in_height_) + (h_ + y) * in_width_
							+ x + w_]);
		}
	}
	return r;
}

inline vec_t ConvolutionalLayer::getW_(size_t in, size_t out) {
	vec_t r;
	for (size_t i = 0; i < kernel_size_ * kernel_size_; i++)
		r.push_back(
				W_[in * out_depth_ * kernel_size_ * kernel_size_
						+ out * kernel_size_ * kernel_size_ + i]);
	return r;
}

inline int ConvolutionalLayer::getb_(size_t out, size_t h_, size_t w_) {
	return out * out_width_ * out_height_ + h_ * out_height_ + w_;
}

/*
 2-dimension convoluton:

 1 2 3                    1 -1 0
 3 4 2  conv with kernel  -1 0 1
 2 1 3                    1  1 0

 ---->
 1*0 + 2*1 + 3*1 + 3*1 + 4*0 + 2*-1 + 2*0 + 1*-1 + 3*1
 return the sum.

 see also:
 */
float_t ConvolutionalLayer::conv(vec_t a, vec_t b) {
	assert(a.size() == b.size());
	float_t sum = 0, size = a.size();
	for (size_t i = 0; i < size; i++) {
		sum += a[i] * b[size - i - 1];
	}
	return sum;
}


