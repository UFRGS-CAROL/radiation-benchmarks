#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

extern "C" void forward_deconvolutional_layer_gpu(layer l, network net) {
	int i;

	int m = l.size * l.size * l.n;
	int n = l.h * l.w;
	int k = l.c;

	fill_gpu(l.outputs * l.batch, 0, l.output_gpu, 1, net.st);

	for (i = 0; i < l.batch; ++i) {
		real_t *a = l.weights_gpu;
		real_t *b = net.input_gpu + i * l.c * l.h * l.w;
		real_t *c = net.workspace;

		gemm_gpu(1, 0, m, n, k, 1, a, m, b, n, 0, c, n, net.use_tensor_cores, net.st);

		col2im_gpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride,
				l.pad, l.output_gpu + i * l.outputs, net.st);
	}
	if (l.batch_normalize) {
		forward_batchnorm_layer_gpu(l, net);
	} else {
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n,
				l.out_w * l.out_h, net.st);
	}
	activate_array_gpu(l.output_gpu, l.batch * l.n * l.out_w * l.out_h,
			l.activation, net.st);
}

extern "C" void backward_deconvolutional_layer_gpu(layer l, network net) {
	int i;

	//constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
	gradient_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation,
			l.delta_gpu, net.st);

	if (l.batch_normalize) {
		backward_batchnorm_layer_gpu(l, net);
	} else {
		backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n,
				l.out_w * l.out_h, net.st);
	}

	//if(net.delta_gpu) memset(net.delta_gpu, 0, l.batch*l.h*l.w*l.c*sizeof(real_t));

	for (i = 0; i < l.batch; ++i) {
		int m = l.c;
		int n = l.size * l.size * l.n;
		int k = l.h * l.w;

		real_t *a = net.input_gpu + i * m * k;
		real_t *b = net.workspace;
		real_t *c = l.weight_updates_gpu;

		im2col_gpu(l.delta_gpu + i * l.outputs, l.out_c, l.out_h, l.out_w,
				l.size, l.stride, l.pad, b, net.st);
		gemm_gpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n, net.use_tensor_cores, net.st);

		if (net.delta_gpu) {
			int m = l.c;
			int n = l.h * l.w;
			int k = l.size * l.size * l.n;

			real_t *a = l.weights_gpu;
			real_t *b = net.workspace;
			real_t *c = net.delta_gpu + i * n * m;

			gemm_gpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n,
					net.use_tensor_cores, net.st);
		}
	}
}

extern "C" void pull_deconvolutional_layer(layer l) {
	cuda_pull_array(l.weights_gpu, l.weights, l.c * l.n * l.size * l.size);
	cuda_pull_array(l.biases_gpu, l.biases, l.n);
	cuda_pull_array(l.weight_updates_gpu, l.weight_updates,
			l.c * l.n * l.size * l.size);
	cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
	if (l.batch_normalize) {
		cuda_pull_array(l.scales_gpu, l.scales, l.n);
		cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
		cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
}

extern "C" void push_deconvolutional_layer(layer l) {
	cuda_push_array(l.weights_gpu, l.weights, l.c * l.n * l.size * l.size);
	cuda_push_array(l.biases_gpu, l.biases, l.n);
	cuda_push_array(l.weight_updates_gpu, l.weight_updates,
			l.c * l.n * l.size * l.size);
	cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
	if (l.batch_normalize) {
		cuda_push_array(l.scales_gpu, l.scales, l.n);
		cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
		cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
}

void update_deconvolutional_layer_gpu(layer l, update_args a,
		cudaStream_t st) {
	real_t learning_rate = a.learning_rate * l.learning_rate_scale;
	real_t momentum = a.momentum;
	real_t decay = a.decay;
	int batch = a.batch;

	if (a.adam) {
		adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu,
				a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch,
				a.t, st);
		adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu,
				l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n,
				batch, a.t, st);
		if (l.scales_gpu) {
			adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu,
					l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n,
					batch, a.t, st);
		}
	} else {
		axpy_gpu(l.nweights, -decay * batch, l.weights_gpu, 1,
				l.weight_updates_gpu, 1, st);
		axpy_gpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1,
				l.weights_gpu, 1, st);
		scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1, st);

		axpy_gpu(l.n, learning_rate / batch, l.bias_updates_gpu, 1,
				l.biases_gpu, 1, st);
		scal_gpu(l.n, momentum, l.bias_updates_gpu, 1, st);

		if (l.scales_gpu) {
			axpy_gpu(l.n, learning_rate / batch, l.scale_updates_gpu, 1,
					l.scales_gpu, 1, st);
			scal_gpu(l.n, momentum, l.scale_updates_gpu, 1,st);
		}
	}
}

