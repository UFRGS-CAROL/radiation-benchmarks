#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"

#include <stdio.h>
#include <time.h>

static size_t get_workspace_size(layer l) {
	return (size_t) l.h * l.w * l.size * l.size * l.n * sizeof(real_t);
}

void bilinear_init(layer l) {
	int i, j, f;
	real_t center = real_t(l.size - 1) / real_t(2.);
	for (f = 0; f < l.n; ++f) {
		for (j = 0; j < l.size; ++j) {
			for (i = 0; i < l.size; ++i) {
				real_t val = real_t((1 - fabs(i - center)) * (1 - fabs(j - center)));
				int c = f % l.c;
				int ind = f * l.size * l.size * l.c + c * l.size * l.size
						+ j * l.size + i;
				l.weights[ind] = val;
			}
		}
	}
}

layer make_deconvolutional_layer(int batch, int h, int w, int c, int n,
		int size, int stride, int padding, ACTIVATION activation,
		int batch_normalize, int adam) {
	int i;
	layer l; // = { 0 };
	l.type = DECONVOLUTIONAL;

	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.batch = batch;
	l.stride = stride;
	l.size = size;

	l.nweights = c * n * size * size;
	l.nbiases = n;

	l.weights = (real_t*) calloc(c * n * size * size, sizeof(real_t));
	l.weight_updates = (real_t*) calloc(c * n * size * size, sizeof(real_t));

	l.biases = (real_t*) calloc(n, sizeof(real_t));
	l.bias_updates = (real_t*) calloc(n, sizeof(real_t));
	//real_t scale = n/(size*size*c);
	//printf("scale: %f\n", scale);
	real_t scale = real_t(.02);
	for (i = 0; i < c * n * size * size; ++i)
		l.weights[i] = scale * rand_normal();
	//bilinear_init(l);
	for (i = 0; i < n; ++i) {
		l.biases[i] = 0;
	}
	l.pad = padding;

	l.out_h = (l.h - 1) * l.stride + l.size - 2 * l.pad;
	l.out_w = (l.w - 1) * l.stride + l.size - 2 * l.pad;
	l.out_c = n;
	l.outputs = l.out_w * l.out_h * l.out_c;
	l.inputs = l.w * l.h * l.c;

	scal_cpu(l.nweights, real_t(l.out_w * l.out_h / (l.w * l.h)), l.weights,
			1);

	l.output = (real_t*) calloc(l.batch * l.outputs, sizeof(real_t));
	l.delta = (real_t*) calloc(l.batch * l.outputs, sizeof(real_t));

	l.forward = forward_deconvolutional_layer;
	l.backward = backward_deconvolutional_layer;
	l.update = update_deconvolutional_layer;

	l.batch_normalize = batch_normalize;

	if (batch_normalize) {
		l.scales = (real_t*) calloc(n, sizeof(real_t));
		l.scale_updates = (real_t*) calloc(n, sizeof(real_t));
		for (i = 0; i < n; ++i) {
			l.scales[i] = 1;
		}

		l.mean = (real_t*) calloc(n, sizeof(real_t));
		l.variance = (real_t*) calloc(n, sizeof(real_t));

		l.mean_delta = (real_t*) calloc(n, sizeof(real_t));
		l.variance_delta = (real_t*) calloc(n, sizeof(real_t));

		l.rolling_mean = (real_t*) calloc(n, sizeof(real_t));
		l.rolling_variance = (real_t*) calloc(n, sizeof(real_t));
		l.x = (real_t*) calloc(l.batch * l.outputs, sizeof(real_t));
		l.x_norm = (real_t*) calloc(l.batch * l.outputs, sizeof(real_t));
	}
	if (adam) {
		l.m = (real_t*) calloc(c * n * size * size, sizeof(real_t));
		l.v = (real_t*) calloc(c * n * size * size, sizeof(real_t));
		l.bias_m = (real_t*) calloc(n, sizeof(real_t));
		l.scale_m = (real_t*) calloc(n, sizeof(real_t));
		l.bias_v = (real_t*) calloc(n, sizeof(real_t));
		l.scale_v = (real_t*) calloc(n, sizeof(real_t));
	}

#ifdef GPU
	l.forward_gpu = forward_deconvolutional_layer_gpu;
	l.backward_gpu = backward_deconvolutional_layer_gpu;
	l.update_gpu = update_deconvolutional_layer_gpu;

	if (gpu_index >= 0) {

		if (adam) {
			l.m_gpu = cuda_make_array(l.m, c * n * size * size);
			l.v_gpu = cuda_make_array(l.v, c * n * size * size);
			l.bias_m_gpu = cuda_make_array(l.bias_m, n);
			l.bias_v_gpu = cuda_make_array(l.bias_v, n);
			l.scale_m_gpu = cuda_make_array(l.scale_m, n);
			l.scale_v_gpu = cuda_make_array(l.scale_v, n);
		}
		l.weights_gpu = cuda_make_array(l.weights, c * n * size * size);
		l.weight_updates_gpu = cuda_make_array(l.weight_updates,
				c * n * size * size);

		l.biases_gpu = cuda_make_array(l.biases, n);
		l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

		l.delta_gpu = cuda_make_array(l.delta, l.batch * l.out_h * l.out_w * n);
		l.output_gpu = cuda_make_array(l.output,
				l.batch * l.out_h * l.out_w * n);

		if (batch_normalize) {
			l.mean_gpu = cuda_make_array(0, n);
			l.variance_gpu = cuda_make_array(0, n);

			l.rolling_mean_gpu = cuda_make_array(0, n);
			l.rolling_variance_gpu = cuda_make_array(0, n);

			l.mean_delta_gpu = cuda_make_array(0, n);
			l.variance_delta_gpu = cuda_make_array(0, n);

			l.scales_gpu = cuda_make_array(l.scales, n);
			l.scale_updates_gpu = cuda_make_array(0, n);

			l.x_gpu = cuda_make_array(0, l.batch * l.out_h * l.out_w * n);
			l.x_norm_gpu = cuda_make_array(0, l.batch * l.out_h * l.out_w * n);
		}
	}
#ifdef CUDNN
	cudnnCreateTensorDescriptor(&l.dstTensorDesc);
	cudnnCreateTensorDescriptor(&l.normTensorDesc);
	cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w);
	cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1);
#endif
#endif

	l.activation = activation;
	l.workspace_size = get_workspace_size(l);

	fprintf(stderr,
			"deconv%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n,
			size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

	return l;
}

void denormalize_deconvolutional_layer(layer l) {
	int i, j;
	for (i = 0; i < l.n; ++i) {
		real_t scale = real_t(l.scales[i] / sqrt(l.rolling_variance[i] + .00001));
		for (j = 0; j < l.c * l.size * l.size; ++j) {
			l.weights[i * l.c * l.size * l.size + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}

void resize_deconvolutional_layer(layer *l, int h, int w) {
	l->h = h;
	l->w = w;
	l->out_h = (l->h - 1) * l->stride + l->size - 2 * l->pad;
	l->out_w = (l->w - 1) * l->stride + l->size - 2 * l->pad;

	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->w * l->h * l->c;

	l->output = (real_t*) realloc(l->output,
			l->batch * l->outputs * sizeof(real_t));
	l->delta = (real_t*) realloc(l->delta,
			l->batch * l->outputs * sizeof(real_t));
	if (l->batch_normalize) {
		l->x = (real_t*) realloc(l->x, l->batch * l->outputs * sizeof(real_t));
		l->x_norm = (real_t*) realloc(l->x_norm,
				l->batch * l->outputs * sizeof(real_t));
	}

#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);

	if (l->batch_normalize) {
		cuda_free(l->x_gpu);
		cuda_free(l->x_norm_gpu);

		l->x_gpu = cuda_make_array(l->output, l->batch * l->outputs);
		l->x_norm_gpu = cuda_make_array(l->output, l->batch * l->outputs);
	}
#ifdef CUDNN
	cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w);
	cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1);
#endif
#endif
	l->workspace_size = get_workspace_size(*l);
}

void forward_deconvolutional_layer(const layer l, network net) {
	int i;

	int m = l.size * l.size * l.n;
	int n = l.h * l.w;
	int k = l.c;

	fill_cpu(l.outputs * l.batch, real_t(0), l.output, 1);

	for (i = 0; i < l.batch; ++i) {
		real_t *a = l.weights;
		real_t *b = net.input + i * l.c * l.h * l.w;
#ifndef GPU
		real_t *c = net.workspace;
		gemm_cpu(1, 0, m, n, k, real_t(1), a, m, b, n, real_t(0), c, n);

		col2im_cpu(net.workspace, l.out_c, l.out_h, l.out_w, l.size, l.stride,
				l.pad, l.output + i * l.outputs);
#endif
	}
	if (l.batch_normalize) {
		forward_batchnorm_layer(l, net);
	} else {
		add_bias(l.output, l.biases, l.batch, l.n, l.out_w * l.out_h);
	}
	activate_array(l.output, l.batch * l.n * l.out_w * l.out_h, l.activation);
}

void backward_deconvolutional_layer(layer l, network net) {
	int i;

	gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);

	if (l.batch_normalize) {
		backward_batchnorm_layer(l, net);
	} else {
		backward_bias(l.bias_updates, l.delta, l.batch, l.n, l.out_w * l.out_h);
	}

	//if(net.delta) memset(net.delta, 0, l.batch*l.h*l.w*l.c*sizeof(real_t));

	for (i = 0; i < l.batch; ++i) {
		int m = l.c;
		int n = l.size * l.size * l.n;
		int k = l.h * l.w;

		real_t *a = net.input + i * m * k;
		real_t *c = l.weight_updates;
#ifndef GPU
		real_t *b = net.workspace;

		im2col_cpu(l.delta + i * l.outputs, l.out_c, l.out_h, l.out_w, l.size,
				l.stride, l.pad, b);
		gemm_cpu(0, 1, m, n, k, real_t(1), a, k, b, k, real_t(1), c, n);
		if (net.delta) {
			int m = l.c;
			int n = l.h * l.w;
			int k = l.size * l.size * l.n;

			real_t *a = l.weights;
			real_t *b = net.workspace;
			real_t *c = net.delta + i * n * m;

			gemm_cpu(0, 0, m, n, k, real_t(1), a, k, b, n, real_t(1), c, n);
		}
#endif

	}
}

void update_deconvolutional_layer(layer l, update_args a) {
	real_t learning_rate = a.learning_rate * l.learning_rate_scale;
	real_t momentum = a.momentum;
	real_t decay = a.decay;
	int batch = a.batch;

	int size = l.size * l.size * l.c * l.n;
	axpy_cpu(l.n, real_t(learning_rate / batch), l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.n, momentum, l.bias_updates, 1);

	if (l.scales) {
		axpy_cpu(l.n, real_t(learning_rate / batch), l.scale_updates, 1, l.scales, 1);
		scal_cpu(l.n, momentum, l.scale_updates, 1);
	}

	axpy_cpu(size, real_t(-decay * batch), l.weights, 1, l.weight_updates, 1);
	axpy_cpu(size, real_t(learning_rate / batch), l.weight_updates, 1, l.weights, 1);
	scal_cpu(size, momentum, l.weight_updates, 1);
}

