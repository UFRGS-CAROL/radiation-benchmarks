#include "local_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

int local_out_height(local_layer l) {
	int h = l.h;
	if (!l.pad)
		h -= l.size;
	else
		h -= 1;
	return h / l.stride + 1;
}

int local_out_width(local_layer l) {
	int w = l.w;
	if (!l.pad)
		w -= l.size;
	else
		w -= 1;
	return w / l.stride + 1;
}

local_layer make_local_layer(int batch, int h, int w, int c, int n, int size,
		int stride, int pad, ACTIVATION activation) {
	int i;
	local_layer l; // = { 0 };
	l.type = LOCAL;

	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.batch = batch;
	l.stride = stride;
	l.size = size;
	l.pad = pad;

	int out_h = local_out_height(l);
	int out_w = local_out_width(l);
	int locations = out_h * out_w;
	l.out_h = out_h;
	l.out_w = out_w;
	l.out_c = n;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = l.w * l.h * l.c;

	l.weights = (real_t*) calloc(c * n * size * size * locations,
			sizeof(real_t));
	l.weight_updates = (real_t*) calloc(c * n * size * size * locations,
			sizeof(real_t));

	l.biases = (real_t*) calloc(l.outputs, sizeof(real_t));
	l.bias_updates = (real_t*) calloc(l.outputs, sizeof(real_t));

	// real_t scale = 1./sqrt(size*size*c);
	real_t scale = real_t(sqrt(2. / (size * size * c)));
	for (i = 0; i < c * n * size * size; ++i)
		l.weights[i] = scale * rand_uniform(real_t(-1), real_t(1));

	l.output = (real_t*) calloc(l.batch * out_h * out_w * n, sizeof(real_t));
	l.delta = (real_t*) calloc(l.batch * out_h * out_w * n, sizeof(real_t));

	l.workspace_size = out_h * out_w * size * size * c;

	l.forward = forward_local_layer;
	l.backward = backward_local_layer;
	l.update = update_local_layer;

#ifdef GPU
	l.forward_gpu = forward_local_layer_gpu;
	l.backward_gpu = backward_local_layer_gpu;
	l.update_gpu = update_local_layer_gpu;

	l.weights_gpu = cuda_make_array(l.weights, c * n * size * size * locations);
	l.weight_updates_gpu = cuda_make_array(l.weight_updates,
			c * n * size * size * locations);

	l.biases_gpu = cuda_make_array(l.biases, l.outputs);
	l.bias_updates_gpu = cuda_make_array(l.bias_updates, l.outputs);

	l.delta_gpu = cuda_make_array(l.delta, l.batch * out_h * out_w * n);
	l.output_gpu = cuda_make_array(l.output, l.batch * out_h * out_w * n);

#endif
	l.activation = activation;

	fprintf(stderr,
			"Local Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n",
			h, w, c, n, out_h, out_w, n);

	return l;
}

void forward_local_layer(const local_layer l, network net) {
	int out_h = local_out_height(l);
	int out_w = local_out_width(l);
	int i, j;
	int locations = out_h * out_w;

	for (i = 0; i < l.batch; ++i) {
		copy_cpu(l.outputs, l.biases, 1, l.output + i * l.outputs, 1);
	}

	for (i = 0; i < l.batch; ++i) {
		real_t *input = net.input + i * l.w * l.h * l.c;
#ifndef GPU
		im2col_cpu(input, l.c, l.h, l.w, l.size, l.stride, l.pad,
				net.workspace);
		real_t *output = l.output + i * l.outputs;
		for (j = 0; j < locations; ++j) {
			real_t *a = l.weights + j * l.size * l.size * l.c * l.n;
			real_t *b = net.workspace + j;
			real_t *c = output + j;

			int m = l.n;
			int n = 1;
			int k = l.size * l.size * l.c;

			gemm(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations);
		}
#endif
	}
	activate_array(l.output, l.outputs * l.batch, l.activation);
}

void backward_local_layer(local_layer l, network net) {
	int i, j;
	int locations = l.out_w * l.out_h;

	gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);

	for (i = 0; i < l.batch; ++i) {
		axpy_cpu(l.outputs, real_t(1), l.delta + i * l.outputs, 1, l.bias_updates, 1);
	}

	for (i = 0; i < l.batch; ++i) {
		real_t *input = net.input + i * l.w * l.h * l.c;
#ifndef GPU
		im2col_cpu(input, l.c, l.h, l.w, l.size, l.stride, l.pad,
				net.workspace);

		for (j = 0; j < locations; ++j) {
			real_t *a = l.delta + i * l.outputs + j;
			real_t *b = net.workspace + j;
			real_t *c = l.weight_updates + j * l.size * l.size * l.c * l.n;
			int m = l.n;
			int n = l.size * l.size * l.c;
			int k = 1;

			gemm(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n);
		}

		if (net.delta) {
			for (j = 0; j < locations; ++j) {
				real_t *a = l.weights + j * l.size * l.size * l.c * l.n;
				real_t *b = l.delta + i * l.outputs + j;
				real_t *c = net.workspace + j;

				int m = l.size * l.size * l.c;
				int n = 1;
				int k = l.n;

				gemm(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations);
			}

			col2im_cpu(net.workspace, l.c, l.h, l.w, l.size, l.stride, l.pad,
					net.delta + i * l.c * l.h * l.w);
		}
#endif
	}
}

void update_local_layer(local_layer l, update_args a) {
	real_t learning_rate = a.learning_rate * l.learning_rate_scale;
	real_t momentum = a.momentum;
	real_t decay = a.decay;
	int batch = a.batch;

	int locations = l.out_w * l.out_h;
	int size = l.size * l.size * l.c * l.n * locations;
	axpy_cpu(l.outputs, real_t(learning_rate / batch), l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.outputs, momentum, l.bias_updates, 1);

	axpy_cpu(size, real_t(-decay * batch), l.weights, 1, l.weight_updates, 1);
	axpy_cpu(size, real_t(learning_rate / batch), l.weight_updates, 1, l.weights, 1);
	scal_cpu(size, momentum, l.weight_updates, 1);
}

#ifdef GPU

void forward_local_layer_gpu(const local_layer l, network net) {
	int out_h = local_out_height(l);
	int out_w = local_out_width(l);
	int i, j;
	int locations = out_h * out_w;

	for (i = 0; i < l.batch; ++i) {
		copy_gpu(l.outputs, l.biases_gpu, 1, l.output_gpu + i * l.outputs, 1);
	}

	for (i = 0; i < l.batch; ++i) {
		real_t_device *input = net.input_gpu + i * l.w * l.h * l.c;
		im2col_gpu(input, l.c, l.h, l.w, l.size, l.stride, l.pad,
				net.workspace);
		real_t_device *output = l.output_gpu + i * l.outputs;
		for (j = 0; j < locations; ++j) {
			real_t_device *a = l.weights_gpu + j * l.size * l.size * l.c * l.n;
			real_t_device *b = net.workspace + j;
			real_t_device *c = output + j;

			int m = l.n;
			int n = 1;
			int k = l.size * l.size * l.c;

			gemm_gpu(0, 0, m, n, k, (1), a, k, b, locations, (1), c, locations);
		}
	}
	activate_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation);
}

void backward_local_layer_gpu(local_layer l, network net) {
	int i, j;
	int locations = l.out_w * l.out_h;

	gradient_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation,
			l.delta_gpu);
	for (i = 0; i < l.batch; ++i) {
		axpy_gpu(l.outputs, (1), l.delta_gpu + i * l.outputs, 1,
				l.bias_updates_gpu, 1);
	}

	for (i = 0; i < l.batch; ++i) {
		real_t_device *input = net.input_gpu + i * l.w * l.h * l.c;
		im2col_gpu(input, l.c, l.h, l.w, l.size, l.stride, l.pad,
				net.workspace);

		for (j = 0; j < locations; ++j) {
			real_t_device *a = l.delta_gpu + i * l.outputs + j;
			real_t_device *b = net.workspace + j;
			real_t_device *c = l.weight_updates_gpu + j * l.size * l.size * l.c * l.n;
			int m = l.n;
			int n = l.size * l.size * l.c;
			int k = 1;

			gemm_gpu(0, 1, m, n, k, (1), a, locations, b, locations, (1), c, n);
		}

		if (net.delta_gpu) {
			for (j = 0; j < locations; ++j) {
				real_t_device *a = l.weights_gpu + j * l.size * l.size * l.c * l.n;
				real_t_device *b = l.delta_gpu + i * l.outputs + j;
				real_t_device *c = net.workspace + j;

				int m = l.size * l.size * l.c;
				int n = 1;
				int k = l.n;

				gemm_gpu(1, 0, m, n, k, (1), a, m, b, locations, (0), c, locations);
			}

			col2im_gpu(net.workspace, l.c, l.h, l.w, l.size, l.stride, l.pad,
					net.delta_gpu + i * l.c * l.h * l.w);
		}
	}
}

void update_local_layer_gpu(local_layer l, update_args a) {
	real_t learning_rate = a.learning_rate * l.learning_rate_scale;
	real_t momentum = a.momentum;
	real_t decay = a.decay;
	int batch = a.batch;

	int locations = l.out_w * l.out_h;
	int size = l.size * l.size * l.c * l.n * locations;
	axpy_gpu(l.outputs, (learning_rate / batch), l.bias_updates_gpu, 1,
			l.biases_gpu, 1);
	scal_gpu(l.outputs, CAST(momentum), l.bias_updates_gpu, 1);

	axpy_gpu(size, (-decay * batch), l.weights_gpu, 1, l.weight_updates_gpu, 1);
	axpy_gpu(size, (learning_rate / batch), l.weight_updates_gpu, 1,
			l.weights_gpu, 1);
	scal_gpu(size, CAST(momentum), l.weight_updates_gpu, 1);
}

void pull_local_layer(local_layer l) {
	int locations = l.out_w * l.out_h;
	int size = l.size * l.size * l.c * l.n * locations;
	cuda_pull_array(l.weights_gpu, l.weights, size);
	cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
}

void push_local_layer(local_layer l) {
	int locations = l.out_w * l.out_h;
	int size = l.size * l.size * l.c * l.n * locations;
	cuda_push_array(l.weights_gpu, l.weights, size);
	cuda_push_array(l.biases_gpu, l.biases, l.outputs);
}
#endif
