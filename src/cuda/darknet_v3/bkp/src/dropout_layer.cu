#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int batch, int inputs, real_t probability) {
	dropout_layer l; // = { 0 };
	l.type = DROPOUT;
	l.probability = probability;
	l.inputs = inputs;
	l.outputs = inputs;
	l.batch = batch;
	l.rand = (real_t*) calloc(inputs * batch, sizeof(real_t));
	l.scale = 1. / (1. - probability);
	l.forward = forward_dropout_layer;
	l.backward = backward_dropout_layer;
#ifdef GPU
	l.forward_gpu = forward_dropout_layer_gpu;
	l.backward_gpu = backward_dropout_layer_gpu;
	l.rand_gpu = cuda_make_array(l.rand, inputs * batch);
#endif
	fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n",
			probability, inputs, inputs);
	return l;
}

void resize_dropout_layer(dropout_layer *l, int inputs) {
	l->rand = (real_t*) realloc(l->rand, l->inputs * l->batch * sizeof(real_t));
#ifdef GPU
	cuda_free(l->rand_gpu);

	l->rand_gpu = cuda_make_array(l->rand, inputs * l->batch);
#endif
}

void forward_dropout_layer(dropout_layer l, network net) {
	int i;
	if (!net.train)
		return;
	for (i = 0; i < l.batch * l.inputs; ++i) {
		real_t r = rand_uniform(real_t(0), real_t(1));
		l.rand[i] = r;
		if (r < l.probability)
			net.input[i] = 0;
		else
			net.input[i] *= l.scale;
	}
}

void backward_dropout_layer(dropout_layer l, network net) {
	int i;
	if (!net.delta)
		return;
	for (i = 0; i < l.batch * l.inputs; ++i) {
		real_t r = l.rand[i];
		if (r < l.probability)
			net.delta[i] = 0;
		else
			net.delta[i] *= l.scale;
	}
}

