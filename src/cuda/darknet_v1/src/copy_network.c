#include "cuda_runtime.h"

#include "network.h"
#include "crop_layer.h"
#include "connected_layer.h"
#include "rnn_layer.h"
#include "gru_layer.h"
#include "crnn_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "deconvolutional_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "cost_layer.h"
#include "local_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "blas.h"

inline void cuda_mem_copy(void *dest, void *src, int size, int kind) {
	cudaMemcpy(dest, src, size, kind);
}

inline void host_mem_copy(void *dest, void *src, int size) {
	memcpy(dest, src, size);
}

void copy_convolutional_layer(layer *dest, layer *src) {
//	int i;
//	convolutional_layer l = { 0 };
	int n = src->n;
	int c = src->c;
	int size = src->size;

//	l.weights = calloc(c * n * size * size, sizeof(float));
//	host_mem_copy(dest->weights, src->weights,
//			c * n * size * size * sizeof(float));

//	l.weight_updates = calloc(c * n * size * size, sizeof(float));
//	host_mem_copy(dest->weight_updates, src->weight_updates,
//			c * n * size * size * sizeof(float));

//	l.biases = calloc(n, sizeof(float));
//	host_mem_copy(dest->biases, src->biases, n * sizeof(float));

//	l.bias_updates = calloc(n, sizeof(float));
//	host_mem_copy(dest->bias_updates, src->bias_updates, n * sizeof(float));

	int out_h = convolutional_out_height(*src);
	int out_w = convolutional_out_width(*src);

	//	l.output = calloc(l.batch * out_h * out_w * n, sizeof(float));
//	host_mem_copy(dest->output, src->output,
//			src->batch * out_h * out_w * n * sizeof(float));

//	l.delta = calloc(l.batch * out_h * out_w * n, sizeof(float));
//	host_mem_copy(dest->delta, src->delta,
//			src->batch * out_h * out_w * n * sizeof(float));

	if (src->binary) {
//		l.binary_weights = calloc(c * n * size * size, sizeof(float));
//		host_mem_copy(dest->binary_weights, src->binary_weights,
//				c * n * size * size * sizeof(float));

//		l.cweights = calloc(c * n * size * size, sizeof(char));
//		host_mem_copy(dest->cweights, src->cweights,
//				c * n * size * size * sizeof(char));

//		l.scales = calloc(n, sizeof(float));
//		host_mem_copy(dest->scales, src->scales, n * sizeof(float));
	}

	if (src->xnor) {
		//l.binary_weights = calloc(c * n * size * size, sizeof(float));
//		host_mem_copy(dest->binary_weights, src->binary_weights,
//				c * n * size * size * sizeof(float));

//l.binary_input = calloc(l.inputs * l.batch, sizeof(float));
//		host_mem_copy(dest->binary_input, src->binary_input,
//				src->inputs * src->batch * sizeof(float));
	}

	if (src->batch_normalize) {
//		l.scales = calloc(n, sizeof(float));
//		host_mem_copy(dest->scales, src->scales, n * sizeof(float));
//		l.scale_updates = calloc(n, sizeof(float));
//		host_mem_copy(dest->scales, src->scales, n * sizeof(float));

//		l.mean = calloc(n, sizeof(float));
//		host_mem_copy(dest->mean, src->mean, n * sizeof(float));
//		l.variance = calloc(n, sizeof(float));
//		host_mem_copy(dest->variance, src->variance, n * sizeof(float));

//		l.rolling_mean = calloc(n, sizeof(float));
//		host_mem_copy(dest->rolling_mean, src->rolling_mean, n * sizeof(float));
//		l.rolling_variance = calloc(n, sizeof(float));
//		host_mem_copy(dest->rolling_variance, src->rolling_variance,
//				n * sizeof(float));
	}

//#ifdef GPU
	if (gpu_index >= 0) {
//		l.weights_gpu = cuda_make_array(l.weights, c * n * size * size);
		cuda_mem_copy(dest->weights_gpu, src->weights_gpu,
				c * n * size * size * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.weight_updates_gpu = cuda_make_array(l.weight_updates,
//				c * n * size * size);
//		cuda_mem_copy(dest->weight_updates_gpu, src->weight_updates_gpu,
//				c * n * size * size * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.biases_gpu = cuda_make_array(l.biases, n);
		cuda_mem_copy(dest->biases_gpu, src->biases_gpu, n * sizeof(float),
				cudaMemcpyDeviceToDevice);

//		l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
//		cuda_mem_copy(dest->bias_updates_gpu, src->bias_updates_gpu,
//				n * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.delta_gpu = cuda_make_array(l.delta, l.batch * out_h * out_w * n);
//		cuda_mem_copy(dest->delta_gpu, src->delta_gpu,
//				src->batch * out_h * out_w * n * sizeof(float),
//				cudaMemcpyDeviceToDevice);

//		l.output_gpu = cuda_make_array(l.output, l.batch * out_h * out_w * n);
		cuda_mem_copy(dest->output_gpu, src->output_gpu,
				src->batch * out_h * out_w * n * sizeof(float),
				cudaMemcpyDeviceToDevice);

		if (src->binary) {
//			l.binary_weights_gpu = cuda_make_array(l.weights,
//					c * n * size * size);
			cuda_mem_copy(dest->binary_weights_gpu, src->binary_weights_gpu,
					c * n * size * size * sizeof(float),
					cudaMemcpyDeviceToDevice);
		}

		if (src->xnor) {
//			l.binary_weights_gpu = cuda_make_array(l.weights,
//					c * n * size * size);
			if (!src->binary)
				cuda_mem_copy(dest->binary_weights_gpu, src->binary_weights_gpu,
						c * n * size * size * sizeof(float),
						cudaMemcpyDeviceToDevice);

//			l.binary_input_gpu = cuda_make_array(0, l.inputs * l.batch);
			cuda_mem_copy(dest->binary_input_gpu, src->binary_input_gpu,
					src->inputs * src->inputs * sizeof(float),
					cudaMemcpyDeviceToDevice);
		}

		if (src->batch_normalize) {
//			l.mean_gpu = cuda_make_array(l.mean, n);
			cuda_mem_copy(dest->mean_gpu, src->mean_gpu, n * sizeof(float),
					cudaMemcpyDeviceToDevice);

//			l.variance_gpu = cuda_make_array(l.variance, n);
			cuda_mem_copy(dest->variance_gpu, src->variance_gpu,
					n * sizeof(float), cudaMemcpyDeviceToDevice);

//			l.rolling_mean_gpu = cuda_make_array(l.mean, n);
			cuda_mem_copy(dest->rolling_mean_gpu, src->rolling_mean_gpu,
					n * sizeof(float), cudaMemcpyDeviceToDevice);

//			l.rolling_variance_gpu = cuda_make_array(l.variance, n);
			cuda_mem_copy(dest->rolling_variance_gpu, src->rolling_variance_gpu,
					n * sizeof(float), cudaMemcpyDeviceToDevice);

//			l.mean_delta_gpu = cuda_make_array(l.mean, n);
			cuda_mem_copy(dest->mean_delta_gpu, src->mean_delta_gpu,
					n * sizeof(float), cudaMemcpyDeviceToDevice);

//			l.variance_delta_gpu = cuda_make_array(l.variance, n);
			cuda_mem_copy(dest->variance_delta_gpu, src->variance_delta_gpu,
					n * sizeof(float), cudaMemcpyDeviceToDevice);

//			l.scales_gpu = cuda_make_array(l.scales, n);
			cuda_mem_copy(dest->scales_gpu, src->scales_gpu, n * sizeof(float),
					cudaMemcpyDeviceToDevice);

//			l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);
			cuda_mem_copy(dest->scales_gpu, src->scales_gpu, n * sizeof(float),
					cudaMemcpyDeviceToDevice);

//			l.x_gpu = cuda_make_array(l.output, l.batch * out_h * out_w * n);
			cuda_mem_copy(dest->x_gpu, src->x_gpu,
					src->batch * out_h * out_w * n * sizeof(float),
					cudaMemcpyDeviceToDevice);

//			l.x_norm_gpu = cuda_make_array(l.output, l.batch * out_h * out_w * n);
			cuda_mem_copy(dest->x_norm_gpu, src->x_norm_gpu,
					src->batch * out_h * out_w * n * sizeof(float),
					cudaMemcpyDeviceToDevice);
		}

	}
//#endif

}

void copy_local_layer(layer *dest, layer *src) {
	int c = src->c;
	int n = src->n;
	int size = src->size;

	int out_h = src->out_h;
	int out_w = src->out_w;
	int locations = out_h * out_w;

//		l.weights = calloc(c * n * size * size * locations, sizeof(float));
//	host_mem_copy(dest->weights, src->weights,
//			c * n * size * locations * sizeof(float));

//		l.weight_updates = calloc(c * n * size * size * locations, sizeof(float));
//	host_mem_copy(dest->weight_updates, src->weight_updates,
//			c * n * size * size * locations * sizeof(float));

//		l.biases = calloc(l.outputs, sizeof(float));
//	host_mem_copy(dest->biases, src->biases, src->outputs * sizeof(float));

//		l.bias_updates = calloc(l.outputs, sizeof(float));
//	host_mem_copy(dest->bias_updates, src->bias_updates,
//			src->outputs * sizeof(float));

// float scale = 1./sqrt(size*size*c);
//		float scale = sqrt(2. / (size * size * c));
//		for (i = 0; i < c * n * size * size; ++i)
//			l.weights[i] = scale * rand_uniform(-1, 1);

//		l.col_image = calloc(out_h * out_w * size * size * c, sizeof(float));
//	host_mem_copy(dest->col_image, src->col_image,
//			out_h * out_w * size * size * c * sizeof(float));

//		l.output = calloc(l.batch * out_h * out_w * n, sizeof(float));
//	host_mem_copy(dest->output, src->output,
//			src->batch * out_h * out_w * n * sizeof(float));

//		l.delta = calloc(l.batch * out_h * out_w * n, sizeof(float));
//	host_mem_copy(dest->delta, src->delta,
//			src->batch * out_h * out_w * n * sizeof(float));

//	#ifdef GPU
//		l.weights_gpu = cuda_make_array(l.weights, c*n*size*size*locations);
	cuda_mem_copy(dest->weights_gpu, src->weights_gpu,
			c * n * size * size * locations * sizeof(float),
			cudaMemcpyDeviceToDevice);

//		l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size*locations);
//	cuda_mem_copy(dest->weight_updates_gpu, src->weight_updates_gpu,
//			c * n * size * size * locations * sizeof(float),
//			cudaMemcpyDeviceToDevice);

//		l.biases_gpu = cuda_make_array(l.biases, l.outputs);
	cuda_mem_copy(dest->biases_gpu, src->biases_gpu,
			src->outputs * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.bias_updates_gpu = cuda_make_array(l.bias_updates, l.outputs);
//	cuda_mem_copy(dest->bias_updates_gpu, src->bias_updates_gpu,
//			src->outputs * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.col_image_gpu = cuda_make_array(l.col_image, out_h*out_w*size*size*c);
	cuda_mem_copy(dest->col_image_gpu, src->col_image_gpu,
			out_h * out_w * size * size * c * sizeof(float),
			cudaMemcpyDeviceToDevice);

//		l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
//	cuda_mem_copy(dest->delta_gpu, src->delta_gpu,
//			src->batch * out_h * out_w * n * sizeof(float),
//			cudaMemcpyDeviceToDevice);

//		l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
	cuda_mem_copy(dest->output_gpu, src->output_gpu,
			src->batch * out_h * out_w * n * sizeof(float),
			cudaMemcpyDeviceToDevice);

//	#endif

}

void copy_connected_layer(layer *dest, layer *src) {
	int inputs = src->inputs;
	int outputs = src->outputs;

//	l.output = calloc(batch * outputs, sizeof(float));
//	host_mem_copy(dest->output, src->output,
//			src->batch * outputs * sizeof(float));

//	l.delta = calloc(batch * outputs, sizeof(float));
//	host_mem_copy(dest->delta, src->delta,
//			src->batch * outputs * sizeof(float));

//	l.weight_updates = calloc(inputs * outputs, sizeof(float));
//	host_mem_copy(dest->weight_updates, src->weight_updates,
//			inputs * outputs * sizeof(float));

//	l.bias_updates = calloc(outputs, sizeof(float));
//	host_mem_copy(dest->bias_updates, src->bias_updates,
//			outputs * sizeof(float));

//	l.weights = calloc(outputs * inputs, sizeof(float));
//	host_mem_copy(dest->weights, src->weights,
//			outputs * inputs * sizeof(float));

//	l.biases = calloc(outputs, sizeof(float));
//	host_mem_copy(dest->biases, src->biases, outputs * sizeof(float));

	if (src->batch_normalize) {
//		l.scales = calloc(outputs, sizeof(float));
//		host_mem_copy(dest->scales, src->scales, outputs * sizeof(float));

//		l.scale_updates = calloc(outputs, sizeof(float));
//		host_mem_copy(dest->scale_updates, src->scale_updates,
//				outputs * sizeof(float));

//		l.mean = calloc(outputs, sizeof(float));
//		host_mem_copy(dest->mean, src->mean, outputs * sizeof(float));

//		l.mean_delta = calloc(outputs, sizeof(float));
//		host_mem_copy(dest->mean_delta, src->mean_delta,
//				outputs * sizeof(float));

//		l.variance = calloc(outputs, sizeof(float));
//		host_mem_copy(dest->variance, src->variance, outputs * sizeof(float));

//		l.variance_delta = calloc(outputs, sizeof(float));
//		host_mem_copy(dest->variance_delta, src->variance_delta,
//				outputs * sizeof(float));

//		l.rolling_mean = calloc(outputs, sizeof(float));
//		host_mem_copy(dest->rolling_mean, src->rolling_mean,
//				outputs * sizeof(float));

//		l.rolling_variance = calloc(outputs, sizeof(float));
//		host_mem_copy(dest->rolling_variance, src->rolling_variance,
//				outputs * sizeof(float));

//		l.x = calloc(batch * outputs, sizeof(float));
//		host_mem_copy(dest->x, src->x, src->batch * outputs * sizeof(float));

//		l.x_norm = calloc(batch * outputs, sizeof(float));
//		host_mem_copy(dest->x_norm, src->x_norm,
//				src->batch * outputs * sizeof(float));
	}

//#ifdef GPU
//	l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
	cuda_mem_copy(dest->weights_gpu, src->weights_gpu,
			outputs * inputs * sizeof(float), cudaMemcpyDeviceToDevice);

//	l.biases_gpu = cuda_make_array(l.biases, outputs);
	cuda_mem_copy(dest->biases_gpu, src->biases_gpu, outputs * sizeof(float),
			cudaMemcpyDeviceToDevice);

//	l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
//	cuda_mem_copy(dest->weight_updates_gpu, src->weight_updates_gpu,
//			outputs * inputs * sizeof(float), cudaMemcpyDeviceToDevice);

//	l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);
//	cuda_mem_copy(dest->bias_updates_gpu, src->bias_updates_gpu,
//			outputs * sizeof(float), cudaMemcpyDeviceToDevice);

//	l.output_gpu = cuda_make_array(l.output, outputs*batch);
	cuda_mem_copy(dest->output_gpu, src->output_gpu,
			outputs * src->batch * sizeof(float), cudaMemcpyDeviceToDevice);

//	l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
//	cuda_mem_copy(dest->delta_gpu, src->delta_gpu,
//			outputs * src->batch * sizeof(float), cudaMemcpyDeviceToDevice);

	if (src->batch_normalize) {
//		l.scales_gpu = cuda_make_array(l.scales, outputs);
		cuda_mem_copy(dest->scales_gpu, src->scales_gpu,
				outputs * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);
//		cuda_mem_copy(dest->scale_updates_gpu, src->scale_updates_gpu,
//				outputs * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.mean_gpu = cuda_make_array(l.mean, outputs);
		cuda_mem_copy(dest->mean_gpu, src->scales_gpu, outputs * sizeof(float),
				cudaMemcpyDeviceToDevice);

//		l.variance_gpu = cuda_make_array(l.variance, outputs);
		cuda_mem_copy(dest->variance_gpu, src->variance_gpu,
				outputs * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
		cuda_mem_copy(dest->rolling_mean_gpu, src->rolling_mean_gpu,
				outputs * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);
		cuda_mem_copy(dest->rolling_variance_gpu, src->rolling_variance_gpu,
				outputs * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
//		cuda_mem_copy(dest->mean_delta_gpu, src->mean_delta_gpu,
//				outputs * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.variance_delta_gpu = cuda_make_array(l.variance, outputs);
//		cuda_mem_copy(dest->variance_delta_gpu, src->variance_delta_gpu,
//				outputs * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.x_gpu = cuda_make_array(l.output, l.batch*outputs);
		cuda_mem_copy(dest->x_gpu, src->x_gpu,
				src->batch * outputs * sizeof(float), cudaMemcpyDeviceToDevice);

//		l.x_norm_gpu = cuda_make_array(l.output, l.batch*outputs);
		cuda_mem_copy(dest->x_norm_gpu, src->x_norm_gpu,
				src->batch * outputs * sizeof(float), cudaMemcpyDeviceToDevice);
	}
//#endif

}

void copy_detection_layer(layer *dest, layer *src) {
	*dest->cost = *src->cost;
//	l.output = calloc(batch * l.outputs, sizeof(float));
//	host_mem_copy(dest->output, src->output,
//			src->batch * src->outputs * sizeof(float));

//	l.delta = calloc(batch * l.outputs, sizeof(float));
//	host_mem_copy(dest->delta, src->delta,
//			src->batch * src->outputs * sizeof(float));

//#ifdef GPU
//	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	cuda_mem_copy(dest->output_gpu, src->output_gpu,
			src->batch * src->outputs * sizeof(float),
			cudaMemcpyDeviceToDevice);

//	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
//	cuda_mem_copy(dest->delta_gpu, src->delta_gpu,
//			src->batch * src->outputs * sizeof(float),
//			cudaMemcpyDeviceToDevice);
//#endif

}

void copy_dropout_layer(layer *dest, layer *src) {
//	int inputs = src->inputs;
//	int batch = src->batch;
//    l.rand = calloc(inputs*batch, sizeof(float));
//	host_mem_copy(dest->rand, src->rand, inputs * batch * sizeof(float));
//    #ifdef GPU
//    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
//	cuda_mem_copy(dest->rand_gpu, src->rand_gpu, inputs * batch * sizeof(float),
//			cudaMemcpyDeviceToDevice);

//	cuda_mem_copy(dest->output_gpu, src->output_gpu, )
//    #endif
}

void copy_maxpool_layer(layer *dest, layer *src) {
	int output_size = src->out_h * src->out_w * src->out_c * src->batch;

	//    l.indexes_gpu = cuda_make_int_array(output_size);
	cuda_mem_copy(dest->indexes_gpu, src->indexes_gpu,
			output_size * sizeof(int), cudaMemcpyDeviceToDevice);
	//    l.output_gpu  = cuda_make_array(l.output, output_size);
	cuda_mem_copy(dest->output_gpu, src->output_gpu, output_size * sizeof(int),
			cudaMemcpyDeviceToDevice);

}

void copy_layer(layer *dest, layer *src) {
	dest->type = src->type;
	dest->activation = src->activation;
	dest->cost_type = src->cost_type;
	dest->batch_normalize = src->batch_normalize;
	dest->shortcut = src->shortcut;
	dest->batch = src->batch;
	dest->forced = src->forced;
	dest->flipped = src->flipped;
	dest->inputs = src->inputs;
	dest->outputs = src->outputs;
	dest->truths = src->truths;
	dest->h = src->h;
	dest->w = src->w;
	dest->c = src->c;
	dest->out_h = src->out_h;
	dest->out_w = src->out_w;
	dest->out_c = src->out_c;
	dest->n = src->n;
	dest->max_boxes = src->max_boxes;
	dest->groups = src->groups;
	dest->size = src->size;
	dest->side = src->side;
	dest->stride = src->stride;
	dest->pad = src->pad;
	dest->sqrt = src->sqrt;
	dest->flip = src->flip;
	dest->index = src->index;
	dest->binary = src->binary;
	dest->xnor = src->xnor;
	dest->steps = src->steps;
	dest->hidden = src->hidden;
	dest->dot = src->dot;
	dest->angle = src->angle;
	dest->jitter = src->jitter;
	dest->saturation = src->saturation;
	dest->exposure = src->exposure;
	dest->shift = src->shift;
	dest->ratio = src->ratio;
	dest->softmax = src->softmax;
	dest->classes = src->classes;
	dest->coords = src->coords;
	dest->background = src->background;
	dest->rescore = src->rescore;
	dest->objectness = src->objectness;
	dest->does_cost = src->does_cost;
	dest->joint = src->joint;
	dest->noadjust = src->noadjust;
	dest->reorg = src->reorg;
	dest->log = src->log;
	dest->alpha = src->alpha;
	dest->beta = src->beta;
	dest->kappa = src->kappa;
	dest->coord_scale = src->coord_scale;
	dest->object_scale = src->object_scale;
	dest->noobject_scale = src->noobject_scale;
	dest->class_scale = src->class_scale;
	dest->random = src->random;
	dest->dontload = src->dontload;
	dest->dontloadscales = src->dontloadscales;
	dest->temperature = src->temperature;
	dest->probability = src->probability;
	dest->scale = src->scale;

	switch (src->type) {
	case CONVOLUTIONAL: {
		copy_convolutional_layer(dest, src);
		break;
	}
	case DECONVOLUTIONAL:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case CONNECTED: {
		copy_connected_layer(dest, src);
	}
		break;
	case MAXPOOL: {
		copy_maxpool_layer(dest, src);
	}
		break;
	case SOFTMAX:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case DETECTION: {
		copy_detection_layer(dest, src);
	}
		break;
	case DROPOUT:
		copy_dropout_layer(dest, src);
		break;
	case CROP:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case ROUTE:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case COST:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case NORMALIZATION:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case AVGPOOL:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case LOCAL: {
		copy_local_layer(dest, src);

	}
		break;
	case SHORTCUT:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case ACTIVE:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case RNN:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case GRU:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case CRNN:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case BATCHNORM:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case NETWORK:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case XNOR:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case REGION:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case REORG:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	case BLANK:
		error("ERROR: LAYER TYPE COPY NOT IMPLEMENTED\n");
		break;
	}

}

void copy_network_state(network_state *dest, network_state *src,
		int start_layer) {
	dest->index = src->index;
	dest->input = dest->net.layers[start_layer].output_gpu;
	dest->train = src->train;
}

void copy_network_content_to_buffer(network *mt_net, network_state *mt_state,
		network *buffer_nets, network_state *buffer_states, int thread_id,
		int mr_size, int start_layer) {

	for (int i = 0; i < mr_size; i++) {
		network *current_net = &buffer_nets[i];
		network_state *current_state = &buffer_states[i];

		//-------------------------------------------------------
		//copy all network content
		current_net->n = mt_net->n;
		current_net->batch = mt_net->batch;

		current_net->epoch = mt_net->epoch;
		current_net->subdivisions = mt_net->subdivisions;
		current_net->momentum = mt_net->momentum;
		current_net->decay = mt_net->decay;
		current_net->outputs = mt_net->outputs;
		current_net->policy = mt_net->policy;

		current_net->learning_rate = mt_net->learning_rate;
		current_net->gamma = mt_net->gamma;
		current_net->scale = mt_net->scale;
		current_net->power = mt_net->power;
		current_net->time_steps = mt_net->time_steps;
		current_net->step = mt_net->step;
		current_net->max_batches = mt_net->max_batches;

		current_net->num_steps = mt_net->num_steps;
		current_net->burn_in = mt_net->burn_in;
		current_net->inputs = mt_net->inputs;
		current_net->h = mt_net->h;
		current_net->w = mt_net->w;
		current_net->c = mt_net->c;
		current_net->max_crop = mt_net->max_crop;
		current_net->min_crop = mt_net->min_crop;
		current_net->angle = mt_net->angle;
		current_net->aspect = mt_net->aspect;
		current_net->exposure = mt_net->exposure;
		current_net->saturation = mt_net->saturation;
		current_net->hue = mt_net->hue;
		current_net->gpu_index = mt_net->gpu_index;

		int workspace_size = 0;

		for (int i = start_layer; i < mt_net->n; i++) {
			layer *l_curr = &current_net->layers[i];
			layer *l_mt = &mt_net->layers[i];
			copy_layer(l_curr, l_mt);
			if (workspace_size < l_mt->workspace_size)
				workspace_size = l_mt->workspace_size;
		}

		int output_size = get_network_output_size(*mt_net);
		int scales_size = mt_net->num_steps;
		int steps_size = mt_net->num_steps;
		int input_gpu_size = get_network_input_size(*mt_net) * mt_net->batch;
		int truth_gpu_size = get_network_output_size(*mt_net) * mt_net->batch;
		if (mt_net->layers[mt_net->n - 1].truths)
			truth_gpu_size = mt_net->layers[mt_net->n - 1].truths
					* mt_net->batch;

//#ifdef GPU
		*current_net->seen = *mt_net->seen;

		cuda_mem_copy(current_net->workspace, mt_net->workspace,
				((workspace_size - 1) / sizeof(float) + 1),
				cudaMemcpyDeviceToDevice);

		host_mem_copy(current_net->output, mt_net->output,
				sizeof(float) * output_size);

		host_mem_copy(current_net->scales, mt_net->scales,
				sizeof(float) * scales_size);

		host_mem_copy(current_net->steps, mt_net->steps,
				sizeof(int) * steps_size);

		//TODO
		// CHECK IF COPY TRUTH_GPU and INPUT_GPU are necessary

//#endif

		copy_network_state(current_state, mt_state, start_layer);
	}

}
