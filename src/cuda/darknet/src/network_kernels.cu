#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "parser.h"

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
}

static network *buffer_nets;
static network_state *buffer_states;

float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float * get_network_output_gpu(network net);

void forward_network_gpu(network net, network_state state) {
	state.workspace = net.workspace;

	for (int i = 0; i < net.n; ++i) {
		state.index = i;
		layer l = net.layers[i];
		if (l.delta_gpu) {
			fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1,
					state.st_handle.stream);
		}
		//printf("Passou ate a layer %d\n", i);
		if (l.type == CONVOLUTIONAL) {
			forward_convolutional_layer_gpu(l, state);
		} else if (l.type == DECONVOLUTIONAL) {
			forward_deconvolutional_layer_gpu(l, state);
		} else if (l.type == ACTIVE) {
			forward_activation_layer_gpu(l, state);
		} else if (l.type == LOCAL) {
			forward_local_layer_gpu(l, state);
		} else if (l.type == DETECTION) {
			forward_detection_layer_gpu(l, state);
		} else if (l.type == REGION) {
			forward_region_layer_gpu(l, state);
		} else if (l.type == CONNECTED) {
			forward_connected_layer_gpu(l, state);
		} else if (l.type == RNN) {
			forward_rnn_layer_gpu(l, state);
		} else if (l.type == GRU) {
			forward_gru_layer_gpu(l, state);
		} else if (l.type == CRNN) {
			forward_crnn_layer_gpu(l, state);
		} else if (l.type == CROP) {
			forward_crop_layer_gpu(l, state);
		} else if (l.type == COST) {
			forward_cost_layer_gpu(l, state);
		} else if (l.type == SOFTMAX) {
			forward_softmax_layer_gpu(l, state);
		} else if (l.type == NORMALIZATION) {
			forward_normalization_layer_gpu(l, state);
		} else if (l.type == BATCHNORM) {
			forward_batchnorm_layer_gpu(l, state);
		} else if (l.type == MAXPOOL) {
			forward_maxpool_layer_gpu(l, state);
		} else if (l.type == REORG) {
			forward_reorg_layer_gpu(l, state);
		} else if (l.type == AVGPOOL) {
			forward_avgpool_layer_gpu(l, state);
		} else if (l.type == DROPOUT) {
			forward_dropout_layer_gpu(l, state);
		} else if (l.type == ROUTE) {
			forward_route_layer_gpu(l, net, state);
		} else if (l.type == SHORTCUT) {
			forward_shortcut_layer_gpu(l, state);
		}
		state.input = l.output_gpu;

		cudaStreamSynchronize(state.st_handle.stream);

	}

}

void init_mr_buffers(int mr_size) {
	buffer_nets = (network*) calloc(mr_size, sizeof(network));
	buffer_states = (network_state*) calloc(mr_size, sizeof(network_state));
}

void destroy_mr_buffers() {
	if (buffer_nets)
		free(buffer_nets);
	if (buffer_states)
		free(buffer_states);
}

void inline copy_layer(layer *dest, layer *src) {
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
	dest->h  = src->h;
	dest->w = src->w;
	dest->c = src->c;
	dest->out_h= src->out_h;
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


	int *indexes;
	float *rand;
	float *cost;
	char *cweights;
	float *state;
	float *prev_state;
	float *forgot_state;
	float *forgot_delta;
	float *state_delta;

	float *concat;
	float *concat_delta;

	float *binary_weights;

	float *biases;
	float *bias_updates;

	float *scales;
	float *scale_updates;

	float *weights;
	float *weight_updates;

	float *col_image;
	int * input_layers;
	int * input_sizes;
	float * delta;
	float * output;
	float * squared;
	float * norms;

	float * spatial_mean;
	float * mean;
	float * variance;

	float * mean_delta;
	float * variance_delta;

	float * rolling_mean;
	float * rolling_variance;

	float * x;
	float * x_norm;

	struct layer *input_layer;
	struct layer *self_layer;
	struct layer *output_layer;

	struct layer *input_gate_layer;
	struct layer *state_gate_layer;
	struct layer *input_save_layer;
	struct layer *state_save_layer;
	struct layer *input_state_layer;
	struct layer *state_state_layer;

	struct layer *input_z_layer;
	struct layer *state_z_layer;

	struct layer *input_r_layer;
	struct layer *state_r_layer;

	struct layer *input_h_layer;
	struct layer *state_h_layer;

	float *z_cpu;
	float *r_cpu;
	float *h_cpu;

	float *binary_input;

	size_t workspace_size;

	float *z_gpu;
	float *r_gpu;
	float *h_gpu;

	int *indexes_gpu;
	float * prev_state_gpu;
	float * forgot_state_gpu;
	float * forgot_delta_gpu;
	float * state_gpu;
	float * state_delta_gpu;
	float * gate_gpu;
	float * gate_delta_gpu;
	float * save_gpu;
	float * save_delta_gpu;
	float * concat_gpu;
	float * concat_delta_gpu;

	float *binary_input_gpu;
	float *binary_weights_gpu;

	float * mean_gpu;
	float * variance_gpu;

	float * rolling_mean_gpu;
	float * rolling_variance_gpu;

	float * variance_delta_gpu;
	float * mean_delta_gpu;

	float * col_image_gpu;

	float * x_gpu;
	float * x_norm_gpu;
	float * weights_gpu;
	float * weight_updates_gpu;

	float * biases_gpu;
	float * bias_updates_gpu;

	float * scales_gpu;
	float * scale_updates_gpu;

	float * output_gpu;
	float * delta_gpu;
	float * rand_gpu;
	float * squared_gpu;
	float * norms_gpu;
}

static void copy_network_content_to_buffer(int thread_id, int mr_size,
		int start_layer) {
//    pthread_mutex_lock(&global_lock);
//Copy everything here
	//main thread network
	network mt_net = buffer_nets[0];
	//main thread network state
	network_state main_thread_state = buffer_states[0];

	for (int i = 1; i < mr_size; i++) {
		network *current_net = &buffer_nets[i];
		network_state *current_state = &buffer_states[i];
		//-------------------------------------------------------
		//copy all network content
		current_net->n = mt_net.n;
		current_net->batch = mt_net.batch;

		current_net->epoch = mt_net.epoch;
		current_net->subdivisions = mt_net.subdivisions;
		current_net->momentum = mt_net.momentum;
		current_net->decay = mt_net.decay;
		current_net->outputs = mt_net.outputs;
		current_net->policy = mt_net.policy;

		current_net->learning_rate = mt_net.learning_rate;
		current_net->gamma = mt_net.gamma;
		current_net->scale = mt_net.scale;
		current_net->power = mt_net.power;
		current_net->time_steps = mt_net.time_steps;
		current_net->step = mt_net.step;
		current_net->max_batches = mt_net.max_batches;

		current_net->num_steps = mt_net.num_steps;
		current_net->burn_in = mt_net.burn_in;
		current_net->inputs = mt_net.inputs;
		current_net->h = mt_net.h;
		current_net->w = mt_net.w;
		current_net->c = mt_net.c;
		current_net->max_crop = mt_net.max_crop;
		current_net->min_crop = mt_net.min_crop;
		current_net->angle = mt_net.angle;
		current_net->aspect = mt_net.aspect;
		current_net->exposure = mt_net.exposure;
		current_net->saturation = mt_net.saturation;
		current_net->hue = mt_net.hue;
		current_net->gpu_index = mt_net.gpu_index;

		for (int i = start_layer; i < mt_net.n; i++) {
			copy_layer(&current_net->layers[i], &current_net->layers[i]);
		}

		//TODO
		// set it to correct values
		int workspace_size = 10;
		int seen_size = 10;
		int output_size = 10;
		int scales_size = 10;
		int steps_size = 10;
		int input_gpu_size = 10;
		int truth_gpu_size = 10;
		memcpy(current_net->workspace, mt_net.workspace,
				sizeof(float) * workspace_size);
		memcpy(current_net->seen, mt_net.seen, sizeof(int) * seen_size);
		memcpy(current_net->output, mt_net.output, sizeof(float) * output_size);
		memcpy(current_net->scales, mt_net.scales, sizeof(float) * scales_size);
		memcpy(current_net->steps, mt_net.steps, sizeof(int) * steps_size);
		memcpy(*current_net->input_gpu, *mt_net.input_gpu,
				sizeof(float) * input_gpu_size);
		memcpy(*current_net->truth_gpu, *mt_net.truth_gpu,
				sizeof(float) * truth_gpu_size);

	}

//    pthread_mutex_unlock(&global_lock);
}

/**
 * int mr_start_layer will be the variable
 * that contains which layer Modular redundancy
 * starts
 */

void forward_network_gpu_mr(network net, network_state state,
		int mr_start_layer, int thread_id, int mr_size) {
	state.workspace = net.workspace;

//if it is main thread it must start at zero,
//but if it is mr threads it must start at mr_start_layer
	int i = 0;
//That lock
// if it is the main thread it will continue, if not
// must wait
	if (thread_id != 0) {
		sem_wait(&global_semaphore);
		i = mr_start_layer;
	}

	for (; i < net.n; ++i) {
		//-----------------------------------------------------------
		// check if main thread is on the layer
		// that the modular redundancy must start
		if (i == mr_start_layer && thread_id == 0) {
			printf("agora aqui\n");
			copy_network_content_to_buffer(thread_id, mr_size, mr_start_layer);
			sem_post(&global_semaphore);
		}
		//-----------------------------------------------------------
		state.index = i;
		layer l = net.layers[i];
		if (l.delta_gpu) {
			fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1,
					state.st_handle.stream);
		}
		//printf("Passou ate a layer %d\n", i);
		if (l.type == CONVOLUTIONAL) {
			forward_convolutional_layer_gpu(l, state);
		} else if (l.type == DECONVOLUTIONAL) {
			forward_deconvolutional_layer_gpu(l, state);
		} else if (l.type == ACTIVE) {
			forward_activation_layer_gpu(l, state);
		} else if (l.type == LOCAL) {
			forward_local_layer_gpu(l, state);
		} else if (l.type == DETECTION) {
			forward_detection_layer_gpu(l, state);
		} else if (l.type == REGION) {
			forward_region_layer_gpu(l, state);
		} else if (l.type == CONNECTED) {
			forward_connected_layer_gpu(l, state);
		} else if (l.type == RNN) {
			forward_rnn_layer_gpu(l, state);
		} else if (l.type == GRU) {
			forward_gru_layer_gpu(l, state);
		} else if (l.type == CRNN) {
			forward_crnn_layer_gpu(l, state);
		} else if (l.type == CROP) {
			forward_crop_layer_gpu(l, state);
		} else if (l.type == COST) {
			forward_cost_layer_gpu(l, state);
		} else if (l.type == SOFTMAX) {
			forward_softmax_layer_gpu(l, state);
		} else if (l.type == NORMALIZATION) {
			forward_normalization_layer_gpu(l, state);
		} else if (l.type == BATCHNORM) {
			forward_batchnorm_layer_gpu(l, state);
		} else if (l.type == MAXPOOL) {
			forward_maxpool_layer_gpu(l, state);
		} else if (l.type == REORG) {
			forward_reorg_layer_gpu(l, state);
		} else if (l.type == AVGPOOL) {
			forward_avgpool_layer_gpu(l, state);
		} else if (l.type == DROPOUT) {
			forward_dropout_layer_gpu(l, state);
		} else if (l.type == ROUTE) {
			forward_route_layer_gpu(l, net, state);
		} else if (l.type == SHORTCUT) {
			forward_shortcut_layer_gpu(l, state);
		}
		state.input = l.output_gpu;

		cudaStreamSynchronize(state.st_handle.stream);

	}

}

void backward_network_gpu(network net, network_state state) {
	state.workspace = net.workspace;
	int i;
	float * original_input = state.input;
	float * original_delta = state.delta;
	for (i = net.n - 1; i >= 0; --i) {
		state.index = i;
		layer l = net.layers[i];
		if (i == 0) {
			state.input = original_input;
			state.delta = original_delta;
		} else {
			layer prev = net.layers[i - 1];
			state.input = prev.output_gpu;
			state.delta = prev.delta_gpu;
		}
		if (l.type == CONVOLUTIONAL) {
			backward_convolutional_layer_gpu(l, state);
		} else if (l.type == DECONVOLUTIONAL) {
			backward_deconvolutional_layer_gpu(l, state);
		} else if (l.type == ACTIVE) {
			backward_activation_layer_gpu(l, state);
		} else if (l.type == LOCAL) {
			backward_local_layer_gpu(l, state);
		} else if (l.type == MAXPOOL) {
			if (i != 0)
				backward_maxpool_layer_gpu(l, state);
		} else if (l.type == REORG) {
			backward_reorg_layer_gpu(l, state);
		} else if (l.type == AVGPOOL) {
			if (i != 0)
				backward_avgpool_layer_gpu(l, state);
		} else if (l.type == DROPOUT) {
			backward_dropout_layer_gpu(l, state);
		} else if (l.type == DETECTION) {
			backward_detection_layer_gpu(l, state);
		} else if (l.type == REGION) {
			backward_region_layer_gpu(l, state);
		} else if (l.type == NORMALIZATION) {
			backward_normalization_layer_gpu(l, state);
		} else if (l.type == BATCHNORM) {
			backward_batchnorm_layer_gpu(l, state);
		} else if (l.type == SOFTMAX) {
			if (i != 0)
				backward_softmax_layer_gpu(l, state);
		} else if (l.type == CONNECTED) {
			backward_connected_layer_gpu(l, state);
		} else if (l.type == RNN) {
			backward_rnn_layer_gpu(l, state);
		} else if (l.type == GRU) {
			backward_gru_layer_gpu(l, state);
		} else if (l.type == CRNN) {
			backward_crnn_layer_gpu(l, state);
		} else if (l.type == COST) {
			backward_cost_layer_gpu(l, state);
		} else if (l.type == ROUTE) {
			backward_route_layer_gpu(l, net, state);
		} else if (l.type == SHORTCUT) {
			backward_shortcut_layer_gpu(l, state);
		}
	}
}

void update_network_gpu(network net, cudaStream_t stream) {
	int i;
	int update_batch = net.batch * net.subdivisions;
	float rate = get_current_rate(net);
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL) {
			update_convolutional_layer_gpu(l, update_batch, rate, net.momentum,
					net.decay, stream);
		} else if (l.type == DECONVOLUTIONAL) {
			update_deconvolutional_layer_gpu(l, rate, net.momentum, net.decay,
					stream);
		} else if (l.type == CONNECTED) {
			update_connected_layer_gpu(l, update_batch, rate, net.momentum,
					net.decay, stream);
		} else if (l.type == GRU) {
			update_gru_layer_gpu(l, update_batch, rate, net.momentum, net.decay,
					stream);
		} else if (l.type == RNN) {
			update_rnn_layer_gpu(l, update_batch, rate, net.momentum, net.decay,
					stream);
		} else if (l.type == CRNN) {
			update_crnn_layer_gpu(l, update_batch, rate, net.momentum,
					net.decay, stream);
		} else if (l.type == LOCAL) {
			update_local_layer_gpu(l, update_batch, rate, net.momentum,
					net.decay, stream);
		}
	}
}

void forward_backward_network_gpu(network net, float *x, float *y) {
	network_state state;
	state.index = 0;
	state.net = net;
	int x_size = get_network_input_size(net) * net.batch;
	int y_size = get_network_output_size(net) * net.batch;
	if (net.layers[net.n - 1].truths)
		y_size = net.layers[net.n - 1].truths * net.batch;
	if (!*net.input_gpu) {
		*net.input_gpu = cuda_make_array(x, x_size);
		*net.truth_gpu = cuda_make_array(y, y_size);
	} else {
		cuda_push_array(*net.input_gpu, x, x_size);
		cuda_push_array(*net.truth_gpu, y, y_size);
	}
	state.input = *net.input_gpu;
	state.delta = 0;
	state.truth = *net.truth_gpu;
	state.train = 1;

	forward_network_gpu(net, state);

	backward_network_gpu(net, state);
}

float train_network_datum_gpu(network net, float *x, float *y,
		cudaStream_t stream) {
	*net.seen += net.batch;
	forward_backward_network_gpu(net, x, y);
	float error = get_network_cost(net);
	if (((*net.seen) / net.batch) % net.subdivisions == 0)
		update_network_gpu(net, stream);

	return error;
}

typedef struct {
	network net;
	data d;
	float *err;
} train_args;

void *train_thread(void *ptr) {
	train_args args = *(train_args*) ptr;
	free(ptr);
	cuda_set_device(args.net.gpu_index);
	*args.err = train_network(args.net, args.d);
	return 0;
}

pthread_t train_network_in_thread(network net, data d, float *err) {
	pthread_t thread;
	train_args *ptr = (train_args *) calloc(1, sizeof(train_args));
	ptr->net = net;
	ptr->d = d;
	ptr->err = err;
	if (pthread_create(&thread, 0, train_thread, ptr))
		error("Thread creation failed");
	return thread;
}

void pull_updates(layer l) {
	if (l.type == CONVOLUTIONAL) {
		cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
		cuda_pull_array(l.weight_updates_gpu, l.weight_updates,
				l.n * l.size * l.size * l.c);
		if (l.scale_updates)
			cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
	} else if (l.type == CONNECTED) {
		cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
		cuda_pull_array(l.weight_updates_gpu, l.weight_updates,
				l.outputs * l.inputs);
	}
}

void push_updates(layer l) {
	if (l.type == CONVOLUTIONAL) {
		cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
		cuda_push_array(l.weight_updates_gpu, l.weight_updates,
				l.n * l.size * l.size * l.c);
		if (l.scale_updates)
			cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
	} else if (l.type == CONNECTED) {
		cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
		cuda_push_array(l.weight_updates_gpu, l.weight_updates,
				l.outputs * l.inputs);
	}
}

void update_layer(layer l, network net) {
	int update_batch = net.batch * net.subdivisions;
	float rate = get_current_rate(net);
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	if (l.type == CONVOLUTIONAL) {
		update_convolutional_layer_gpu(l, update_batch, rate, net.momentum,
				net.decay, stream);
	} else if (l.type == DECONVOLUTIONAL) {
		update_deconvolutional_layer_gpu(l, rate, net.momentum, net.decay,
				stream);
	} else if (l.type == CONNECTED) {
		update_connected_layer_gpu(l, update_batch, rate, net.momentum,
				net.decay, stream);
	} else if (l.type == RNN) {
		update_rnn_layer_gpu(l, update_batch, rate, net.momentum, net.decay,
				stream);
	} else if (l.type == GRU) {
		update_gru_layer_gpu(l, update_batch, rate, net.momentum, net.decay,
				stream);
	} else if (l.type == CRNN) {
		update_crnn_layer_gpu(l, update_batch, rate, net.momentum, net.decay,
				stream);
	} else if (l.type == LOCAL) {
		update_local_layer_gpu(l, update_batch, rate, net.momentum, net.decay,
				stream);
	}
	cudaStreamDestroy(stream);
}

void merge_weights(layer l, layer base) {
	if (l.type == CONVOLUTIONAL) {
		axpy_cpu(l.n, 1, l.biases, 1, base.biases, 1);
		axpy_cpu(l.n * l.size * l.size * l.c, 1, l.weights, 1, base.weights, 1);
		if (l.scales) {
			axpy_cpu(l.n, 1, l.scales, 1, base.scales, 1);
		}
	} else if (l.type == CONNECTED) {
		axpy_cpu(l.outputs, 1, l.biases, 1, base.biases, 1);
		axpy_cpu(l.outputs * l.inputs, 1, l.weights, 1, base.weights, 1);
	}
}

void scale_weights(layer l, float s) {
	if (l.type == CONVOLUTIONAL) {
		scal_cpu(l.n, s, l.biases, 1);
		scal_cpu(l.n * l.size * l.size * l.c, s, l.weights, 1);
		if (l.scales) {
			scal_cpu(l.n, s, l.scales, 1);
		}
	} else if (l.type == CONNECTED) {
		scal_cpu(l.outputs, s, l.biases, 1);
		scal_cpu(l.outputs * l.inputs, s, l.weights, 1);
	}
}

void pull_weights(layer l) {
	if (l.type == CONVOLUTIONAL) {
		cuda_pull_array(l.biases_gpu, l.biases, l.n);
		cuda_pull_array(l.weights_gpu, l.weights, l.n * l.size * l.size * l.c);
		if (l.scales)
			cuda_pull_array(l.scales_gpu, l.scales, l.n);
	} else if (l.type == CONNECTED) {
		cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
		cuda_pull_array(l.weights_gpu, l.weights, l.outputs * l.inputs);
	}
}

void push_weights(layer l) {
	if (l.type == CONVOLUTIONAL) {
		cuda_push_array(l.biases_gpu, l.biases, l.n);
		cuda_push_array(l.weights_gpu, l.weights, l.n * l.size * l.size * l.c);
		if (l.scales)
			cuda_push_array(l.scales_gpu, l.scales, l.n);
	} else if (l.type == CONNECTED) {
		cuda_push_array(l.biases_gpu, l.biases, l.outputs);
		cuda_push_array(l.weights_gpu, l.weights, l.outputs * l.inputs);
	}
}

void distribute_weights(layer l, layer base) {
	if (l.type == CONVOLUTIONAL) {
		cuda_push_array(l.biases_gpu, base.biases, l.n);
		cuda_push_array(l.weights_gpu, base.weights,
				l.n * l.size * l.size * l.c);
		if (base.scales)
			cuda_push_array(l.scales_gpu, base.scales, l.n);
	} else if (l.type == CONNECTED) {
		cuda_push_array(l.biases_gpu, base.biases, l.outputs);
		cuda_push_array(l.weights_gpu, base.weights, l.outputs * l.inputs);
	}
}

void merge_updates(layer l, layer base) {
	if (l.type == CONVOLUTIONAL) {
		axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
		axpy_cpu(l.n * l.size * l.size * l.c, 1, l.weight_updates, 1,
				base.weight_updates, 1);
		if (l.scale_updates) {
			axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
		}
	} else if (l.type == CONNECTED) {
		axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
		axpy_cpu(l.outputs * l.inputs, 1, l.weight_updates, 1,
				base.weight_updates, 1);
	}
}

void distribute_updates(layer l, layer base) {
	if (l.type == CONVOLUTIONAL) {
		cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
		cuda_push_array(l.weight_updates_gpu, base.weight_updates,
				l.n * l.size * l.size * l.c);
		if (base.scale_updates)
			cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
	} else if (l.type == CONNECTED) {
		cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
		cuda_push_array(l.weight_updates_gpu, base.weight_updates,
				l.outputs * l.inputs);
	}
}

void sync_layer(network *nets, int n, int j) {
//printf("Syncing layer %d\n", j);
	int i;
	network net = nets[0];
	layer base = net.layers[j];
	cuda_set_device(net.gpu_index);
	pull_weights(base);
	for (i = 1; i < n; ++i) {
		cuda_set_device(nets[i].gpu_index);
		layer l = nets[i].layers[j];
		pull_weights(l);
		merge_weights(l, base);
	}
	scale_weights(base, 1. / n);
	for (i = 0; i < n; ++i) {
		cuda_set_device(nets[i].gpu_index);
		layer l = nets[i].layers[j];
		distribute_weights(l, base);
	}
//printf("Done syncing layer %d\n", j);
}

typedef struct {
	network *nets;
	int n;
	int j;
} sync_args;

void *sync_layer_thread(void *ptr) {
	sync_args args = *(sync_args*) ptr;
	sync_layer(args.nets, args.n, args.j);
	free(ptr);
	return 0;
}

pthread_t sync_layer_in_thread(network *nets, int n, int j) {
	pthread_t thread;
	sync_args *ptr = (sync_args *) calloc(1, sizeof(sync_args));
	ptr->nets = nets;
	ptr->n = n;
	ptr->j = j;
	if (pthread_create(&thread, 0, sync_layer_thread, ptr))
		error("Thread creation failed");
	return thread;
}

void sync_nets(network *nets, int n, int interval) {
	int j;
	int layers = nets[0].n;
	pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

	*nets[0].seen += interval * (n - 1) * nets[0].batch * nets[0].subdivisions;
	for (j = 0; j < n; ++j) {
		*nets[j].seen = *nets[0].seen;
	}
	for (j = 0; j < layers; ++j) {
		threads[j] = sync_layer_in_thread(nets, n, j);
	}
	for (j = 0; j < layers; ++j) {
		pthread_join(threads[j], 0);
	}
	free(threads);
}

float train_networks(network *nets, int n, data d, int interval) {
	int i;
	int batch = nets[0].batch;
	int subdivisions = nets[0].subdivisions;
	assert(batch * subdivisions * n == d.X.rows);
	pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
	float *errors = (float *) calloc(n, sizeof(float));

	float sum = 0;
	for (i = 0; i < n; ++i) {
		data p = get_data_part(d, i, n);
		threads[i] = train_network_in_thread(nets[i], p, errors + i);
	}
	for (i = 0; i < n; ++i) {
		pthread_join(threads[i], 0);
		printf("%f\n", errors[i]);
		sum += errors[i];
	}
	if (get_current_batch(nets[0]) % interval == 0) {
		printf("Syncing... ");
		sync_nets(nets, n, interval);
		printf("Done!\n");
	}
	free(threads);
	free(errors);
	return (float) sum / (n);
}

float *get_network_output_layer_gpu(network net, int i) {
	layer l = net.layers[i];
	cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
	return l.output;
}

float *get_network_output_gpu(network net) {
	int i;
	for (i = net.n - 1; i > 0; --i)
		if (net.layers[i].type != COST)
			break;
	return get_network_output_layer_gpu(net, i);
}

/**
 * This function will be called by the pthread create
 */
void *network_predict_gpu_mr(void* data) {
	network net = ((thread_parameters*) data)->net;
	float *input = ((thread_parameters*) data)->input;
	multi_thread_hd_st st_handle = create_handle();
	int start_layer = ((thread_parameters*) data)->start_layer;
	int thread_id = ((thread_parameters*) data)->thread_id;
	int mr_size = ((thread_parameters*) data)->mr_size;

	int size = get_network_input_size(net) * net.batch;
	network_state state;
	state.index = 0;
	state.net = net;
	state.input = cuda_make_array(input, size);
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
	state.st_handle = st_handle;

//If start_layer == 0 then it is normal DMR or TMR
	if (start_layer == 0) {
		forward_network_gpu(net, state);
	} else {
		buffer_states[thread_id] = state;
		buffer_nets[thread_id] = net;
		forward_network_gpu_mr(net, state, start_layer, thread_id, mr_size);
	}

	((thread_parameters*) data)->out = get_network_output_gpu(net);
	cuda_free(state.input);
	destroy_handle(&st_handle);
	return (void*) NULL;
}

float *network_predict_gpu(network net, float *input) {
	int size = get_network_input_size(net) * net.batch;
	network_state state;
	state.index = 0;
	state.net = net;
	state.input = cuda_make_array(input, size);
	state.truth = 0;
	state.train = 0;
	state.delta = 0;
	state.st_handle = create_handle();

	forward_network_gpu(net, state);
	float *out = get_network_output_gpu(net);
	cuda_free(state.input);
	destroy_handle(&state.st_handle);
	return out;
}

