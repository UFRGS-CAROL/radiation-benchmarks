#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

#include "abft.h"

typedef layer maxpool_layer;



image get_maxpool_image(maxpool_layer l);
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void resize_maxpool_layer(maxpool_layer *l, int w, int h);
void forward_maxpool_layer(const maxpool_layer l, network net);
void backward_maxpool_layer(const maxpool_layer l, network net);


#ifdef GPU
void forward_maxpool_layer_gpu(maxpool_layer l, network net);
void backward_maxpool_layer_gpu(maxpool_layer l, network net);

void forward_maxpool_layer_gpu_hardened(maxpool_layer l, network net);

void get_and_reset_error_detected_values(error_return *host_error);
void free_err_detected();

#endif


/**
 * 0 for no abft
 * 1 for Abraham abft
 * 2 for maxpool hardened
 * 3 l1
 * 4 l2
 * 5 trained_weights
 */
void set_abft_smartpool(int type);




#endif

