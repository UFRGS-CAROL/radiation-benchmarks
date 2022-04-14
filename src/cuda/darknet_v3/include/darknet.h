#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "type.h"

#ifdef GPU
#define BLOCK 512

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef CUDNN
#include "cudnn.h"
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234
extern int gpu_index;

typedef struct {
	int classes;
	char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct {
	int *leaf;
	int n;
	int *parent;
	int *child;
	int *group;
	char **name;

	int groups;
	int *group_size;
	int *group_offset;
} tree;
tree *read_tree(char *filename);

typedef enum {
	LOGISTIC,
	RELU,
	RELIE,
	LINEAR,
	RAMP,
	TANH,
	PLSE,
	LEAKY,
	ELU,
	LOGGY,
	STAIR,
	HARDTAN,
	LHTAN,
	SELU
} ACTIVATION;

typedef enum {
	PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum {
	MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
	CONVOLUTIONAL,
	DECONVOLUTIONAL,
	CONNECTED,
	MAXPOOL,
	SOFTMAX,
	DETECTION,
	DROPOUT,
	CROP,
	ROUTE,
	COST,
	NORMALIZATION,
	AVGPOOL,
	LOCAL,
	SHORTCUT,
	ACTIVE,
	RNN,
	GRU,
	LSTM,
	CRNN,
	BATCHNORM,
	NETWORK,
	XNOR,
	REGION,
	YOLO,
	ISEG,
	REORG,
	UPSAMPLE,
	LOGXENT,
	L2NORM,
	BLANK
} LAYER_TYPE;

typedef enum {
	SSE, MASKED, L1, SEG, SMOOTH, WGAN
} COST_TYPE;

typedef struct {
	int batch;
	real_t learning_rate;
	real_t momentum;
	real_t decay;
	int adam;
	real_t B1;
	real_t B2;
	real_t eps;
	int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct layer {
	LAYER_TYPE type;
	ACTIVATION activation;
	COST_TYPE cost_type;
	void (*forward)(struct layer, struct network);
	void (*backward)(struct layer, struct network);
	void (*update)(struct layer, update_args);
	void (*forward_gpu)(struct layer, struct network);
	void (*backward_gpu)(struct layer, struct network);
#ifdef GPU
	void (*update_gpu)(struct layer, update_args, cudaStream_t st);
#else
	void (*update_gpu)(struct layer, update_args);
#endif
	int batch_normalize;
	int shortcut;
	int batch;
	int forced;
	int flipped;
	int inputs;
	int outputs;
	int nweights;
	int nbiases;
	int extra;
	int truths;
	int h, w, c;
	int out_h, out_w, out_c;
	int n;
	int max_boxes;
	int groups;
	int size;
	int side;
	int stride;
	int reverse;
	int flatten;
	int spatial;
	int pad;
	int sqrt;
	int flip;
	int index;
	int binary;
	int xnor;
	int steps;
	int hidden;
	int truth;
	real_t smooth;
	real_t dot;
	real_t angle;
	real_t jitter;
	real_t saturation;
	real_t exposure;
	real_t shift;
	real_t ratio;
	real_t learning_rate_scale;
	real_t clip;
	int noloss;
	int softmax;
	int classes;
	int coords;
	int background;
	int rescore;
	int objectness;
	int joint;
	int noadjust;
	int reorg;
	int log;
	int tanh;
	int *mask;
	int total;

	real_t alpha;
	real_t beta;
	real_t kappa;

	real_t coord_scale;
	real_t object_scale;
	real_t noobject_scale;
	real_t mask_scale;
	real_t class_scale;
	int bias_match;
	int random;
	real_t ignore_thresh;
	real_t truth_thresh;
	real_t thresh;
	real_t focus;
	int classfix;
	int absolute;

	int onlyforward;
	int stopbackward;
	int dontload;
	int dontsave;
	int dontloadscales;
	int numload;

	real_t temperature;
	real_t probability;
	real_t scale;

	char * cweights;
	int * indexes;
	int * input_layers;
	int * input_sizes;
	int * map;
	int * counts;
	real_t ** sums;
	real_t * rand;
	real_t * cost;
	real_t * state;
	real_t * prev_state;
	real_t * forgot_state;
	real_t * forgot_delta;
	real_t * state_delta;
	real_t * combine_cpu;
	real_t * combine_delta_cpu;

	real_t * concat;
	real_t * concat_delta;

	real_t * binary_weights;

	real_t * biases;
	real_t * bias_updates;

	real_t * scales;
	real_t * scale_updates;

	real_t * weights;
	real_t * weight_updates;

	real_t * delta;
	real_t * output;
	real_t * loss;
	real_t * squared;
	real_t * norms;

	real_t * spatial_mean;
	real_t * mean;
	real_t * variance;

	real_t * mean_delta;
	real_t * variance_delta;

	real_t * rolling_mean;
	real_t * rolling_variance;

	real_t * x;
	real_t * x_norm;

	real_t * m;
	real_t * v;

	real_t * bias_m;
	real_t * bias_v;
	real_t * scale_m;
	real_t * scale_v;

	real_t *z_cpu;
	real_t *r_cpu;
	real_t *h_cpu;
	real_t * prev_state_cpu;

	real_t *temp_cpu;
	real_t *temp2_cpu;
	real_t *temp3_cpu;

	real_t *dh_cpu;
	real_t *hh_cpu;
	real_t *prev_cell_cpu;
	real_t *cell_cpu;
	real_t *f_cpu;
	real_t *i_cpu;
	real_t *g_cpu;
	real_t *o_cpu;
	real_t *c_cpu;
	real_t *dc_cpu;

	real_t * binary_input;

	struct layer *input_layer;
	struct layer *self_layer;
	struct layer *output_layer;

	struct layer *reset_layer;
	struct layer *update_layer;
	struct layer *state_layer;

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

	struct layer *wz;
	struct layer *uz;
	struct layer *wr;
	struct layer *ur;
	struct layer *wh;
	struct layer *uh;
	struct layer *uo;
	struct layer *wo;
	struct layer *uf;
	struct layer *wf;
	struct layer *ui;
	struct layer *wi;
	struct layer *ug;
	struct layer *wg;

	tree *softmax_tree;

	size_t workspace_size;

#ifdef GPU
	int *indexes_gpu;

	real_t *z_gpu;
	real_t *r_gpu;
	real_t *h_gpu;

	real_t *temp_gpu;
	real_t *temp2_gpu;
	real_t *temp3_gpu;

	real_t *dh_gpu;
	real_t *hh_gpu;
	real_t *prev_cell_gpu;
	real_t *cell_gpu;
	real_t *f_gpu;
	real_t *i_gpu;
	real_t *g_gpu;
	real_t *o_gpu;
	real_t *c_gpu;
	real_t *dc_gpu;

	real_t *m_gpu;
	real_t *v_gpu;
	real_t *bias_m_gpu;
	real_t *scale_m_gpu;
	real_t *bias_v_gpu;
	real_t *scale_v_gpu;

	real_t * combine_gpu;
	real_t * combine_delta_gpu;

	real_t * prev_state_gpu;
	real_t * forgot_state_gpu;
	real_t * forgot_delta_gpu;
	real_t * state_gpu;
	real_t * state_delta_gpu;
	real_t * gate_gpu;
	real_t * gate_delta_gpu;
	real_t * save_gpu;
	real_t * save_delta_gpu;
	real_t * concat_gpu;
	real_t * concat_delta_gpu;

	real_t * binary_input_gpu;
	real_t * binary_weights_gpu;

	real_t * mean_gpu;
	real_t * variance_gpu;

	real_t * rolling_mean_gpu;
	real_t * rolling_variance_gpu;

	real_t * variance_delta_gpu;
	real_t * mean_delta_gpu;

	real_t * x_gpu;
	real_t * x_norm_gpu;
	real_t * weights_gpu;
	real_t * weight_updates_gpu;
	real_t * weight_change_gpu;

	real_t * biases_gpu;
	real_t * bias_updates_gpu;
	real_t * bias_change_gpu;

	real_t * scales_gpu;
	real_t * scale_updates_gpu;
	real_t * scale_change_gpu;

	real_t * output_gpu;
	real_t * loss_gpu;
	real_t * delta_gpu;
	real_t * rand_gpu;
	real_t * squared_gpu;
	real_t * norms_gpu;
#ifdef CUDNN
	cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
	cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
	cudnnTensorDescriptor_t normTensorDesc;
	cudnnFilterDescriptor_t weightDesc;
	cudnnFilterDescriptor_t dweightDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionFwdAlgo_t fw_algo;
	cudnnConvolutionBwdDataAlgo_t bd_algo;
	cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

void free_layer(layer);

typedef enum {
	CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network {
	int n;
	int batch;
	size_t *seen;
	int *t;
	real_t epoch;
	int subdivisions;
	layer *layers;
	real_t *output;
	learning_rate_policy policy;

	real_t learning_rate;
	real_t momentum;
	real_t decay;
	real_t gamma;
	real_t scale;
	real_t power;
	int time_steps;
	int step;
	int max_batches;
	real_t *scales;
	int *steps;
	int num_steps;
	int burn_in;

	int adam;
	real_t B1;
	real_t B2;
	real_t eps;

	int inputs;
	int outputs;
	int truths;
	int notruth;
	int h, w, c;
	int max_crop;
	int min_crop;
	real_t max_ratio;
	real_t min_ratio;
	int center;
	real_t angle;
	real_t aspect;
	real_t exposure;
	real_t saturation;
	real_t hue;
	int random;

	int gpu_index;
	tree *hierarchy;

	real_t *input;
	real_t *truth;
	real_t *delta;
	real_t *workspace;
	int train;
	int index;
	real_t *cost;
	real_t clip;

#ifdef GPU
	real_t *input_gpu;
	real_t *truth_gpu;
	real_t *delta_gpu;
	real_t *output_gpu;

	unsigned char use_tensor_cores;
	cudaStream_t st;
#endif
    //Does not work
//	int smx_redundancy;


} network;

typedef struct {
	int w;
	int h;
	real_t scale;
	real_t rad;
	real_t dx;
	real_t dy;
	real_t aspect;
} augment_args;

typedef struct {
	int w;
	int h;
	int c;
	real_t *data;
} image;

typedef struct {
	real_t x, y, w, h;
} box;

typedef struct detection {
	box bbox;
	int classes;
	real_t *prob;
	real_t *mask;
	real_t objectness;
	int sort_class;
} detection;

typedef struct matrix {
	int rows, cols;
	real_t **vals;
} matrix;

typedef struct {
	int w, h;
	matrix X;
	matrix y;
	int shallow;
	int *num_boxes;
	box **boxes;
} data;

typedef enum {
	CLASSIFICATION_DATA,
	DETECTION_DATA,
	CAPTCHA_DATA,
	REGION_DATA,
	IMAGE_DATA,
	COMPARE_DATA,
	WRITING_DATA,
	SWAG_DATA,
	TAG_DATA,
	OLD_CLASSIFICATION_DATA,
	STUDY_DATA,
	DET_DATA,
	SUPER_DATA,
	LETTERBOX_DATA,
	REGRESSION_DATA,
	SEGMENTATION_DATA,
	INSTANCE_DATA,
	ISEG_DATA
} data_type;

typedef struct load_args {
	int threads;
	char **paths;
	char *path;
	int n;
	int m;
	char **labels;
	int h;
	int w;
	int out_w;
	int out_h;
	int nh;
	int nw;
	int num_boxes;
	int min, max, size;
	int classes;
	int background;
	int scale;
	int center;
	int coords;
	real_t jitter;
	real_t angle;
	real_t aspect;
	real_t saturation;
	real_t exposure;
	real_t hue;
	data *d;
	image *im;
	image *resized;
	data_type type;
	tree *hierarchy;
} load_args;

typedef struct {
	int id;
	real_t x, y, w, h;
	real_t left, right, top, bottom;
} box_label;

network *load_network(char *cfg, char *weights, int clear);
load_args get_base_args(network *net);

void free_data(data d);

typedef struct node {
	void *val;
	struct node *next;
	struct node *prev;
} node;

typedef struct list {
	int size;
	node *front;
	node *back;
} list;

pthread_t load_data(load_args args);
list *read_data_cfg(char *filename);
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);
data resize_data(data orig, int w, int h);
data *tile_data(data orig, int divs, int size);
data select_data(data *orig, int *inds);

void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);

real_t dot_cpu(int N, real_t *X, int INCX, real_t *Y, int INCY);
void axpy_cpu(int N, real_t ALPHA, real_t *X, int INCX, real_t *Y, int INCY);
void copy_cpu(int N, real_t *X, int INCX, real_t *Y, int INCY);
void scal_cpu(int N, real_t ALPHA, real_t *X, int INCX);
void fill_cpu(int N, real_t ALPHA, real_t * X, int INCX);
void normalize_cpu(real_t *x, real_t *mean, real_t *variance, int batch,
		int filters, int spatial);
void softmax(real_t *input, int n, real_t temp, int stride, real_t *output);

int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
void axpy_gpu(int N, real_t ALPHA, real_t * X, int INCX, real_t * Y, int INCY,
		cudaStream_t st);
void fill_gpu(int N, real_t ALPHA, real_t * X, int INCX,
		cudaStream_t st);
void scal_gpu(int N, real_t ALPHA, real_t * X, int INCX,
		cudaStream_t st);
void copy_gpu(int N, real_t * X, int INCX, real_t * Y, int INCY,
		cudaStream_t st);

void cuda_set_device(int n);
void cuda_free(real_t *x_gpu);
real_t *cuda_make_array(real_t *x, size_t n);
void cuda_pull_array(real_t *x_gpu, real_t *x, size_t n);
real_t cuda_mag_array(real_t *x_gpu, size_t n);
void cuda_push_array(real_t *x_gpu, real_t *x, size_t n);

void forward_network_gpu(network *net);
void forward_network_gpu_parallel(network **netp_array);
void backward_network_gpu(network *net);
void update_network_gpu(network *net);

real_t train_networks(network **nets, int n, data d, int interval);
void sync_nets(network **nets, int n, int interval);
void harmless_update_network_gpu(network *net);
#endif
image get_label(image **characters, char *string, int size);
void draw_label(image a, int r, int c, image label, const real_t *rgb);
void save_image(image im, const char *name);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void get_next_batch(data d, int n, int offset, real_t *X, real_t *y);
void grayscale_image_3c(image im);
void normalize_image(image p);
void matrix_to_csv(matrix m);
real_t train_network_sgd(network *net, data d, int n);
void rgbgr_image(image im);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_cifar10_data(char *filename);
real_t matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, real_t scale);
matrix csv_to_matrix(char *filename);
real_t *network_accuracies(network *net, data d, int n);
real_t train_network_datum(network *net);
image make_random_image(int w, int h, int c);

void denormalize_connected_layer(layer l);
void denormalize_convolutional_layer(layer l);
void statistics_connected_layer(layer l);
void rescale_weights(layer l, real_t scale, real_t trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

void demo(char *cfgfile, char *weightfile, real_t thresh, int cam_index,
		const char *filename, char **names, int classes, int frame_skip,
		char *prefix, int avg, real_t hier_thresh, int w, int h, int fps,
		int fullscreen);
void get_detection_detections(layer l, int w, int h, real_t thresh,
		detection *dets);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);

network *parse_network_cfg(char *filename);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);

void zero_objectness(layer l);
void get_region_detections(layer l, int w, int h, int netw, int neth,
		real_t thresh, int *map, real_t tree_thresh, int relative,
		detection *dets);
int get_yolo_detections(layer l, int w, int h, int netw, int neth,
		real_t thresh, int *map, int relative, detection *dets);
void free_network(network *net);
void set_batch_network(network *net, int b);
void set_temp_network(network *net, real_t t);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void censor_image(image im, int dx, int dy, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, real_t thresh);
image mask_to_rgb(image mask);
int resize_network(network *net, int w, int h);
void free_matrix(matrix m);
void test_resize(char *filename);
int show_image(image p, const char *name, int ms);
image copy_image(image p);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, real_t r,
		real_t g, real_t b);
real_t get_current_rate(network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w,
		int h);
size_t get_current_batch(network *net);
void constrain_image(image im);
image get_network_image_layer(network *net, int i);
layer get_network_output_layer(network *net);
void top_predictions(network *net, int n, int *index);
void flip_image(image a);
image real_t_to_image(int w, int h, int c, real_t *data);
void ghost_image(image source, image dest, int dx, int dy);
real_t network_accuracy(network *net, data d);
void random_distort_image(image im, real_t hue, real_t saturation,
		real_t exposure);
void fill_image(image m, real_t s);
image grayscale_image(image im);
void rotate_image_cw(image im, int times);
double what_time_is_it_now();
image rotate_image(image m, real_t rad);
void visualize_network(network *net);
real_t box_iou(box a, box b);
data load_all_cifar10();
box_label *read_boxes(char *filename, int *n);
box real_t_to_box(real_t *f, int stride);
void draw_detections(image im, detection *dets, int num, real_t thresh,
		char **names, image **alphabet, int classes);

matrix network_predict_data(network *net, data test);
image **load_alphabet();
image get_network_image(network *net);

real_t *network_predict(network *net, real_t *input);

//TODO: Fernando
void network_predict_smx_red(network **net, real_t **input);

int network_width(network *net);
int network_height(network *net);
real_t *network_predict_image(network *net, image im);
void network_detect(network *net, image im, real_t thresh, real_t hier_thresh,
		real_t nms, detection *dets);
detection *get_network_boxes(network *net, int w, int h, real_t thresh,
		real_t hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);

void reset_network_state(network *net, int b);

char **get_labels(char *filename);
void do_nms_obj(detection *dets, int total, int classes, real_t thresh);
void do_nms_sort(detection *dets, int total, int classes, real_t thresh);

matrix make_matrix(int rows, int cols);

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
void make_window(char *name, int w, int h, int fullscreen);
#endif

void free_image(image m);
real_t train_network(network *net, data d);
pthread_t load_data_in_thread(load_args args);
void load_data_blocking(load_args args);
list *get_paths(char *filename);
void hierarchy_predictions(real_t *predictions, int n, tree *hier,
		int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
real_t find_real_t_arg(int argc, char **argv, char *arg, real_t def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
real_t sec(clock_t clocks);
void **list_to_array(list *l);
void top_k(real_t *a, int n, int k, int *index);
int *read_map(char *filename);
void error(const char *s);
int max_index(real_t *a, int n);
int max_int_index(int *a, int n);
int sample_array(real_t *a, int n);
int *random_index_order(int min, int max);
void free_list(list *l);
real_t mse_array(real_t *a, int n);
real_t variance_array(real_t *a, int n);
real_t mag_array(real_t *a, int n);
void scale_array(real_t *a, int n, real_t s);
real_t mean_array(real_t *a, int n);
real_t sum_array(real_t *a, int n);
void normalize_array(real_t *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
real_t rand_normal();
real_t rand_uniform(real_t min, real_t max);

#ifdef __cplusplus
}
#endif
#endif
