#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "cuda.h"
#include "math.h"

ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a);
real_t activate(real_t x, ACTIVATION a);
real_t gradient(real_t x, ACTIVATION a);
void gradient_array(const real_t *x, const int n, const ACTIVATION a,
		real_t *delta);
void activate_array(real_t *x, const int n, const ACTIVATION a);
#ifdef GPU
void activate_array_gpu(real_t *x, int n, ACTIVATION a, cudaStream_t st);
void gradient_array_gpu(real_t *x, int n, ACTIVATION a, real_t *delta, cudaStream_t st);
#endif

static inline real_t stair_activate(real_t x) {
	int n = floor(x);
	if (n % 2 == 0)
		return floor(x / 2.);
	else
		return (x - n) + floor(x / 2.);
}
static inline real_t hardtan_activate(real_t x) {
	if (x < -1)
		return -1;
	if (x > 1)
		return 1;
	return x;
}
static inline real_t linear_activate(real_t x) {
	return x;
}
static inline real_t logistic_activate(real_t x) {
	return 1. / (1. + exp(-x));
}
static inline real_t loggy_activate(real_t x) {
	return 2. / (1. + exp(-x)) - 1;
}
static inline real_t relu_activate(real_t x) {
	return x * (x > 0);
}
static inline real_t elu_activate(real_t x) {
	return (x >= 0) * x + (x < 0) * (exp(x) - 1);
}
static inline real_t selu_activate(real_t x) {
	return (x >= 0) * 1.0507 * x + (x < 0) * 1.0507 * 1.6732 * (exp(x) - 1);
}
static inline real_t relie_activate(real_t x) {
	return (x > 0) ? x : .01 * x;
}
static inline real_t ramp_activate(real_t x) {
	return x * (x > 0) + .1 * x;
}
static inline real_t leaky_activate(real_t x) {
	return (x > 0) ? x : .1 * x;
}
static inline real_t tanh_activate(real_t x) {
	return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}
static inline real_t plse_activate(real_t x) {
	if (x < -4)
		return .01 * (x + 4);
	if (x > 4)
		return .01 * (x - 4) + 1;
	return .125 * x + .5;
}

static inline real_t lhtan_activate(real_t x) {
	if (x < 0)
		return .001 * x;
	if (x > 1)
		return .001 * (x - 1) + 1;
	return x;
}
static inline real_t lhtan_gradient(real_t x) {
	if (x > 0 && x < 1)
		return 1;
	return .001;
}

static inline real_t hardtan_gradient(real_t x) {
	if (x > -1 && x < 1)
		return 1;
	return 0;
}
static inline real_t linear_gradient(real_t x) {
	return 1;
}
static inline real_t logistic_gradient(real_t x) {
	return (1 - x) * x;
}
static inline real_t loggy_gradient(real_t x) {
	real_t y = (x + 1.) / 2.;
	return 2 * (1 - y) * y;
}
static inline real_t stair_gradient(real_t x) {
	if (floor(x) == x)
		return 0;
	return 1;
}
static inline real_t relu_gradient(real_t x) {
	return (x > 0);
}
static inline real_t elu_gradient(real_t x) {
	return (x >= 0) + (x < 0) * (x + 1);
}
static inline real_t selu_gradient(real_t x) {
	return (x >= 0) * 1.0507 + (x < 0) * (x + 1.0507 * 1.6732);
}
static inline real_t relie_gradient(real_t x) {
	return (x > 0) ? 1 : .01;
}
static inline real_t ramp_gradient(real_t x) {
	return (x > 0) + .1;
}
static inline real_t leaky_gradient(real_t x) {
	return (x > 0) ? 1 : .1;
}
static inline real_t tanh_gradient(real_t x) {
	return 1 - x * x;
}
static inline real_t plse_gradient(real_t x) {
	return (x < 0 || x > 1) ? .01 : .125;
}

#endif

