#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "cuda.h"
#include "math.h"
#include "type.h"

ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a);
real_t activate(real_t x, ACTIVATION a);
real_t gradient(real_t x, ACTIVATION a);
void gradient_array(const real_t *x, const int n, const ACTIVATION a,
		real_t *delta);
void activate_array(real_t *x, const int n, const ACTIVATION a);
#ifdef GPU
void activate_array_gpu(real_t_device *x, int n, ACTIVATION a);
void gradient_array_gpu(real_t_device *x, int n, ACTIVATION a, real_t_device *delta);
#endif

static inline real_t stair_activate(real_t x) {
	int n = floor(x);
	if (n % 2 == 0)
		return real_t(floor(x / 2.));
	else
		return real_t((x - n) + floor(x / 2.));
}
static inline real_t hardtan_activate(real_t x) {
	if (x < -1)
		return real_t(-1);
	if (x > 1)
		return real_t(1);
	return x;
}
static inline real_t linear_activate(real_t x) {
	return x;
}
static inline real_t logistic_activate(real_t x) {
	return real_t(1. / (1. + exp(-x)));
}
static inline real_t loggy_activate(real_t x) {
	return real_t(2. / (1. + exp(-x)) - 1);
}
static inline real_t relu_activate(real_t x) {
	return real_t(x * (x > 0));
}
static inline real_t elu_activate(real_t x) {
	return real_t((x >= 0) * x + (x < 0) * (exp(x) - 1));
}
static inline real_t selu_activate(real_t x) {
	return real_t((x >= 0) * 1.0507 * x + (x < 0) * 1.0507 * 1.6732 * (exp(x) - 1));
}
static inline real_t relie_activate(real_t x) {
	return real_t((x > 0) ? x : .01 * x);
}
static inline real_t ramp_activate(real_t x) {
	return real_t(x * (x > 0) + .1 * x);
}
static inline real_t leaky_activate(real_t x) {
	return real_t((x > 0) ? x : .1 * x);
}
static inline real_t tanh_activate(real_t x) {
	return real_t((exp(2 * x) - 1) / (exp(2 * x) + 1));
}
static inline real_t plse_activate(real_t x) {
	if (x < -4)
		return real_t(.01 * (x + 4));
	if (x > 4)
		return real_t(.01 * (x - 4) + 1);
	return real_t(.125 * x + .5);
}

static inline real_t lhtan_activate(real_t x) {
	if (x < 0)
		return real_t(.001 * x);
	if (x > 1)
		return real_t(.001 * (x - 1) + 1);
	return x;
}
static inline real_t lhtan_gradient(real_t x) {
	if (x > 0 && x < 1)
		return real_t(1);
	return real_t(.001);
}

static inline real_t hardtan_gradient(real_t x) {
	if (x > -1 && x < 1)
		return real_t(1);
	return real_t(0);
}
static inline real_t linear_gradient(real_t x) {
	return real_t(1);
}
static inline real_t logistic_gradient(real_t x) {
	return real_t((1 - x) * x);
}
static inline real_t loggy_gradient(real_t x) {
	real_t y = real_t((x + 1.) / 2.);
	return real_t(2 * (1 - y) * y);
}
static inline real_t stair_gradient(real_t x) {
	if (floor(x) == x)
		return real_t(0);
	return real_t(1);
}
static inline real_t relu_gradient(real_t x) {
	return real_t(x > 0);
}
static inline real_t elu_gradient(real_t x) {
	return real_t((x >= 0) + (x < 0) * (x + 1));
}
static inline real_t selu_gradient(real_t x) {
	return real_t((x >= 0) * 1.0507 + (x < 0) * (x + 1.0507 * 1.6732));
}
static inline real_t relie_gradient(real_t x) {
	return real_t((x > 0) ? 1 : .01);
}
static inline real_t ramp_gradient(real_t x) {
	return real_t((x > 0) + .1);
}
static inline real_t leaky_gradient(real_t x) {
	return real_t((x > 0) ? 1 : .1);
}
static inline real_t tanh_gradient(real_t x) {
	return real_t(1 - x * x);
}
static inline real_t plse_gradient(real_t x) {
	return real_t((x < 0 || x > 1) ? .01 : .125);
}

#endif

