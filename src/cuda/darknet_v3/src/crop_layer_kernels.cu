#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "crop_layer.h"
#include "utils.h"
#include "cuda.h"
#include "image.h"
#include "type.h"

}

__device__ real_t3 make_real_t3(real_t x, real_t y, real_t z) {
	real_t3 mem;
	mem.x = x;
	mem.y = y;
	mem.z = z;
	return mem;
}

__device__ real_t get_pixel_kernel(real_t *image, int w, int h, int x, int y,
		int c) {
	if (x < 0 || x >= w || y < 0 || y >= h)
		return 0;
	return image[x + w * (y + c * h)];
}

__device__ real_t3 rgb_to_hsv_kernel(real_t3 rgb) {
	real_t r = rgb.x;
	real_t g = rgb.y;
	real_t b = rgb.z;

	real_t h, s, v;
	real_t max = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
	real_t min = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);
	real_t delta = max - min;
	v = max;
	if (max == 0) {
		s = 0;
		h = -1;
	} else {
		s = delta / max;
		if (r == max) {
			h = (g - b) / delta;
		} else if (g == max) {
			h = 2 + (b - r) / delta;
		} else {
			h = 4 + (r - g) / delta;
		}
		if (h < 0)
			h += 6;
	}
	return make_real_t3(h, s, v);
}

__device__ real_t3 hsv_to_rgb_kernel(real_t3 hsv) {
	real_t h = hsv.x;
	real_t s = hsv.y;
	real_t v = hsv.z;

	real_t r, g, b;
	real_t f, p, q, t;

	if (s == 0) {
		r = g = b = v;
	} else {
		int index = (int) floorf(h);
		f = h - index;
		p = v * (1 - s);
		q = v * (1 - s * f);
		t = v * (1 - s * (1 - f));
		if (index == 0) {
			r = v;
			g = t;
			b = p;
		} else if (index == 1) {
			r = q;
			g = v;
			b = p;
		} else if (index == 2) {
			r = p;
			g = v;
			b = t;
		} else if (index == 3) {
			r = p;
			g = q;
			b = v;
		} else if (index == 4) {
			r = t;
			g = p;
			b = v;
		} else {
			r = v;
			g = p;
			b = q;
		}
	}
	r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
	g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
	b = (b < 0) ? 0 : ((b > 1) ? 1 : b);
	return make_real_t3(r, g, b);
}

__device__ real_t bilinear_interpolate_kernel(real_t *image, int w, int h,
		real_t x, real_t y, int c) {
	int ix = (int) floorf(x);
	int iy = (int) floorf(y);

	real_t dx = x - ix;
	real_t dy = y - iy;

	real_t val = (1 - dy) * (1 - dx) * get_pixel_kernel(image, w, h, ix, iy, c)
			+ dy * (1 - dx) * get_pixel_kernel(image, w, h, ix, iy + 1, c)
			+ (1 - dy) * dx * get_pixel_kernel(image, w, h, ix + 1, iy, c)
			+ dy * dx * get_pixel_kernel(image, w, h, ix + 1, iy + 1, c);
	return val;
}

__global__ void levels_image_kernel(real_t *image, real_t *rand, int batch,
		int w, int h, int train, real_t saturation, real_t exposure,
		real_t translate, real_t scale, real_t shift) {
	int size = batch * w * h;
	int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= size)
		return;
	int x = id % w;
	id /= w;
	int y = id % h;
	id /= h;
	real_t rshift = rand[0];
	real_t gshift = rand[1];
	real_t bshift = rand[2];
	real_t r0 = rand[8 * id + 0];
	real_t r1 = rand[8 * id + 1];
	real_t r2 = rand[8 * id + 2];
	real_t r3 = rand[8 * id + 3];

	saturation = r0 * (saturation - 1) + 1;
	saturation = (r1 > .5f) ? 1.f / saturation : saturation;
	exposure = r2 * (exposure - 1) + 1;
	exposure = (r3 > .5f) ? 1.f / exposure : exposure;

	size_t offset = id * h * w * 3;
	image += offset;
	real_t r = image[x + w * (y + h * 0)];
	real_t g = image[x + w * (y + h * 1)];
	real_t b = image[x + w * (y + h * 2)];
	real_t3 rgb = make_real_t3(r, g, b);
	if (train) {
		real_t3 hsv = rgb_to_hsv_kernel(rgb);
		hsv.y *= saturation;
		hsv.z *= exposure;
		rgb = hsv_to_rgb_kernel(hsv);
	} else {
		shift = 0;
	}
	image[x + w * (y + h * 0)] = rgb.x * scale + translate
			+ (rshift - .5f) * shift;
	image[x + w * (y + h * 1)] = rgb.y * scale + translate
			+ (gshift - .5f) * shift;
	image[x + w * (y + h * 2)] = rgb.z * scale + translate
			+ (bshift - .5f) * shift;
}

__global__ void forward_crop_layer_kernel(real_t *input, real_t *rand, int size,
		int c, int h, int w, int crop_height, int crop_width, int train,
		int flip, real_t angle, real_t *output) {
	int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= size)
		return;

	real_t cx = w / 2.f;
	real_t cy = h / 2.f;

	int count = id;
	int j = id % crop_width;
	id /= crop_width;
	int i = id % crop_height;
	id /= crop_height;
	int k = id % c;
	id /= c;
	int b = id;

	real_t r4 = rand[8 * b + 4];
	real_t r5 = rand[8 * b + 5];
	real_t r6 = rand[8 * b + 6];
	real_t r7 = rand[8 * b + 7];

	real_t dw = (w - crop_width) * r4;
	real_t dh = (h - crop_height) * r5;
	flip = (flip && (r6 > .5f));
	angle = 2 * angle * r7 - angle;
	if (!train) {
		dw = (w - crop_width) / 2.f;
		dh = (h - crop_height) / 2.f;
		flip = 0;
		angle = 0;
	}

	input += w * h * c * b;

	real_t x = (flip) ? w - dw - j - 1 : j + dw;
	real_t y = i + dh;

	real_t rx = cosf(angle) * (x - cx) - sinf(angle) * (y - cy) + cx;
	real_t ry = sinf(angle) * (x - cx) + cosf(angle) * (y - cy) + cy;

	output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k);
}

extern "C" void forward_crop_layer_gpu(crop_layer layer, network net) {
	cuda_random(layer.rand_gpu, layer.batch * 8);

	real_t radians = layer.angle * 3.14159265f / 180.f;

	real_t scale = 2;
	real_t translate = -1;
	if (layer.noadjust) {
		scale = 1;
		translate = 0;
	}

	int size = layer.batch * layer.w * layer.h;

	levels_image_kernel<<<cuda_gridsize(size), BLOCK, 0, net.st>>>(net.input_gpu,
			layer.rand_gpu, layer.batch, layer.w, layer.h, net.train,
			layer.saturation, layer.exposure, translate, scale, layer.shift);
	check_error(cudaPeekAtLastError());

	size = layer.batch * layer.c * layer.out_w * layer.out_h;

	forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK, 0, net.st>>>(net.input_gpu,
			layer.rand_gpu, size, layer.c, layer.h, layer.w, layer.out_h,
			layer.out_w, net.train, layer.flip, radians, layer.output_gpu);
	check_error(cudaPeekAtLastError());

	/*
	 cuda_pull_array(layer.output_gpu, layer.output, size);
	 image im = real_t_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 0*(size/layer.batch));
	 image im2 = real_t_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 1*(size/layer.batch));
	 image im3 = real_t_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 2*(size/layer.batch));

	 translate_image(im, -translate);
	 scale_image(im, 1/scale);
	 translate_image(im2, -translate);
	 scale_image(im2, 1/scale);
	 translate_image(im3, -translate);
	 scale_image(im3, 1/scale);

	 show_image(im, "cropped");
	 show_image(im2, "cropped2");
	 show_image(im3, "cropped3");
	 cvWaitKey(0);
	 */
}

