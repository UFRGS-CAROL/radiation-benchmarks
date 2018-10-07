#include "box.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int nms_comparator(const void *pa, const void *pb) {
	detection a = *(detection *) pa;
	detection b = *(detection *) pb;
	real_t diff = 0;
	if (b.sort_class >= 0) {
		diff = a.prob[b.sort_class] - b.prob[b.sort_class];
	} else {
		diff = a.objectness - b.objectness;
	}
	if (diff < 0)
		return 1;
	else if (diff > 0)
		return -1;
	return 0;
}

void do_nms_obj(detection *dets, int total, int classes, real_t thresh) {
	int i, j, k;
	k = total - 1;
	for (i = 0; i <= k; ++i) {
		if (dets[i].objectness == 0) {
			detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k + 1;

	for (i = 0; i < total; ++i) {
		dets[i].sort_class = -1;
	}

	qsort(dets, total, sizeof(detection), nms_comparator);
	for (i = 0; i < total; ++i) {
		if (dets[i].objectness == 0)
			continue;
		box a = dets[i].bbox;
		for (j = i + 1; j < total; ++j) {
			if (dets[j].objectness == 0)
				continue;
			box b = dets[j].bbox;
			if (box_iou(a, b) > thresh) {
				dets[j].objectness = 0;
				for (k = 0; k < classes; ++k) {
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}

void do_nms_sort(detection *dets, int total, int classes, real_t thresh) {
	int i, j, k;
	k = total - 1;
	for (i = 0; i <= k; ++i) {
		if (dets[i].objectness == 0) {
			detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k + 1;

	for (k = 0; k < classes; ++k) {
		for (i = 0; i < total; ++i) {
			dets[i].sort_class = k;
		}
		qsort(dets, total, sizeof(detection), nms_comparator);
		for (i = 0; i < total; ++i) {
			if (dets[i].prob[k] == 0)
				continue;
			box a = dets[i].bbox;
			for (j = i + 1; j < total; ++j) {
				box b = dets[j].bbox;
				if (box_iou(a, b) > thresh) {
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}

box real_t_to_box(real_t *f, int stride) {
	box b = { 0 };
	b.x = f[0];
	b.y = f[1 * stride];
	b.w = f[2 * stride];
	b.h = f[3 * stride];
	return b;
}

dbox derivative(box a, box b) {
	dbox d;
	d.dx = 0;
	d.dw = 0;
	real_t l1 = a.x - a.w / 2;
	real_t l2 = b.x - b.w / 2;
	if (l1 > l2) {
		d.dx -= 1;
		d.dw += .5;
	}
	real_t r1 = a.x + a.w / 2;
	real_t r2 = b.x + b.w / 2;
	if (r1 < r2) {
		d.dx += 1;
		d.dw += .5;
	}
	if (l1 > r2) {
		d.dx = -1;
		d.dw = 0;
	}
	if (r1 < l2) {
		d.dx = 1;
		d.dw = 0;
	}

	d.dy = 0;
	d.dh = 0;
	real_t t1 = a.y - a.h / 2;
	real_t t2 = b.y - b.h / 2;
	if (t1 > t2) {
		d.dy -= 1;
		d.dh += .5;
	}
	real_t b1 = a.y + a.h / 2;
	real_t b2 = b.y + b.h / 2;
	if (b1 < b2) {
		d.dy += 1;
		d.dh += .5;
	}
	if (t1 > b2) {
		d.dy = -1;
		d.dh = 0;
	}
	if (b1 < t2) {
		d.dy = 1;
		d.dh = 0;
	}
	return d;
}

real_t overlap(real_t x1, real_t w1, real_t x2, real_t w2) {
	real_t l1 = x1 - w1 / 2;
	real_t l2 = x2 - w2 / 2;
	real_t left = l1 > l2 ? l1 : l2;
	real_t r1 = x1 + w1 / 2;
	real_t r2 = x2 + w2 / 2;
	real_t right = r1 < r2 ? r1 : r2;
	return right - left;
}

real_t box_intersection(box a, box b) {
	real_t w = overlap(a.x, a.w, b.x, b.w);
	real_t h = overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0)
		return 0;
	real_t area = w * h;
	return area;
}

real_t box_union(box a, box b) {
	real_t i = box_intersection(a, b);
	real_t u = a.w * a.h + b.w * b.h - i;
	return u;
}

real_t box_iou(box a, box b) {
	return box_intersection(a, b) / box_union(a, b);
}

real_t box_rmse(box a, box b) {
	return sqrt(
			pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.w - b.w, 2)
					+ pow(a.h - b.h, 2));
}

dbox dintersect(box a, box b) {
	real_t w = overlap(a.x, a.w, b.x, b.w);
	real_t h = overlap(a.y, a.h, b.y, b.h);
	dbox dover = derivative(a, b);
	dbox di;

	di.dw = dover.dw * h;
	di.dx = dover.dx * h;
	di.dh = dover.dh * w;
	di.dy = dover.dy * w;

	return di;
}

dbox dunion(box a, box b) {
	dbox du;

	dbox di = dintersect(a, b);
	du.dw = a.h - di.dw;
	du.dh = a.w - di.dh;
	du.dx = -di.dx;
	du.dy = -di.dy;

	return du;
}

void test_dunion() {
	box a = { 0, 0, 1, 1 };
	box dxa = { 0 + .0001, 0, 1, 1 };
	box dya = { 0, 0 + .0001, 1, 1 };
	box dwa = { 0, 0, 1 + .0001, 1 };
	box dha = { 0, 0, 1, 1 + .0001 };

	box b = { .5, .5, .2, .2 };
	dbox di = dunion(a, b);
	printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
	real_t inter = box_union(a, b);
	real_t xinter = box_union(dxa, b);
	real_t yinter = box_union(dya, b);
	real_t winter = box_union(dwa, b);
	real_t hinter = box_union(dha, b);
	xinter = (xinter - inter) / (.0001);
	yinter = (yinter - inter) / (.0001);
	winter = (winter - inter) / (.0001);
	hinter = (hinter - inter) / (.0001);
	printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}
void test_dintersect() {
	box a = { 0, 0, 1, 1 };
	box dxa = { 0 + .0001, 0, 1, 1 };
	box dya = { 0, 0 + .0001, 1, 1 };
	box dwa = { 0, 0, 1 + .0001, 1 };
	box dha = { 0, 0, 1, 1 + .0001 };

	box b = { .5, .5, .2, .2 };
	dbox di = dintersect(a, b);
	printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
	real_t inter = box_intersection(a, b);
	real_t xinter = box_intersection(dxa, b);
	real_t yinter = box_intersection(dya, b);
	real_t winter = box_intersection(dwa, b);
	real_t hinter = box_intersection(dha, b);
	xinter = (xinter - inter) / (.0001);
	yinter = (yinter - inter) / (.0001);
	winter = (winter - inter) / (.0001);
	hinter = (hinter - inter) / (.0001);
	printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_box() {
	test_dintersect();
	test_dunion();
	box a = { 0, 0, 1, 1 };
	box dxa = { 0 + .00001, 0, 1, 1 };
	box dya = { 0, 0 + .00001, 1, 1 };
	box dwa = { 0, 0, 1 + .00001, 1 };
	box dha = { 0, 0, 1, 1 + .00001 };

	box b = { .5, 0, .2, .2 };

	real_t iou = box_iou(a, b);
	iou = (1 - iou) * (1 - iou);
	printf("%f\n", iou);
	dbox d = diou(a, b);
	printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

	real_t xiou = box_iou(dxa, b);
	real_t yiou = box_iou(dya, b);
	real_t wiou = box_iou(dwa, b);
	real_t hiou = box_iou(dha, b);
	xiou = ((1 - xiou) * (1 - xiou) - iou) / (.00001);
	yiou = ((1 - yiou) * (1 - yiou) - iou) / (.00001);
	wiou = ((1 - wiou) * (1 - wiou) - iou) / (.00001);
	hiou = ((1 - hiou) * (1 - hiou) - iou) / (.00001);
	printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}

dbox diou(box a, box b) {
	real_t u = box_union(a, b);
	real_t i = box_intersection(a, b);
	dbox di = dintersect(a, b);
	dbox du = dunion(a, b);
	dbox dd = { 0, 0, 0, 0 };

	if (i <= 0 || 1) {
		dd.dx = b.x - a.x;
		dd.dy = b.y - a.y;
		dd.dw = b.w - a.w;
		dd.dh = b.h - a.h;
		return dd;
	}

	dd.dx = 2 * pow((1 - (i / u)), 1) * (di.dx * u - du.dx * i) / (u * u);
	dd.dy = 2 * pow((1 - (i / u)), 1) * (di.dy * u - du.dy * i) / (u * u);
	dd.dw = 2 * pow((1 - (i / u)), 1) * (di.dw * u - du.dw * i) / (u * u);
	dd.dh = 2 * pow((1 - (i / u)), 1) * (di.dh * u - du.dh * i) / (u * u);
	return dd;
}

void do_nms(box *boxes, real_t **probs, int total, int classes, real_t thresh) {
	int i, j, k;
	for (i = 0; i < total; ++i) {
		int any = 0;
		for (k = 0; k < classes; ++k)
			any = any || (probs[i][k] > 0);
		if (!any) {
			continue;
		}
		for (j = i + 1; j < total; ++j) {
			if (box_iou(boxes[i], boxes[j]) > thresh) {
				for (k = 0; k < classes; ++k) {
					if (probs[i][k] < probs[j][k])
						probs[i][k] = 0;
					else
						probs[j][k] = 0;
				}
			}
		}
	}
}

box encode_box(box b, box anchor) {
	box encode;
	encode.x = (b.x - anchor.x) / anchor.w;
	encode.y = (b.y - anchor.y) / anchor.h;
	encode.w = log2(b.w / anchor.w);
	encode.h = log2(b.h / anchor.h);
	return encode;
}

box decode_box(box b, box anchor) {
	box decode;
	decode.x = b.x * anchor.w + anchor.x;
	decode.y = b.y * anchor.h + anchor.y;
	decode.w = pow(2., b.w) * anchor.w;
	decode.h = pow(2., b.h) * anchor.h;
	return decode;
}
