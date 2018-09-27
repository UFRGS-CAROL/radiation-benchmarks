#include "data.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

list *get_paths(char *filename) {
	char *path;
	FILE *file = fopen(filename, "r");
	if (!file)
		file_error(filename);
	list *lines = make_list();
	while ((path = fgetl(file))) {
		list_insert(lines, path);
	}
	fclose(file);
	return lines;
}

/*
 char **get_random_paths_indexes(char **paths, int n, int m, int *indexes)
 {
 char **random_paths = calloc(n, sizeof(char*));
 int i;
 pthread_mutex_lock(&mutex);
 for(i = 0; i < n; ++i){
 int index = rand()%m;
 indexes[i] = index;
 random_paths[i] = paths[index];
 if(i == 0) printf("%s\n", paths[index]);
 }
 pthread_mutex_unlock(&mutex);
 return random_paths;
 }
 */

char **get_random_paths(char **paths, int n, int m) {
	char **random_paths = (char**) calloc(n, sizeof(char*));
	int i;
	pthread_mutex_lock(&mutex);
	for (i = 0; i < n; ++i) {
		int index = rand() % m;
		random_paths[i] = paths[index];
		//if(i == 0) printf("%s\n", paths[index]);
	}
	pthread_mutex_unlock(&mutex);
	return random_paths;
}

char **find_replace_paths(char **paths, int n, char *find, char *replace) {
	char **replace_paths = (char**) calloc(n, sizeof(char*));
	int i;
	for (i = 0; i < n; ++i) {
		char replaced[4096];
		find_replace(paths[i], find, replace, replaced);
		replace_paths[i] = copy_string(replaced);
	}
	return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h) {
	int i;
	matrix X;
	X.rows = n;
	X.vals = (real_t**) calloc(X.rows, sizeof(real_t*));
	X.cols = 0;

	for (i = 0; i < n; ++i) {
		image im = load_image(paths[i], w, h, 3);

		image gray = grayscale_image(im);
		free_image(im);
		im = gray;

		X.vals[i] = im.data;
		X.cols = im.h * im.w * im.c;
	}
	return X;
}

matrix load_image_paths(char **paths, int n, int w, int h) {
	int i;
	matrix X;
	X.rows = n;
	X.vals = (real_t**) calloc(X.rows, sizeof(real_t*));
	X.cols = 0;

	for (i = 0; i < n; ++i) {
		image im = load_image_color(paths[i], w, h);
		X.vals[i] = im.data;
		X.cols = im.h * im.w * im.c;
	}
	return X;
}

matrix load_image_augment_paths(char **paths, int n, int min, int max, int size,
		real_t angle, real_t aspect, real_t hue, real_t saturation,
		real_t exposure, int center) {
	int i;
	matrix X;
	X.rows = n;
	X.vals = (real_t**) calloc(X.rows, sizeof(real_t*));
	X.cols = 0;

	for (i = 0; i < n; ++i) {
		image im = load_image_color(paths[i], 0, 0);
		image crop;
		if (center) {
			crop = center_crop_image(im, size, size);
		} else {
			crop = random_augment_image(im, angle, aspect, min, max, size,
					size);
		}
		int flip = rand() % 2;
		if (flip)
			flip_image(crop);
		random_distort_image(crop, hue, saturation, exposure);

		/*
		 show_image(im, "orig");
		 show_image(crop, "crop");
		 cvWaitKey(0);
		 */
		//grayscale_image_3c(crop);
		free_image(im);
		X.vals[i] = crop.data;
		X.cols = crop.h * crop.w * crop.c;
	}
	return X;
}

box_label *read_boxes(char *filename, int *n) {
	FILE *file = fopen(filename, "r");
	if (!file)
		file_error(filename);
	//TODO:
	// make it ready to accept all float precisions
	//real_t x, y, h, w;
	float x, y, h, w;

	int id;
	int count = 0;
	int size = 64;
	box_label *boxes = (box_label*) calloc(size, sizeof(box_label));
	while (fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5) {

		if (count == size) {
			size = size * 2;
			boxes = (box_label*) realloc(boxes, size * sizeof(box_label));
		}
		boxes[count].id = id;
		boxes[count].x = x;
		boxes[count].y = y;
		boxes[count].h = h;
		boxes[count].w = w;
		boxes[count].left = x - w / 2;
		boxes[count].right = x + w / 2;
		boxes[count].top = y - h / 2;
		boxes[count].bottom = y + h / 2;
		++count;
	}
	fclose(file);
	*n = count;
	return boxes;
}

void randomize_boxes(box_label *b, int n) {
	int i;
	for (i = 0; i < n; ++i) {
		box_label swap = b[i];
		int index = rand() % n;
		b[i] = b[index];
		b[index] = swap;
	}
}

void correct_boxes(box_label *boxes, int n, real_t dx, real_t dy, real_t sx,
		real_t sy, int flip) {
	int i;
	for (i = 0; i < n; ++i) {
		if (boxes[i].x == 0 && boxes[i].y == 0) {
			boxes[i].x = 999999;
			boxes[i].y = 999999;
			boxes[i].w = 999999;
			boxes[i].h = 999999;
			continue;
		}
		boxes[i].left = boxes[i].left * sx - dx;
		boxes[i].right = boxes[i].right * sx - dx;
		boxes[i].top = boxes[i].top * sy - dy;
		boxes[i].bottom = boxes[i].bottom * sy - dy;

		if (flip) {
			real_t swap = boxes[i].left;
			boxes[i].left = 1. - boxes[i].right;
			boxes[i].right = 1. - swap;
		}

		boxes[i].left = constrain(real_t(0), real_t(1), boxes[i].left);
		boxes[i].right = constrain(real_t(0), real_t(1), boxes[i].right);
		boxes[i].top = constrain(real_t(0), real_t(1), boxes[i].top);
		boxes[i].bottom = constrain(real_t(0), real_t(1), boxes[i].bottom);

		boxes[i].x = (boxes[i].left + boxes[i].right) / 2;
		boxes[i].y = (boxes[i].top + boxes[i].bottom) / 2;
		boxes[i].w = (boxes[i].right - boxes[i].left);
		boxes[i].h = (boxes[i].bottom - boxes[i].top);

		boxes[i].w = constrain(real_t(0), real_t(1), boxes[i].w);
		boxes[i].h = constrain(real_t(0), real_t(1), boxes[i].h);
	}
}

void fill_truth_swag(char *path, real_t *truth, int classes, int flip,
		real_t dx, real_t dy, real_t sx, real_t sy) {
	char labelpath[4096];
	find_replace(path, "images", "labels", labelpath);
	find_replace(labelpath, "JPEGImages", "labels", labelpath);
	find_replace(labelpath, ".jpg", ".txt", labelpath);
	find_replace(labelpath, ".JPG", ".txt", labelpath);
	find_replace(labelpath, ".JPEG", ".txt", labelpath);

	int count = 0;
	box_label *boxes = read_boxes(labelpath, &count);
	randomize_boxes(boxes, count);
	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	real_t x, y, w, h;
	int id;
	int i;

	for (i = 0; i < count && i < 90; ++i) {
		x = boxes[i].x;
		y = boxes[i].y;
		w = boxes[i].w;
		h = boxes[i].h;
		id = boxes[i].id;

		if (w < .0 || h < .0)
			continue;

		int index = (4 + classes) * i;

		truth[index++] = x;
		truth[index++] = y;
		truth[index++] = w;
		truth[index++] = h;

		if (id < classes)
			truth[index + id] = 1;
	}
	free(boxes);
}

void fill_truth_region(char *path, real_t *truth, int classes, int num_boxes,
		int flip, real_t dx, real_t dy, real_t sx, real_t sy) {
	char labelpath[4096];
	find_replace(path, "images", "labels", labelpath);
	find_replace(labelpath, "JPEGImages", "labels", labelpath);

	find_replace(labelpath, ".jpg", ".txt", labelpath);
	find_replace(labelpath, ".png", ".txt", labelpath);
	find_replace(labelpath, ".JPG", ".txt", labelpath);
	find_replace(labelpath, ".JPEG", ".txt", labelpath);
	int count = 0;
	box_label *boxes = read_boxes(labelpath, &count);
	randomize_boxes(boxes, count);
	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	real_t x, y, w, h;
	int id;
	int i;

	for (i = 0; i < count; ++i) {
		x = boxes[i].x;
		y = boxes[i].y;
		w = boxes[i].w;
		h = boxes[i].h;
		id = boxes[i].id;

		if (w < .005 || h < .005)
			continue;

		int col = (int) (x * num_boxes);
		int row = (int) (y * num_boxes);

		x = x * num_boxes - col;
		y = y * num_boxes - row;

		int index = (col + row * num_boxes) * (5 + classes);
		if (truth[index])
			continue;
		truth[index++] = 1;

		if (id < classes)
			truth[index + id] = 1;
		index += classes;

		truth[index++] = x;
		truth[index++] = y;
		truth[index++] = w;
		truth[index++] = h;
	}
	free(boxes);
}

void load_rle(image im, int *rle, int n) {
	int count = 0;
	int curr = 0;
	int i, j;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < rle[i]; ++j) {
			im.data[count++] = curr;
		}
		curr = 1 - curr;
	}
	for (; count < im.h * im.w * im.c; ++count) {
		im.data[count] = curr;
	}
}

void or_image(image src, image dest, int c) {
	int i;
	for (i = 0; i < src.w * src.h; ++i) {
		if (src.data[i])
			dest.data[dest.w * dest.h * c + i] = 1;
	}
}

void exclusive_image(image src) {
	int k, j, i;
	int s = src.w * src.h;
	for (k = 0; k < src.c - 1; ++k) {
		for (i = 0; i < s; ++i) {
			if (src.data[k * s + i]) {
				for (j = k + 1; j < src.c; ++j) {
					src.data[j * s + i] = 0;
				}
			}
		}
	}
}

box bound_image(image im) {
	int x, y;
	int minx = im.w;
	int miny = im.h;
	int maxx = 0;
	int maxy = 0;
	for (y = 0; y < im.h; ++y) {
		for (x = 0; x < im.w; ++x) {
			if (im.data[y * im.w + x]) {
				minx = (x < minx) ? x : minx;
				miny = (y < miny) ? y : miny;
				maxx = (x > maxx) ? x : maxx;
				maxy = (y > maxy) ? y : maxy;
			}
		}
	}
	box b = { real_t(minx), real_t(miny), real_t(maxx - minx + 1), real_t(maxy - miny + 1) };
	//printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
	return b;
}

void fill_truth_iseg(char *path, int num_boxes, real_t *truth, int classes,
		int w, int h, augment_args aug, int flip, int mw, int mh) {
	char labelpath[4096];
	find_replace(path, "images", "mask", labelpath);
	find_replace(labelpath, "JPEGImages", "mask", labelpath);
	find_replace(labelpath, ".jpg", ".txt", labelpath);
	find_replace(labelpath, ".JPG", ".txt", labelpath);
	find_replace(labelpath, ".JPEG", ".txt", labelpath);
	FILE *file = fopen(labelpath, "r");
	if (!file)
		file_error(labelpath);
	char buff[32788];
	int id;
	int i = 0;
	int j;
	image part = make_image(w, h, 1);
	while ((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes) {
		int n = 0;
		int *rle = read_intlist(buff, &n, 0);
		load_rle(part, rle, n);
		image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h,
				aug.dx, aug.dy, aug.aspect);
		if (flip)
			flip_image(sized);

		image mask = resize_image(sized, mw, mh);
		truth[i * (mw * mh + 1)] = id;
		for (j = 0; j < mw * mh; ++j) {
			truth[i * (mw * mh + 1) + 1 + j] = mask.data[j];
		}
		++i;

		free_image(mask);
		free_image(sized);
		free(rle);
	}
	if (i < num_boxes)
		truth[i * (mw * mh + 1)] = -1;
	fclose(file);
	free_image(part);
}

void fill_truth_mask(char *path, int num_boxes, real_t *truth, int classes,
		int w, int h, augment_args aug, int flip, int mw, int mh) {
	char labelpath[4096];
	find_replace(path, "images", "mask", labelpath);
	find_replace(labelpath, "JPEGImages", "mask", labelpath);
	find_replace(labelpath, ".jpg", ".txt", labelpath);
	find_replace(labelpath, ".JPG", ".txt", labelpath);
	find_replace(labelpath, ".JPEG", ".txt", labelpath);
	FILE *file = fopen(labelpath, "r");
	if (!file)
		file_error(labelpath);
	char buff[32788];
	int id;
	int i = 0;
	image part = make_image(w, h, 1);
	while ((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes) {
		int n = 0;
		int *rle = read_intlist(buff, &n, 0);
		load_rle(part, rle, n);
		image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h,
				aug.dx, aug.dy, aug.aspect);
		if (flip)
			flip_image(sized);
		box b = bound_image(sized);
		if (b.w > 0) {
			image crop = crop_image(sized, b.x, b.y, b.w, b.h);
			image mask = resize_image(crop, mw, mh);
			truth[i * (4 + mw * mh + 1) + 0] = (b.x + b.w / 2.) / sized.w;
			truth[i * (4 + mw * mh + 1) + 1] = (b.y + b.h / 2.) / sized.h;
			truth[i * (4 + mw * mh + 1) + 2] = b.w / sized.w;
			truth[i * (4 + mw * mh + 1) + 3] = b.h / sized.h;
			int j;
			for (j = 0; j < mw * mh; ++j) {
				truth[i * (4 + mw * mh + 1) + 4 + j] = mask.data[j];
			}
			truth[i * (4 + mw * mh + 1) + 4 + mw * mh] = id;
			free_image(crop);
			free_image(mask);
			++i;
		}
		free_image(sized);
		free(rle);
	}
	fclose(file);
	free_image(part);
}

void fill_truth_detection(char *path, int num_boxes, real_t *truth, int classes,
		int flip, real_t dx, real_t dy, real_t sx, real_t sy) {
	char labelpath[4096];
	find_replace(path, "images", "labels", labelpath);
	find_replace(labelpath, "JPEGImages", "labels", labelpath);

	find_replace(labelpath, "raw", "labels", labelpath);
	find_replace(labelpath, ".jpg", ".txt", labelpath);
	find_replace(labelpath, ".png", ".txt", labelpath);
	find_replace(labelpath, ".JPG", ".txt", labelpath);
	find_replace(labelpath, ".JPEG", ".txt", labelpath);
	int count = 0;
	box_label *boxes = read_boxes(labelpath, &count);
	randomize_boxes(boxes, count);
	correct_boxes(boxes, count, dx, dy, sx, sy, flip);
	if (count > num_boxes)
		count = num_boxes;
	real_t x, y, w, h;
	int id;
	int i;
	int sub = 0;

	for (i = 0; i < count; ++i) {
		x = boxes[i].x;
		y = boxes[i].y;
		w = boxes[i].w;
		h = boxes[i].h;
		id = boxes[i].id;

		if ((w < .001 || h < .001)) {
			++sub;
			continue;
		}

		truth[(i - sub) * 5 + 0] = x;
		truth[(i - sub) * 5 + 1] = y;
		truth[(i - sub) * 5 + 2] = w;
		truth[(i - sub) * 5 + 3] = h;
		truth[(i - sub) * 5 + 4] = id;
	}
	free(boxes);
}

#define NUMCHARS 37

void print_letters(real_t *pred, int n) {
	int i;
	for (i = 0; i < n; ++i) {
		int index = max_index(pred + i * NUMCHARS, NUMCHARS);
		printf("%c", int_to_alphanum(index));
	}
	printf("\n");
}

void fill_truth_captcha(char *path, int n, real_t *truth) {
	char *begin = strrchr(path, '/');
	++begin;
	int i;
	for (i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i) {
		int index = alphanum_to_int(begin[i]);
		if (index > 35)
			printf("Bad %c\n", begin[i]);
		truth[i * NUMCHARS + index] = 1;
	}
	for (; i < n; ++i) {
		truth[i * NUMCHARS + NUMCHARS - 1] = 1;
	}
}

data load_data_captcha(char **paths, int n, int m, int k, int w, int h) {
	if (m)
		paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_paths(paths, n, w, h);
	d.y = make_matrix(n, k * NUMCHARS);
	int i;
	for (i = 0; i < n; ++i) {
		fill_truth_captcha(paths[i], k, d.y.vals[i]);
	}
	if (m)
		free(paths);
	return d;
}

data load_data_captcha_encode(char **paths, int n, int m, int w, int h) {
	if (m)
		paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_paths(paths, n, w, h);
	d.X.cols = 17100;
	d.y = d.X;
	if (m)
		free(paths);
	return d;
}

void fill_truth(char *path, char **labels, int k, real_t *truth) {
	int i;
	memset(truth, 0, k * sizeof(real_t));
	int count = 0;
	for (i = 0; i < k; ++i) {
		if (strstr(path, labels[i])) {
			truth[i] = 1;
			++count;
			//printf("%s %s %d\n", path, labels[i], i);
		}
	}
	if (count != 1 && (k != 1 || count != 0))
		printf("Too many or too few labels: %d, %s\n", count, path);
}

void fill_hierarchy(real_t *truth, int k, tree *hierarchy) {
	int j;
	for (j = 0; j < k; ++j) {
		if (truth[j]) {
			int parent = hierarchy->parent[j];
			while (parent >= 0) {
				truth[parent] = 1;
				parent = hierarchy->parent[parent];
			}
		}
	}
	int i;
	int count = 0;
	for (j = 0; j < hierarchy->groups; ++j) {
		//printf("%d\n", count);
		int mask = 1;
		for (i = 0; i < hierarchy->group_size[j]; ++i) {
			if (truth[count + i]) {
				mask = 0;
				break;
			}
		}
		if (mask) {
			for (i = 0; i < hierarchy->group_size[j]; ++i) {
				truth[count + i] = SECRET_NUM;
			}
		}
		count += hierarchy->group_size[j];
	}
}

matrix load_regression_labels_paths(char **paths, int n, int k) {
	matrix y = make_matrix(n, k);
	int i, j;
	for (i = 0; i < n; ++i) {
		char labelpath[4096];
		find_replace(paths[i], "images", "labels", labelpath);
		find_replace(labelpath, "JPEGImages", "labels", labelpath);
		find_replace(labelpath, ".BMP", ".txt", labelpath);
		find_replace(labelpath, ".JPEG", ".txt", labelpath);
		find_replace(labelpath, ".JPG", ".txt", labelpath);
		find_replace(labelpath, ".JPeG", ".txt", labelpath);
		find_replace(labelpath, ".Jpeg", ".txt", labelpath);
		find_replace(labelpath, ".PNG", ".txt", labelpath);
		find_replace(labelpath, ".TIF", ".txt", labelpath);
		find_replace(labelpath, ".bmp", ".txt", labelpath);
		find_replace(labelpath, ".jpeg", ".txt", labelpath);
		find_replace(labelpath, ".jpg", ".txt", labelpath);
		find_replace(labelpath, ".png", ".txt", labelpath);
		find_replace(labelpath, ".tif", ".txt", labelpath);

		FILE *file = fopen(labelpath, "r");
		for (j = 0; j < k; ++j) {
			//TODO: make ready to all precisions
			float tmp;
			fscanf(file, "%f", &(tmp));
			y.vals[i][j] = tmp;
		}
		fclose(file);
	}
	return y;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k,
		tree *hierarchy) {
	matrix y = make_matrix(n, k);
	int i;
	for (i = 0; i < n && labels; ++i) {
		fill_truth(paths[i], labels, k, y.vals[i]);
		if (hierarchy) {
			fill_hierarchy(y.vals[i], k, hierarchy);
		}
	}
	return y;
}

matrix load_tags_paths(char **paths, int n, int k) {
	matrix y = make_matrix(n, k);
	int i;
	//int count = 0;
	for (i = 0; i < n; ++i) {
		char label[4096];
		find_replace(paths[i], "images", "labels", label);
		find_replace(label, ".jpg", ".txt", label);
		FILE *file = fopen(label, "r");
		if (!file)
			continue;
		//++count;
		int tag;
		while (fscanf(file, "%d", &tag) == 1) {
			if (tag < k) {
				y.vals[i][tag] = 1;
			}
		}
		fclose(file);
	}
	//printf("%d/%d\n", count, n);
	return y;
}

char **get_labels(char *filename) {
	list *plist = get_paths(filename);
	char **labels = (char **) list_to_array(plist);
	free_list(plist);
	return labels;
}

void free_data(data d) {
	if (!d.shallow) {
		free_matrix(d.X);
		free_matrix(d.y);
	} else {
		free(d.X.vals);
		free(d.y.vals);
	}
}

image get_segmentation_image(char *path, int w, int h, int classes) {
	char labelpath[4096];
	find_replace(path, "images", "mask", labelpath);
	find_replace(labelpath, "JPEGImages", "mask", labelpath);
	find_replace(labelpath, ".jpg", ".txt", labelpath);
	find_replace(labelpath, ".JPG", ".txt", labelpath);
	find_replace(labelpath, ".JPEG", ".txt", labelpath);
	image mask = make_image(w, h, classes);
	FILE *file = fopen(labelpath, "r");
	if (!file)
		file_error(labelpath);
	char buff[32788];
	int id;
	image part = make_image(w, h, 1);
	while (fscanf(file, "%d %s", &id, buff) == 2) {
		int n = 0;
		int *rle = read_intlist(buff, &n, 0);
		load_rle(part, rle, n);
		or_image(part, mask, id);
		free(rle);
	}
	//exclusive_image(mask);
	fclose(file);
	free_image(part);
	return mask;
}

image get_segmentation_image2(char *path, int w, int h, int classes) {
	char labelpath[4096];
	find_replace(path, "images", "mask", labelpath);
	find_replace(labelpath, "JPEGImages", "mask", labelpath);
	find_replace(labelpath, ".jpg", ".txt", labelpath);
	find_replace(labelpath, ".JPG", ".txt", labelpath);
	find_replace(labelpath, ".JPEG", ".txt", labelpath);
	image mask = make_image(w, h, classes + 1);
	int i;
	for (i = 0; i < w * h; ++i) {
		mask.data[w * h * classes + i] = 1;
	}
	FILE *file = fopen(labelpath, "r");
	if (!file)
		file_error(labelpath);
	char buff[32788];
	int id;
	image part = make_image(w, h, 1);
	while (fscanf(file, "%d %s", &id, buff) == 2) {
		int n = 0;
		int *rle = read_intlist(buff, &n, 0);
		load_rle(part, rle, n);
		or_image(part, mask, id);
		for (i = 0; i < w * h; ++i) {
			if (part.data[i])
				mask.data[w * h * classes + i] = 0;
		}
		free(rle);
	}
	//exclusive_image(mask);
	fclose(file);
	free_image(part);
	return mask;
}

data load_data_seg(int n, char **paths, int m, int w, int h, int classes,
		int min, int max, real_t angle, real_t aspect, real_t hue,
		real_t saturation, real_t exposure, int div) {
	char **random_paths = get_random_paths(paths, n, m);
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (real_t**) calloc(d.X.rows, sizeof(real_t*));
	d.X.cols = h * w * 3;

	d.y.rows = n;
	d.y.cols = h * w * classes / div / div;
	d.y.vals = (real_t**) calloc(d.X.rows, sizeof(real_t*));

	for (i = 0; i < n; ++i) {
		image orig = load_image_color(random_paths[i], 0, 0);
		augment_args a = random_augment_args(orig, angle, aspect, min, max, w,
				h);
		image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx,
				a.dy, a.aspect);

		int flip = rand() % 2;
		if (flip)
			flip_image(sized);
		random_distort_image(sized, hue, saturation, exposure);
		d.X.vals[i] = sized.data;

		image mask = get_segmentation_image(random_paths[i], orig.w, orig.h,
				classes);
		//image mask = make_image(orig.w, orig.h, classes+1);
		image sized_m = rotate_crop_image(mask, a.rad, real_t(a.scale / div), real_t(a.w / div),
				a.h / div, real_t(a.dx / div), real_t(a.dy / div), a.aspect);

		if (flip)
			flip_image(sized_m);
		d.y.vals[i] = sized_m.data;

		free_image(orig);
		free_image(mask);

		/*
		 image rgb = mask_to_rgb(sized_m, classes);
		 show_image(rgb, "part");
		 show_image(sized, "orig");
		 cvWaitKey(0);
		 free_image(rgb);
		 */
	}
	free(random_paths);
	return d;
}

data load_data_iseg(int n, char **paths, int m, int w, int h, int classes,
		int boxes, int div, int min, int max, real_t angle, real_t aspect,
		real_t hue, real_t saturation, real_t exposure) {
	char **random_paths = get_random_paths(paths, n, m);
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (real_t**) calloc(d.X.rows, sizeof(real_t*));
	d.X.cols = h * w * 3;

	d.y = make_matrix(n, (((w / div) * (h / div)) + 1) * boxes);

	for (i = 0; i < n; ++i) {
		image orig = load_image_color(random_paths[i], 0, 0);
		augment_args a = random_augment_args(orig, angle, aspect, min, max, w,
				h);
		image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx,
				a.dy, a.aspect);

		int flip = rand() % 2;
		if (flip)
			flip_image(sized);
		random_distort_image(sized, hue, saturation, exposure);
		d.X.vals[i] = sized.data;
		//show_image(sized, "image");

		fill_truth_iseg(random_paths[i], boxes, d.y.vals[i], classes, orig.w,
				orig.h, a, flip, w / div, h / div);

		free_image(orig);

		/*
		 image rgb = mask_to_rgb(sized_m, classes);
		 show_image(rgb, "part");
		 show_image(sized, "orig");
		 cvWaitKey(0);
		 free_image(rgb);
		 */
	}
	free(random_paths);
	return d;
}

data load_data_mask(int n, char **paths, int m, int w, int h, int classes,
		int boxes, int coords, int min, int max, real_t angle, real_t aspect,
		real_t hue, real_t saturation, real_t exposure) {
	char **random_paths = get_random_paths(paths, n, m);
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (real_t**) calloc(d.X.rows, sizeof(real_t*));
	d.X.cols = h * w * 3;

	d.y = make_matrix(n, (coords + 1) * boxes);

	for (i = 0; i < n; ++i) {
		image orig = load_image_color(random_paths[i], 0, 0);
		augment_args a = random_augment_args(orig, angle, aspect, min, max, w,
				h);
		image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx,
				a.dy, a.aspect);

		int flip = rand() % 2;
		if (flip)
			flip_image(sized);
		random_distort_image(sized, hue, saturation, exposure);
		d.X.vals[i] = sized.data;
		//show_image(sized, "image");

		fill_truth_mask(random_paths[i], boxes, d.y.vals[i], classes, orig.w,
				orig.h, a, flip, 14, 14);

		free_image(orig);

		/*
		 image rgb = mask_to_rgb(sized_m, classes);
		 show_image(rgb, "part");
		 show_image(sized, "orig");
		 cvWaitKey(0);
		 free_image(rgb);
		 */
	}
	free(random_paths);
	return d;
}

data load_data_region(int n, char **paths, int m, int w, int h, int size,
		int classes, real_t jitter, real_t hue, real_t saturation,
		real_t exposure) {
	char **random_paths = get_random_paths(paths, n, m);
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (real_t**) calloc(d.X.rows, sizeof(real_t*));
	d.X.cols = h * w * 3;

	int k = size * size * (5 + classes);
	d.y = make_matrix(n, k);
	for (i = 0; i < n; ++i) {
		image orig = load_image_color(random_paths[i], 0, 0);

		int oh = orig.h;
		int ow = orig.w;

		int dw = (ow * jitter);
		int dh = (oh * jitter);

		int pleft = rand_uniform(real_t(-dw), real_t(dw));
		int pright = rand_uniform(real_t(-dw), real_t(dw));
		int ptop = rand_uniform(real_t(-dh), real_t(dh));
		int pbot = rand_uniform(real_t(-dh), real_t(dh));

		int swidth = ow - pleft - pright;
		int sheight = oh - ptop - pbot;

		real_t sx = real_t(swidth / ow);
		real_t sy = real_t(sheight / oh);

		int flip = rand() % 2;
		image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

		real_t dx = real_t((pleft / ow) / sx);
		real_t dy = real_t((ptop / oh) / sy);

		image sized = resize_image(cropped, w, h);
		if (flip)
			flip_image(sized);
		random_distort_image(sized, hue, saturation, exposure);
		d.X.vals[i] = sized.data;

		fill_truth_region(random_paths[i], d.y.vals[i], classes, size, flip, dx,
				dy, real_t(1. / sx), real_t(1. / sy));

		free_image(orig);
		free_image(cropped);
	}
	free(random_paths);
	return d;
}

data load_data_compare(int n, char **paths, int m, int classes, int w, int h) {
	if (m)
		paths = get_random_paths(paths, 2 * n, m);
	int i, j;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (real_t**) calloc(d.X.rows, sizeof(real_t*));
	d.X.cols = h * w * 6;

	int k = 2 * (classes);
	d.y = make_matrix(n, k);
	for (i = 0; i < n; ++i) {
		image im1 = load_image_color(paths[i * 2], w, h);
		image im2 = load_image_color(paths[i * 2 + 1], w, h);

		d.X.vals[i] = (real_t*) calloc(d.X.cols, sizeof(real_t));
		memcpy(d.X.vals[i], im1.data, h * w * 3 * sizeof(real_t));
		memcpy(d.X.vals[i] + h * w * 3, im2.data, h * w * 3 * sizeof(real_t));

		int id;
		real_t iou;

		char imlabel1[4096];
		char imlabel2[4096];
		find_replace(paths[i * 2], "imgs", "labels", imlabel1);
		find_replace(imlabel1, "jpg", "txt", imlabel1);
		FILE *fp1 = fopen(imlabel1, "r");

		//TODO: make ready for all precision
		float tmp;
		while (fscanf(fp1, "%d %f", &id, &tmp) == 2) {
			iou = tmp;
			if (d.y.vals[i][2 * id] < iou)
				d.y.vals[i][2 * id] = iou;
		}

		find_replace(paths[i * 2 + 1], "imgs", "labels", imlabel2);
		find_replace(imlabel2, "jpg", "txt", imlabel2);
		FILE *fp2 = fopen(imlabel2, "r");

		while (fscanf(fp2, "%d %f", &id, &tmp) == 2) {
			iou = tmp;
			if (d.y.vals[i][2 * id + 1] < iou)
				d.y.vals[i][2 * id + 1] = iou;
		}

		for (j = 0; j < classes; ++j) {
			if (d.y.vals[i][2 * j] > .5 && d.y.vals[i][2 * j + 1] < .5) {
				d.y.vals[i][2 * j] = 1;
				d.y.vals[i][2 * j + 1] = 0;
			} else if (d.y.vals[i][2 * j] < .5 && d.y.vals[i][2 * j + 1] > .5) {
				d.y.vals[i][2 * j] = 0;
				d.y.vals[i][2 * j + 1] = 1;
			} else {
				d.y.vals[i][2 * j] = SECRET_NUM;
				d.y.vals[i][2 * j + 1] = SECRET_NUM;
			}
		}
		fclose(fp1);
		fclose(fp2);

		free_image(im1);
		free_image(im2);
	}
	if (m)
		free(paths);
	return d;
}

data load_data_swag(char **paths, int n, int classes, real_t jitter) {
	int index = rand() % n;
	char *random_path = paths[index];

	image orig = load_image_color(random_path, 0, 0);
	int h = orig.h;
	int w = orig.w;

	data d = { 0 };
	d.shallow = 0;
	d.w = w;
	d.h = h;

	d.X.rows = 1;
	d.X.vals = (real_t**) calloc(d.X.rows, sizeof(real_t*));
	d.X.cols = h * w * 3;

	int k = (4 + classes) * 90;
	d.y = make_matrix(1, k);

	int dw = w * jitter;
	int dh = h * jitter;

	int pleft = rand_uniform(real_t(-dw), real_t(dw));
	int pright = rand_uniform(real_t(-dw), real_t(dw));
	int ptop = rand_uniform(real_t(-dh), real_t(dh));
	int pbot = rand_uniform(real_t(-dh), real_t(dh));

	int swidth = w - pleft - pright;
	int sheight = h - ptop - pbot;

	real_t sx = real_t(swidth / w);
	real_t sy = real_t(sheight / h);

	int flip = rand() % 2;
	image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

	real_t dx = real_t((pleft / w) / sx);
	real_t dy = real_t((ptop / h) / sy);

	image sized = resize_image(cropped, w, h);
	if (flip)
		flip_image(sized);
	d.X.vals[0] = sized.data;

	fill_truth_swag(random_path, d.y.vals[0], classes, flip, dx, dy, real_t(1. / sx),
			real_t(1. / sy));

	free_image(orig);
	free_image(cropped);

	return d;
}

data load_data_detection(int n, char **paths, int m, int w, int h, int boxes,
		int classes, real_t jitter, real_t hue, real_t saturation,
		real_t exposure) {
	char **random_paths = get_random_paths(paths, n, m);
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = (real_t**) calloc(d.X.rows, sizeof(real_t*));
	d.X.cols = h * w * 3;

	d.y = make_matrix(n, 5 * boxes);
	for (i = 0; i < n; ++i) {
		image orig = load_image_color(random_paths[i], 0, 0);
		image sized = make_image(w, h, orig.c);
		fill_image(sized, real_t(.5));

		real_t dw = real_t(jitter * orig.w);
		real_t dh = real_t(jitter * orig.h);

		real_t new_ar = real_t((orig.w + rand_uniform(-dw, dw))
				/ (orig.h + rand_uniform(-dh, dh)));
		//real_t scale = rand_uniform(.25, 2);
		real_t scale = real_t(1);

		real_t nw, nh;

		if (new_ar < 1) {
			nh = scale * h;
			nw = nh * new_ar;
		} else {
			nw = scale * w;
			nh = nw / new_ar;
		}

		real_t dx = rand_uniform(real_t(0), real_t(w - nw));
		real_t dy = rand_uniform(real_t(0), real_t(h - nh));

		place_image(orig, nw, nh, dx, dy, sized);

		random_distort_image(sized, hue, saturation, exposure);

		int flip = rand() % 2;
		if (flip)
			flip_image(sized);
		d.X.vals[i] = sized.data;

		fill_truth_detection(random_paths[i], boxes, d.y.vals[i], classes, flip,
				real_t(-dx / w), real_t(-dy / h), real_t(nw / w), real_t(nh / h));

		free_image(orig);
	}
	free(random_paths);
	return d;
}

void *load_thread(void *ptr) {
	//printf("Loading data: %d\n", rand());
	load_args a = *(struct load_args*) ptr;
	if (a.exposure == 0)
		a.exposure = 1;
	if (a.saturation == 0)
		a.saturation = 1;
	if (a.aspect == 0)
		a.aspect = 1;

	if (a.type == OLD_CLASSIFICATION_DATA) {
		*a.d = load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
	} else if (a.type == REGRESSION_DATA) {
		*a.d = load_data_regression(a.paths, a.n, a.m, a.classes, a.min, a.max,
				a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
	} else if (a.type == CLASSIFICATION_DATA) {
		*a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes,
				a.hierarchy, a.min, a.max, a.size, a.angle, a.aspect, a.hue,
				a.saturation, a.exposure, a.center);
	} else if (a.type == SUPER_DATA) {
		*a.d = load_data_super(a.paths, a.n, a.m, a.w, a.h, a.scale);
	} else if (a.type == WRITING_DATA) {
		*a.d = load_data_writing(a.paths, a.n, a.m, a.w, a.h, a.out_w, a.out_h);
	} else if (a.type == ISEG_DATA) {
		*a.d = load_data_iseg(a.n, a.paths, a.m, a.w, a.h, a.classes,
				a.num_boxes, a.scale, a.min, a.max, a.angle, a.aspect, a.hue,
				a.saturation, a.exposure);
	} else if (a.type == INSTANCE_DATA) {
		*a.d = load_data_mask(a.n, a.paths, a.m, a.w, a.h, a.classes,
				a.num_boxes, a.coords, a.min, a.max, a.angle, a.aspect, a.hue,
				a.saturation, a.exposure);
	} else if (a.type == SEGMENTATION_DATA) {
		*a.d = load_data_seg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.min,
				a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure,
				a.scale);
	} else if (a.type == REGION_DATA) {
		*a.d = load_data_region(a.n, a.paths, a.m, a.w, a.h, a.num_boxes,
				a.classes, a.jitter, a.hue, a.saturation, a.exposure);
	} else if (a.type == DETECTION_DATA) {
		*a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.num_boxes,
				a.classes, a.jitter, a.hue, a.saturation, a.exposure);
	} else if (a.type == SWAG_DATA) {
		*a.d = load_data_swag(a.paths, a.n, a.classes, a.jitter);
	} else if (a.type == COMPARE_DATA) {
		*a.d = load_data_compare(a.n, a.paths, a.m, a.classes, a.w, a.h);
	} else if (a.type == IMAGE_DATA) {
		*(a.im) = load_image_color(a.path, 0, 0);
		*(a.resized) = resize_image(*(a.im), a.w, a.h);
	} else if (a.type == LETTERBOX_DATA) {
		*(a.im) = load_image_color(a.path, 0, 0);
		*(a.resized) = letterbox_image(*(a.im), a.w, a.h);
	} else if (a.type == TAG_DATA) {
		*a.d = load_data_tag(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size,
				a.angle, a.aspect, a.hue, a.saturation, a.exposure);
	}
	free(ptr);
	return 0;
}

pthread_t load_data_in_thread(load_args args) {
	pthread_t thread;
	struct load_args *ptr = (load_args*) calloc(1, sizeof(struct load_args));
	*ptr = args;
	if (pthread_create(&thread, 0, load_thread, ptr))
		error("Thread creation failed");
	return thread;
}

void *load_threads(void *ptr) {
	int i;
	load_args args = *(load_args *) ptr;
	if (args.threads == 0)
		args.threads = 1;
	data *out = args.d;
	int total = args.n;
	free(ptr);
	data *buffers = (data*) calloc(args.threads, sizeof(data));
	pthread_t *threads = (pthread_t*) calloc(args.threads, sizeof(pthread_t));
	for (i = 0; i < args.threads; ++i) {
		args.d = buffers + i;
		args.n = (i + 1) * total / args.threads - i * total / args.threads;
		threads[i] = load_data_in_thread(args);
	}
	for (i = 0; i < args.threads; ++i) {
		pthread_join(threads[i], 0);
	}
	*out = concat_datas(buffers, args.threads);
	out->shallow = 0;
	for (i = 0; i < args.threads; ++i) {
		buffers[i].shallow = 1;
		free_data(buffers[i]);
	}
	free(buffers);
	free(threads);
	return 0;
}

void load_data_blocking(load_args args) {
	struct load_args *ptr = (load_args*) calloc(1, sizeof(struct load_args));
	*ptr = args;
	load_thread(ptr);
}

pthread_t load_data(load_args args) {
	pthread_t thread;
	struct load_args *ptr = (load_args*) calloc(1, sizeof(struct load_args));
	*ptr = args;
	if (pthread_create(&thread, 0, load_threads, ptr))
		error("Thread creation failed");
	return thread;
}

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w,
		int out_h) {
	if (m)
		paths = get_random_paths(paths, n, m);
	char **replace_paths = find_replace_paths(paths, n, ".png", "-label.png");
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_paths(paths, n, w, h);
	d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
	if (m)
		free(paths);
	int i;
	for (i = 0; i < n; ++i)
		free(replace_paths[i]);
	free(replace_paths);
	return d;
}

data load_data_old(char **paths, int n, int m, char **labels, int k, int w,
		int h) {
	if (m)
		paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_paths(paths, n, w, h);
	d.y = load_labels_paths(paths, n, labels, k, 0);
	if (m)
		free(paths);
	return d;
}

/*
 data load_data_study(char **paths, int n, int m, char **labels, int k, int min, int max, int size, real_t angle, real_t aspect, real_t hue, real_t saturation, real_t exposure)
 {
 data d = {0};
 d.indexes = calloc(n, sizeof(int));
 if(m) paths = get_random_paths_indexes(paths, n, m, d.indexes);
 d.shallow = 0;
 d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure);
 d.y = load_labels_paths(paths, n, labels, k);
 if(m) free(paths);
 return d;
 }
 */

data load_data_super(char **paths, int n, int m, int w, int h, int scale) {
	if (m)
		paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;

	int i;
	d.X.rows = n;
	d.X.vals = (real_t**) calloc(n, sizeof(real_t*));
	d.X.cols = w * h * 3;

	d.y.rows = n;
	d.y.vals = (real_t**) calloc(n, sizeof(real_t*));
	d.y.cols = w * scale * h * scale * 3;

	for (i = 0; i < n; ++i) {
		image im = load_image_color(paths[i], 0, 0);
		image crop = random_crop_image(im, w * scale, h * scale);
		int flip = rand() % 2;
		if (flip)
			flip_image(crop);
		image resize = resize_image(crop, w, h);
		d.X.vals[i] = resize.data;
		d.y.vals[i] = crop.data;
		free_image(im);
	}

	if (m)
		free(paths);
	return d;
}

data load_data_regression(char **paths, int n, int m, int k, int min, int max,
		int size, real_t angle, real_t aspect, real_t hue, real_t saturation,
		real_t exposure) {
	if (m)
		paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue,
			saturation, exposure, 0);
	d.y = load_regression_labels_paths(paths, n, k);
	if (m)
		free(paths);
	return d;
}

data select_data(data *orig, int *inds) {
	data d = { 0 };
	d.shallow = 1;
	d.w = orig[0].w;
	d.h = orig[0].h;

	d.X.rows = orig[0].X.rows;
	d.y.rows = orig[0].X.rows;

	d.X.cols = orig[0].X.cols;
	d.y.cols = orig[0].y.cols;

	d.X.vals = (real_t**) calloc(orig[0].X.rows, sizeof(real_t *));
	d.y.vals = (real_t**) calloc(orig[0].y.rows, sizeof(real_t *));
	int i;
	for (i = 0; i < d.X.rows; ++i) {
		d.X.vals[i] = orig[inds[i]].X.vals[i];
		d.y.vals[i] = orig[inds[i]].y.vals[i];
	}
	return d;
}

data *tile_data(data orig, int divs, int size) {
	data *ds = (data*) calloc(divs * divs, sizeof(data));
	int i, j;
#pragma omp parallel for
	for (i = 0; i < divs * divs; ++i) {
		data d;
		d.shallow = 0;
		d.w = orig.w / divs * size;
		d.h = orig.h / divs * size;
		d.X.rows = orig.X.rows;
		d.X.cols = d.w * d.h * 3;
		d.X.vals = (real_t**) calloc(d.X.rows, sizeof(real_t*));

		d.y = copy_matrix(orig.y);
#pragma omp parallel for
		for (j = 0; j < orig.X.rows; ++j) {
			int x = (i % divs) * orig.w / divs - (d.w - orig.w / divs) / 2;
			int y = (i / divs) * orig.h / divs - (d.h - orig.h / divs) / 2;
			image im = real_t_to_image(orig.w, orig.h, 3, orig.X.vals[j]);
			d.X.vals[j] = crop_image(im, x, y, d.w, d.h).data;
		}
		ds[i] = d;
	}
	return ds;
}

data resize_data(data orig, int w, int h) {
	data d = { 0 };
	d.shallow = 0;
	d.w = w;
	d.h = h;
	int i;
	d.X.rows = orig.X.rows;
	d.X.cols = w * h * 3;
	d.X.vals = (real_t**) calloc(d.X.rows, sizeof(real_t*));

	d.y = copy_matrix(orig.y);
#pragma omp parallel for
	for (i = 0; i < orig.X.rows; ++i) {
		image im = real_t_to_image(orig.w, orig.h, 3, orig.X.vals[i]);
		d.X.vals[i] = resize_image(im, w, h).data;
	}
	return d;
}

data load_data_augment(char **paths, int n, int m, char **labels, int k,
		tree *hierarchy, int min, int max, int size, real_t angle,
		real_t aspect, real_t hue, real_t saturation, real_t exposure,
		int center) {
	if (m)
		paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.shallow = 0;
	d.w = size;
	d.h = size;
	d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue,
			saturation, exposure, center);
	d.y = load_labels_paths(paths, n, labels, k, hierarchy);
	if (m)
		free(paths);
	return d;
}

data load_data_tag(char **paths, int n, int m, int k, int min, int max,
		int size, real_t angle, real_t aspect, real_t hue, real_t saturation,
		real_t exposure) {
	if (m)
		paths = get_random_paths(paths, n, m);
	data d = { 0 };
	d.w = size;
	d.h = size;
	d.shallow = 0;
	d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue,
			saturation, exposure, 0);
	d.y = load_tags_paths(paths, n, k);
	if (m)
		free(paths);
	return d;
}

matrix concat_matrix(matrix m1, matrix m2) {
	int i, count = 0;
	matrix m;
	m.cols = m1.cols;
	m.rows = m1.rows + m2.rows;
	m.vals = (real_t**) calloc(m1.rows + m2.rows, sizeof(real_t*));
	for (i = 0; i < m1.rows; ++i) {
		m.vals[count++] = m1.vals[i];
	}
	for (i = 0; i < m2.rows; ++i) {
		m.vals[count++] = m2.vals[i];
	}
	return m;
}

data concat_data(data d1, data d2) {
	data d = { 0 };
	d.shallow = 1;
	d.X = concat_matrix(d1.X, d2.X);
	d.y = concat_matrix(d1.y, d2.y);
	d.w = d1.w;
	d.h = d1.h;
	return d;
}

data concat_datas(data *d, int n) {
	int i;
	data out = { 0 };
	for (i = 0; i < n; ++i) {
		data new_ = concat_data(d[i], out);
		free_data(out);
		out = new_;
	}
	return out;
}

data load_categorical_data_csv(char *filename, int target, int k) {
	data d = { 0 };
	d.shallow = 0;
	matrix X = csv_to_matrix(filename);
	real_t *truth_1d = pop_column(&X, target);
	real_t **truth = one_hot_encode(truth_1d, X.rows, k);
	matrix y;
	y.rows = X.rows;
	y.cols = k;
	y.vals = truth;
	d.X = X;
	d.y = y;
	free(truth_1d);
	return d;
}

data load_cifar10_data(char *filename) {
	data d = { 0 };
	d.shallow = 0;
	long i, j;
	matrix X = make_matrix(10000, 3072);
	matrix y = make_matrix(10000, 10);
	d.X = X;
	d.y = y;

	FILE *fp = fopen(filename, "rb");
	if (!fp)
		file_error(filename);
	for (i = 0; i < 10000; ++i) {
		unsigned char bytes[3073];
		fread(bytes, 1, 3073, fp);
		int class_ = bytes[0];
		y.vals[i][class_] = 1;
		for (j = 0; j < X.cols; ++j) {
			X.vals[i][j] = (double) bytes[j + 1];
		}
	}
	scale_data_rows(d, real_t(1. / 255));
	//normalize_data_rows(d);
	fclose(fp);
	return d;
}

void get_random_batch(data d, int n, real_t *X, real_t *y) {
	int j;
	for (j = 0; j < n; ++j) {
		int index = rand() % d.X.rows;
		memcpy(X + j * d.X.cols, d.X.vals[index], d.X.cols * sizeof(real_t));
		memcpy(y + j * d.y.cols, d.y.vals[index], d.y.cols * sizeof(real_t));
	}
}

void get_next_batch(data d, int n, int offset, real_t *X, real_t *y) {
	int j;
	for (j = 0; j < n; ++j) {
		int index = offset + j;
		memcpy(X + j * d.X.cols, d.X.vals[index], d.X.cols * sizeof(real_t));
		if (y)
			memcpy(y + j * d.y.cols, d.y.vals[index],
					d.y.cols * sizeof(real_t));
	}
}

void smooth_data(data d) {
	int i, j;
	real_t scale = real_t(1. / d.y.cols);
	real_t eps = real_t(.1);
	for (i = 0; i < d.y.rows; ++i) {
		for (j = 0; j < d.y.cols; ++j) {
			d.y.vals[i][j] = eps * scale + (1 - eps) * d.y.vals[i][j];
		}
	}
}

data load_all_cifar10() {
	data d = { 0 };
	d.shallow = 0;
	int i, j, b;
	matrix X = make_matrix(50000, 3072);
	matrix y = make_matrix(50000, 10);
	d.X = X;
	d.y = y;

	for (b = 0; b < 5; ++b) {
		char buff[256];
		sprintf(buff, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin",
				b + 1);
		FILE *fp = fopen(buff, "rb");
		if (!fp)
			file_error(buff);
		for (i = 0; i < 10000; ++i) {
			unsigned char bytes[3073];
			fread(bytes, 1, 3073, fp);
			int class_ = bytes[0];
			y.vals[i + b * 10000][class_] = 1;
			for (j = 0; j < X.cols; ++j) {
				X.vals[i + b * 10000][j] = (double) bytes[j + 1];
			}
		}
		fclose(fp);
	}
	//normalize_data_rows(d);
	scale_data_rows(d, real_t(1. / 255));
	smooth_data(d);
	return d;
}

data load_go(char *filename) {
	FILE *fp = fopen(filename, "rb");
	matrix X = make_matrix(3363059, 361);
	matrix y = make_matrix(3363059, 361);
	int row, col;

	if (!fp)
		file_error(filename);
	char *label;
	int count = 0;
	while ((label = fgetl(fp))) {
		int i;
		if (count == X.rows) {
			X = resize_matrix(X, count * 2);
			y = resize_matrix(y, count * 2);
		}
		sscanf(label, "%d %d", &row, &col);
		char *board = fgetl(fp);

		int index = row * 19 + col;
		y.vals[count][index] = 1;

		for (i = 0; i < 19 * 19; ++i) {
			real_t val = real_t(0);
			if (board[i] == '1')
				val = 1;
			else if (board[i] == '2')
				val = -1;
			X.vals[count][i] = val;
		}
		++count;
		free(label);
		free(board);
	}
	X = resize_matrix(X, count);
	y = resize_matrix(y, count);

	data d = { 0 };
	d.shallow = 0;
	d.X = X;
	d.y = y;

	fclose(fp);

	return d;
}

void randomize_data(data d) {
	int i;
	for (i = d.X.rows - 1; i > 0; --i) {
		int index = rand() % i;
		real_t *swap = d.X.vals[index];
		d.X.vals[index] = d.X.vals[i];
		d.X.vals[i] = swap;

		swap = d.y.vals[index];
		d.y.vals[index] = d.y.vals[i];
		d.y.vals[i] = swap;
	}
}

void scale_data_rows(data d, real_t s) {
	int i;
	for (i = 0; i < d.X.rows; ++i) {
		scale_array(d.X.vals[i], d.X.cols, s);
	}
}

void translate_data_rows(data d, real_t s) {
	int i;
	for (i = 0; i < d.X.rows; ++i) {
		translate_array(d.X.vals[i], d.X.cols, s);
	}
}

data copy_data(data d) {
	data c = { 0 };
	c.w = d.w;
	c.h = d.h;
	c.shallow = 0;
	c.num_boxes = d.num_boxes;
	c.boxes = d.boxes;
	c.X = copy_matrix(d.X);
	c.y = copy_matrix(d.y);
	return c;
}

void normalize_data_rows(data d) {
	int i;
	for (i = 0; i < d.X.rows; ++i) {
		normalize_array(d.X.vals[i], d.X.cols);
	}
}

data get_data_part(data d, int part, int total) {
	data p = { 0 };
	p.shallow = 1;
	p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
	p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
	p.X.cols = d.X.cols;
	p.y.cols = d.y.cols;
	p.X.vals = d.X.vals + d.X.rows * part / total;
	p.y.vals = d.y.vals + d.y.rows * part / total;
	return p;
}

data get_random_data(data d, int num) {
	data r = { 0 };
	r.shallow = 1;

	r.X.rows = num;
	r.y.rows = num;

	r.X.cols = d.X.cols;
	r.y.cols = d.y.cols;

	r.X.vals = (real_t**) calloc(num, sizeof(real_t *));
	r.y.vals = (real_t**) calloc(num, sizeof(real_t *));

	int i;
	for (i = 0; i < num; ++i) {
		int index = rand() % d.X.rows;
		r.X.vals[i] = d.X.vals[index];
		r.y.vals[i] = d.y.vals[index];
	}
	return r;
}

data *split_data(data d, int part, int total) {
	data *split = (data*) calloc(2, sizeof(data));
	int i;
	int start = part * d.X.rows / total;
	int end = (part + 1) * d.X.rows / total;
	data train;
	data test;
	train.shallow = test.shallow = 1;

	test.X.rows = test.y.rows = end - start;
	train.X.rows = train.y.rows = d.X.rows - (end - start);
	train.X.cols = test.X.cols = d.X.cols;
	train.y.cols = test.y.cols = d.y.cols;

	train.X.vals = (real_t**) calloc(train.X.rows, sizeof(real_t*));
	test.X.vals = (real_t**) calloc(test.X.rows, sizeof(real_t*));
	train.y.vals = (real_t**) calloc(train.y.rows, sizeof(real_t*));
	test.y.vals = (real_t**) calloc(test.y.rows, sizeof(real_t*));

	for (i = 0; i < start; ++i) {
		train.X.vals[i] = d.X.vals[i];
		train.y.vals[i] = d.y.vals[i];
	}
	for (i = start; i < end; ++i) {
		test.X.vals[i - start] = d.X.vals[i];
		test.y.vals[i - start] = d.y.vals[i];
	}
	for (i = end; i < d.X.rows; ++i) {
		train.X.vals[i - (end - start)] = d.X.vals[i];
		train.y.vals[i - (end - start)] = d.y.vals[i];
	}
	split[0] = train;
	split[1] = test;
	return split;
}

