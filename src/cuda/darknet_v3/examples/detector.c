#include "darknet.h"
#include "detection_gold_w.h"
#include "cuda.h"
#define PRINT_INTERVAL 10

static int coco_ids[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
		18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
		39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
		58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
		80, 81, 82, 84, 85, 86, 87, 88, 89, 90 };

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus,
		int ngpus, int clear) {
	list *options = read_data_cfg(datacfg);
	char *train_images = option_find_str(options, "train", "data/train.list");
	char *backup_directory = option_find_str(options, "backup", "/backup/");

	srand(time(0));
	char *base = basecfg(cfgfile);
	printf("%s\n", base);
	real_t avg_loss = -1;
	network **nets = calloc(ngpus, sizeof(network));

	srand(time(0));
	int seed = rand();
	int i;
	for (i = 0; i < ngpus; ++i) {
		srand(seed);
#ifdef GPU
		cuda_set_device(gpus[i]);
#endif
		nets[i] = load_network(cfgfile, weightfile, clear);
		nets[i]->learning_rate *= ngpus;
	}
	srand(time(0));
	network *net = nets[0];

	int imgs = net->batch * net->subdivisions * ngpus;
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate,
			net->momentum, net->decay);
	data train, buffer;

	layer l = net->layers[net->n - 1];

	int classes = l.classes;
	real_t jitter = l.jitter;

	list *plist = get_paths(train_images);
	//int N = plist->size;
	char **paths = (char **) list_to_array(plist);

	load_args args = get_base_args(net);
	args.coords = l.coords;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	args.jitter = jitter;
	args.num_boxes = l.max_boxes;
	args.d = &buffer;
	args.type = DETECTION_DATA;
	//args.type = INSTANCE_DATA;
	args.threads = 64;

	pthread_t load_thread = load_data(args);
	double time;
	int count = 0;
	//while(i*imgs < N*120){
	while (get_current_batch(net) < net->max_batches) {
		if (l.random && count++ % 10 == 0) {
			printf("Resizing\n");
			int dim = (rand() % 10 + 10) * 32;
			if (get_current_batch(net) + 200 > net->max_batches)
				dim = 608;
			//int dim = (rand() % 4 + 16) * 32;
			printf("%d\n", dim);
			args.w = dim;
			args.h = dim;

			pthread_join(load_thread, 0);
			train = buffer;
			free_data(train);
			load_thread = load_data(args);

#pragma omp parallel for
			for (i = 0; i < ngpus; ++i) {
				resize_network(nets[i], dim, dim);
			}
			net = nets[0];
		}
		time = what_time_is_it_now();
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data(args);

		/*
		 int k;
		 for(k = 0; k < l.max_boxes; ++k){
		 box b = real_t_to_box(train.y.vals[10] + 1 + k*5);
		 if(!b.x) break;
		 printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
		 }
		 */
		/*
		 int zz;
		 for(zz = 0; zz < train.X.cols; ++zz){
		 image im = real_t_to_image(net->w, net->h, 3, train.X.vals[zz]);
		 int k;
		 for(k = 0; k < l.max_boxes; ++k){
		 box b = real_t_to_box(train.y.vals[zz] + k*5, 1);
		 printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
		 draw_bbox(im, b, 1, 1,0,0);
		 }
		 show_image(im, "truth11");
		 cvWaitKey(0);
		 save_image(im, "truth11");
		 }
		 */

		printf("Loaded: %lf seconds\n", what_time_is_it_now() - time);

		time = what_time_is_it_now();
		real_t loss = 0;
#ifdef GPU
		if (ngpus == 1) {
			loss = train_network(net, train);
		} else {
			loss = train_networks(nets, ngpus, train, 4);
		}
#else
		loss = train_network(net, train);
#endif
		if (avg_loss < 0)
			avg_loss = loss;
		avg_loss = avg_loss * .9 + loss * .1;

		i = get_current_batch(net);
		printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n",
				get_current_batch(net), loss, avg_loss, get_current_rate(net),
				what_time_is_it_now() - time, i * imgs);
		if (i % 100 == 0) {
#ifdef GPU
			if (ngpus != 1)
				sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s.backup", backup_directory, base);
			save_weights(net, buff);
		}
		if (i % 10000 == 0 || (i < 1000 && i % 100 == 0)) {
#ifdef GPU
			if (ngpus != 1)
				sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
	}
#ifdef GPU
	if (ngpus != 1)
		sync_nets(nets, ngpus, 0);
#endif
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}

static int get_coco_image_id(char *filename) {
	char *p = strrchr(filename, '/');
	char *c = strrchr(filename, '_');
	if (c)
		p = c;
	return atoi(p + 1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets,
		int num_boxes, int classes, int w, int h) {
	int i, j;
	int image_id = get_coco_image_id(image_path);
	for (i = 0; i < num_boxes; ++i) {
		real_t xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
		real_t xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
		real_t ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
		real_t ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

		if (xmin < 0)
			xmin = 0;
		if (ymin < 0)
			ymin = 0;
		if (xmax > w)
			xmax = w;
		if (ymax > h)
			ymax = h;

		real_t bx = xmin;
		real_t by = ymin;
		real_t bw = xmax - xmin;
		real_t bh = ymax - ymin;

		for (j = 0; j < classes; ++j) {
			if (dets[i].prob[j])
				fprintf(fp,
						"{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n",
						image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
		}
	}
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total,
		int classes, int w, int h) {
	int i, j;
	for (i = 0; i < total; ++i) {
		real_t xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
		real_t xmax = dets[i].bbox.x + dets[i].bbox.w / 2. + 1;
		real_t ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
		real_t ymax = dets[i].bbox.y + dets[i].bbox.h / 2. + 1;

		if (xmin < 1)
			xmin = 1;
		if (ymin < 1)
			ymin = 1;
		if (xmax > w)
			xmax = w;
		if (ymax > h)
			ymax = h;

		for (j = 0; j < classes; ++j) {
			if (dets[i].prob[j])
				fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
						xmin, ymin, xmax, ymax);
		}
	}
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total,
		int classes, int w, int h) {
	int i, j;
	for (i = 0; i < total; ++i) {
		real_t xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
		real_t xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
		real_t ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
		real_t ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

		if (xmin < 0)
			xmin = 0;
		if (ymin < 0)
			ymin = 0;
		if (xmax > w)
			xmax = w;
		if (ymax > h)
			ymax = h;

		for (j = 0; j < classes; ++j) {
			int class = j;
			if (dets[i].prob[class])
				fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1,
						dets[i].prob[class], xmin, ymin, xmax, ymax);
		}
	}
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile,
		char *outfile) {
	int j;
	list *options = read_data_cfg(datacfg);
	char *valid_images = option_find_str(options, "valid", "data/train.list");
	char *name_list = option_find_str(options, "names", "data/names.list");
	char *prefix = option_find_str(options, "results", "results");
	char **names = get_labels(name_list);
	char *mapf = option_find_str(options, "map", 0);
	int *map = 0;
	if (mapf)
		map = read_map(mapf);

	network *net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 2);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n",
			net->learning_rate, net->momentum, net->decay);
	srand(time(0));

	list *plist = get_paths(valid_images);
	char **paths = (char **) list_to_array(plist);

	layer l = net->layers[net->n - 1];
	int classes = l.classes;

	char buff[1024];
	char *type = option_find_str(options, "eval", "voc");
	FILE *fp = 0;
	FILE **fps = 0;
	int coco = 0;
	int imagenet = 0;
	if (0 == strcmp(type, "coco")) {
		if (!outfile)
			outfile = "coco_results";
		snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
		fp = fopen(buff, "w");
		fprintf(fp, "[\n");
		coco = 1;
	} else if (0 == strcmp(type, "imagenet")) {
		if (!outfile)
			outfile = "imagenet-detection";
		snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
		fp = fopen(buff, "w");
		imagenet = 1;
		classes = 200;
	} else {
		if (!outfile)
			outfile = "comp4_det_test_";
		fps = calloc(classes, sizeof(FILE *));
		for (j = 0; j < classes; ++j) {
			snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
			fps[j] = fopen(buff, "w");
		}
	}

	int m = plist->size;
	int i = 0;
	int t;

	real_t thresh = .005;
	real_t nms = .45;

	int nthreads = 4;
	image *val = calloc(nthreads, sizeof(image));
	image *val_resized = calloc(nthreads, sizeof(image));
	image *buf = calloc(nthreads, sizeof(image));
	image *buf_resized = calloc(nthreads, sizeof(image));
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

	image input = make_image(net->w, net->h, net->c * 2);

	load_args args = { 0 };
	args.w = net->w;
	args.h = net->h;
	//args.type = IMAGE_DATA;
	args.type = LETTERBOX_DATA;

	for (t = 0; t < nthreads; ++t) {
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}
	double start = what_time_is_it_now();
	for (i = nthreads; i < m + nthreads; i += nthreads) {
		fprintf(stderr, "%d\n", i);
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			pthread_join(thr[t], 0);
			val[t] = buf[t];
			val_resized[t] = buf_resized[t];
		}
		for (t = 0; t < nthreads && i + t < m; ++t) {
			args.path = paths[i + t];
			args.im = &buf[t];
			args.resized = &buf_resized[t];
			thr[t] = load_data_in_thread(args);
		}
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			char *path = paths[i + t - nthreads];
			char *id = basecfg(path);
			copy_cpu(net->w * net->h * net->c, val_resized[t].data, 1,
					input.data, 1);
			flip_image(val_resized[t]);
			copy_cpu(net->w * net->h * net->c, val_resized[t].data, 1,
					input.data + net->w * net->h * net->c, 1);

			network_predict(net, input.data);
			int w = val[t].w;
			int h = val[t].h;
			int num = 0;
			detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0,
					&num);
			if (nms)
				do_nms_sort(dets, num, classes, nms);
			if (coco) {
				print_cocos(fp, path, dets, num, classes, w, h);
			} else if (imagenet) {
				print_imagenet_detections(fp, i + t - nthreads + 1, dets, num,
						classes, w, h);
			} else {
				print_detector_detections(fps, id, dets, num, classes, w, h);
			}
			free_detections(dets, num);
			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}
	for (j = 0; j < classes; ++j) {
		if (fps)
			fclose(fps[j]);
	}
	if (coco) {
		fseek(fp, -2, SEEK_CUR);
		fprintf(fp, "\n]\n");
		fclose(fp);
	}
	fprintf(stderr, "Total Detection Time: %f Seconds\n",
			what_time_is_it_now() - start);
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile,
		char *outfile) {
	int j;
	list *options = read_data_cfg(datacfg);
	char *valid_images = option_find_str(options, "valid", "data/train.list");
	char *name_list = option_find_str(options, "names", "data/names.list");
	char *prefix = option_find_str(options, "results", "results");
	char **names = get_labels(name_list);
	char *mapf = option_find_str(options, "map", 0);
	int *map = 0;
	if (mapf)
		map = read_map(mapf);

	network *net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n",
			net->learning_rate, net->momentum, net->decay);
	srand(time(0));

	list *plist = get_paths(valid_images);
	char **paths = (char **) list_to_array(plist);

	layer l = net->layers[net->n - 1];
	int classes = l.classes;

	char buff[1024];
	char *type = option_find_str(options, "eval", "voc");
	FILE *fp = 0;
	FILE **fps = 0;
	int coco = 0;
	int imagenet = 0;
	if (0 == strcmp(type, "coco")) {
		if (!outfile)
			outfile = "coco_results";
		snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
		fp = fopen(buff, "w");
		fprintf(fp, "[\n");
		coco = 1;
	} else if (0 == strcmp(type, "imagenet")) {
		if (!outfile)
			outfile = "imagenet-detection";
		snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
		fp = fopen(buff, "w");
		imagenet = 1;
		classes = 200;
	} else {
		if (!outfile)
			outfile = "comp4_det_test_";
		fps = calloc(classes, sizeof(FILE *));
		for (j = 0; j < classes; ++j) {
			snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
			fps[j] = fopen(buff, "w");
		}
	}

	int m = plist->size;
	int i = 0;
	int t;

	real_t thresh = .005;
	real_t nms = .45;

	int nthreads = 4;
	image *val = calloc(nthreads, sizeof(image));
	image *val_resized = calloc(nthreads, sizeof(image));
	image *buf = calloc(nthreads, sizeof(image));
	image *buf_resized = calloc(nthreads, sizeof(image));
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

	load_args args = { 0 };
	args.w = net->w;
	args.h = net->h;
	//args.type = IMAGE_DATA;
	args.type = LETTERBOX_DATA;

	for (t = 0; t < nthreads; ++t) {
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}
	double start = what_time_is_it_now();
	for (i = nthreads; i < m + nthreads; i += nthreads) {
		fprintf(stderr, "%d\n", i);
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			pthread_join(thr[t], 0);
			val[t] = buf[t];
			val_resized[t] = buf_resized[t];
		}
		for (t = 0; t < nthreads && i + t < m; ++t) {
			args.path = paths[i + t];
			args.im = &buf[t];
			args.resized = &buf_resized[t];
			thr[t] = load_data_in_thread(args);
		}
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			char *path = paths[i + t - nthreads];
			char *id = basecfg(path);
			real_t *X = val_resized[t].data;
			network_predict(net, X);
			int w = val[t].w;
			int h = val[t].h;
			int nboxes = 0;
			detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0,
					&nboxes);
			if (nms)
				do_nms_sort(dets, nboxes, classes, nms);
			if (coco) {
				print_cocos(fp, path, dets, nboxes, classes, w, h);
			} else if (imagenet) {
				print_imagenet_detections(fp, i + t - nthreads + 1, dets,
						nboxes, classes, w, h);
			} else {
				print_detector_detections(fps, id, dets, nboxes, classes, w, h);
			}
			free_detections(dets, nboxes);
			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}
	for (j = 0; j < classes; ++j) {
		if (fps)
			fclose(fps[j]);
	}
	if (coco) {
		fseek(fp, -2, SEEK_CUR);
		fprintf(fp, "\n]\n");
		fclose(fp);
	}
	fprintf(stderr, "Total Detection Time: %f Seconds\n",
			what_time_is_it_now() - start);
}

void validate_detector_recall(char *cfgfile, char *weightfile) {
	network *net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n",
			net->learning_rate, net->momentum, net->decay);
	srand(time(0));

	list *plist = get_paths("data/coco_val_5k.list");
	char **paths = (char **) list_to_array(plist);

	layer l = net->layers[net->n - 1];

	int j, k;

	int m = plist->size;
	int i = 0;

	real_t thresh = .001;
	real_t iou_thresh = .5;
	real_t nms = .4;

	int total = 0;
	int correct = 0;
	int proposals = 0;
	real_t avg_iou = 0;

	for (i = 0; i < m; ++i) {
		char *path = paths[i];
		image orig = load_image_color(path, 0, 0);
		image sized = resize_image(orig, net->w, net->h);
		char *id = basecfg(path);
		network_predict(net, sized.data);
		int nboxes = 0;
		detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5,
				0, 1, &nboxes);
		if (nms)
			do_nms_obj(dets, nboxes, 1, nms);

		char labelpath[4096];
		find_replace(path, "images", "labels", labelpath);
		find_replace(labelpath, "JPEGImages", "labels", labelpath);
		find_replace(labelpath, ".jpg", ".txt", labelpath);
		find_replace(labelpath, ".JPEG", ".txt", labelpath);

		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		for (k = 0; k < nboxes; ++k) {
			if (dets[k].objectness > thresh) {
				++proposals;
			}
		}
		for (j = 0; j < num_labels; ++j) {
			++total;
			box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
			real_t best_iou = 0;
			for (k = 0; k < l.w * l.h * l.n; ++k) {
				real_t iou = box_iou(dets[k].bbox, t);
				if (dets[k].objectness > thresh && iou > best_iou) {
					best_iou = iou;
				}
			}
			avg_iou += best_iou;
			if (best_iou > iou_thresh) {
				++correct;
			}
		}

		fprintf(stderr,
				"%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i,
				correct, total, (real_t) proposals / (i + 1),
				avg_iou * 100 / total, 100. * correct / total);
		free(id);
		free_image(orig);
		free_image(sized);
	}
}

void test_detector(char *datacfg, char *cfgfile, char *weightfile,
		char *filename, real_t thresh, real_t hier_thresh, char *outfile,
		int fullscreen) {
	list *options = read_data_cfg(datacfg);
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);

	image **alphabet = load_alphabet();
	network *net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 1);
	srand(2222222);
	double time;
	char buff[256];
	char *input = buff;
	real_t nms = .45;
	while (1) {
		if (filename) {
			strncpy(input, filename, 256);
		} else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input)
				return;
			strtok(input, "\n");
		}
		image im = load_image_color(input, 0, 0);
		image sized = letterbox_image(im, net->w, net->h);
		//image sized = resize_image(im, net->w, net->h);
		//image sized2 = resize_max(im, net->w);
		//image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
		//resize_network(net, sized.w, sized.h);
		layer l = net->layers[net->n - 1];

		real_t *X = sized.data;
		time = what_time_is_it_now();
		network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input,
				what_time_is_it_now() - time);
		int nboxes = 0;
		detection *dets = get_network_boxes(net, im.w, im.h, thresh,
				hier_thresh, 0, 1, &nboxes);
		//printf("%d\n", nboxes);
		//if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
		if (nms)
			do_nms_sort(dets, nboxes, l.classes, nms);
		draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
		free_detections(dets, nboxes);
		if (outfile) {
			save_image(im, outfile);
		} else {
			save_image(im, "predictions");
#ifdef OPENCV
			make_window("predictions", 512, 512, 0);
			show_image(im, "predictions", 0);
#endif
		}

		free_image(im);
		free_image(sized);
		if (filename)
			break;
	}
}

void load_all_images(image* imgs, image* sized_images, char** img_names, int plist_size, int net_w, int net_h) {
	int i;
	for (i = 0; i < plist_size; i++) {
		imgs[i] = load_image_color(img_names[i], 0, 0);
		sized_images[i] = letterbox_image(imgs[i], net_w, net_h);
	}
}

void free_all_images(image *imgs, image* sized_images, int list_size) {
	//          free_image(im);
//	int i, s;
//	for (s = 0; s < smx_red; s++) {
    for (int i = 0; i < list_size; i++) {
        free_image(imgs[i]);
        free_image(sized_images[i]);
    }
    free(imgs);
    free(sized_images);
//	}
//	free(imgs);
//	free(sized_images);
}
// DOES NOT WORK
//#ifdef GPU
//cudaStream_t* init_multi_streams(int smx_size) {
//	cudaStream_t* stream_array = malloc(sizeof(cudaStream_t) * smx_size);
//	int smx;
//	for (smx = 0; smx < smx_size; smx++) {
//		stream_array[smx] = NULL;
////		cudaError_t status = cudaStreamCreate(&stream_array[smx]);
////		check_error(status);
//	}
//	return stream_array;
//}
//
//void del_multi_streams(cudaStream_t* stream_array, int smx_size) {
//	int smx;
//	for (smx = 0; smx < smx_size; smx++) {
////		cudaError_t status = cudaStreamDestroy(stream_array[smx]);
////		check_error(status);
//	}
//	free(stream_array);
//}
//#endif

void test_detector_radiation(char *datacfg, char *cfgfile, char *weightfile,
		char *filename, real_t thresh, real_t hier_thresh, char *outfile,
		int fullscreen, int argc, char** argv) {
	/**
	 * DetectionGold declaration
	 */
	detection_gold_t *gold = create_detection_gold(argc, argv, thresh,
			hier_thresh, filename, cfgfile, datacfg, "detector", weightfile);
//	int smx_redundancy = get_smx_redundancy(gold);
//	network** net_array = malloc(sizeof(network*) * smx_redundancy);

	printf("CFG FILE: %s\nDATA CFG: %s\nWeightfile: %s\nImage data path file: %s\nThresh: %f\n",
			cfgfile, datacfg, weightfile, filename, thresh);

	//load images
//	image** image_array = malloc(sizeof(image*) * smx_redundancy);
//	image** sized_array = malloc(sizeof(image*) * smx_redundancy);

	char **img_names = get_labels(filename);
	int max_it = get_iterations(gold);
	int plist_size = get_img_num(gold);

//	int inet;
//	for (inet = 0; inet < smx_redundancy; inet++) {
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    //Set tensor cores on the net
//    net->smx_redundancy = smx_redundancy;
#ifdef GPU
    //	cudaStream_t *stream_array = init_multi_streams(smx_redundancy);
	//--------------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    net->use_tensor_cores = get_use_tensor_cores(gold);
    net->st = stream; //stream_array[inet];
#endif
//    net_array[inet] = net;

    //load images
    printf("Loading images for network\n");
//    image_array[inet] = (image*) malloc(sizeof(image) * plist_size);
//    sized_array[inet] = (image*) malloc(sizeof(image) * plist_size);
    image* image_array = (image*) malloc(sizeof(image) * plist_size);
    image* sized_array = (image*) malloc(sizeof(image) * plist_size);
    load_all_images(image_array, sized_array, img_names, plist_size, net->w, net->h);
    printf("Images loaded\n");

//	}
	srand(2222222);
	double time;
	real_t nms = .45;

	int iteration, img;
//
//	image* images = (image*) malloc(sizeof(image) * plist_size);
//	image* sized_images = (image*) malloc(sizeof(image) * plist_size);
//	load_all_images(images, sized_images, img_names, plist_size,
//			net_array[0]->w, net_array[0]->h);
//	real_t** X_arr = malloc(sizeof(real_t*) * smx_redundancy);
//	detection** dets_array = malloc(sizeof(detection*) * smx_redundancy);
//	int* nboxes_array = malloc(sizeof(int) * smx_redundancy);
	//start the process
	for (iteration = 0; iteration < max_it; iteration++) {
//		int last_errors = 0;
		for (img = 0; img < plist_size; img++) {

			layer l = net->layers[net->n - 1];

//			real_t *X = sized.data;
			image im = image_array[img];
//			for (inet = 0; inet < smx_redundancy; inet++) {
//				image sized = sized_array[inet][img];
//				X_arr[inet] = sized_array[inet][img].data;
//			}
            real_t* X = sized_array[img].data;
			time = what_time_is_it_now();

			//Run one iteration
			start_iteration_wrapper(gold);
			network_predict(net, X);
//			network_predict_smx_red(net_array, X_arr);
			end_iteration_wrapper(gold);

//			int nboxes = 0;
//			printf("aui antes do dets\n");
//			for (inet = 0; inet < smx_redundancy; inet++) {
            int nboxes = 0;
            detection* dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);

            if (nms)
                do_nms_sort(dets, nboxes, l.classes, nms);
//			}
//			printf("aui antes do run\n");
			//Save or compare
			double start = what_time_is_it_now();
			int curr_err = run(gold, dets, nboxes, img, l.classes, im.w, im.h);
			double end = what_time_is_it_now();

			/*
			if (last_errors && curr_err) {
				printf(
						"IT IS LESS PROBLABLE THAT DARKNET GIVE US TWO ERRORS SEQUENTIALY, ABORTING\n");
				exit(-1);
			}*/
//			for (inet = 0; inet < smx_redundancy; inet++) {
            //			if ((iteration * img) % PRINT_INTERVAL == 0) {
            printf(
                    "Iteration %d img %d, %d objects predicted in %f seconds. %d errors, coparisson took %lfs\n",
                    iteration, img, nboxes,
                    what_time_is_it_now() - time, curr_err, end - start);
            //			}

            free_detections(dets, nboxes);
//			}
//			last_errors = curr_err;
		}
	}

//	free(dets_array);
//	free(nboxes_array);
//	for (inet = 0; inet < smx_redundancy; inet++) {
	free_network(net);
//	}
//	free(net_array);
#ifdef GPU
//	del_multi_streams(stream_array, smx_redundancy);
    cudaStreamDestroy(stream);
#endif
	destroy_detection_gold(gold);
	free_all_images(image_array, sized_array, plist_size);
//	free(X_arr);
}

void run_detector(int argc, char **argv) {
	char *prefix = find_char_arg(argc, argv, "-prefix", 0);
	real_t thresh = find_real_t_arg(argc, argv, "-thresh", .5);
	real_t hier_thresh = find_real_t_arg(argc, argv, "-hier", .5);
	int cam_index = find_int_arg(argc, argv, "-c", 0);
	int frame_skip = find_int_arg(argc, argv, "-s", 0);
	int avg = find_int_arg(argc, argv, "-avg", 3);
	if (argc < 4) {
		fprintf(stderr,
				"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n",
				argv[0], argv[1]);
		return;
	}
	char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
	char *outfile = find_char_arg(argc, argv, "-out", 0);
	int *gpus = 0;
	int gpu = 0;
	int ngpus = 0;
	if (gpu_list) {
		printf("%s\n", gpu_list);
		int len = strlen(gpu_list);
		ngpus = 1;
		int i;
		for (i = 0; i < len; ++i) {
			if (gpu_list[i] == ',')
				++ngpus;
		}
		gpus = calloc(ngpus, sizeof(int));
		for (i = 0; i < ngpus; ++i) {
			gpus[i] = atoi(gpu_list);
			gpu_list = strchr(gpu_list, ',') + 1;
		}
	} else {
		gpu = gpu_index;
		gpus = &gpu;
		ngpus = 1;
	}

	int clear = find_arg(argc, argv, "-clear");
	int fullscreen = find_arg(argc, argv, "-fullscreen");
	int width = find_int_arg(argc, argv, "-w", 0);
	int height = find_int_arg(argc, argv, "-h", 0);
	int fps = find_int_arg(argc, argv, "-fps", 0);
	//int class = find_int_arg(argc, argv, "-class", 0);

	char *datacfg = argv[3];
	char *cfg = argv[4];
	char *weights = (argc > 5) ? argv[5] : 0;
	char *filename = (argc > 6) ? argv[6] : 0;
	if (0 == strcmp(argv[2], "test"))
		test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh,
				outfile, fullscreen);

	else if (0 == strcmp(argv[2], "test_radiation"))
		test_detector_radiation(datacfg, cfg, weights, filename, thresh,
				hier_thresh, outfile, fullscreen, argc, argv);

	else if (0 == strcmp(argv[2], "train"))
		train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
	else if (0 == strcmp(argv[2], "valid"))
		validate_detector(datacfg, cfg, weights, outfile);
	else if (0 == strcmp(argv[2], "valid2"))
		validate_detector_flip(datacfg, cfg, weights, outfile);
	else if (0 == strcmp(argv[2], "recall"))
		validate_detector_recall(cfg, weights);
	else if (0 == strcmp(argv[2], "demo")) {
		list *options = read_data_cfg(datacfg);
		int classes = option_find_int(options, "classes", 20);
		char *name_list = option_find_str(options, "names", "data/names.list");
		char **names = get_labels(name_list);
		demo(cfg, weights, thresh, cam_index, filename, names, classes,
				frame_skip, prefix, avg, hier_thresh, width, height, fps,
				fullscreen);
	}
	//else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
	//else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}
