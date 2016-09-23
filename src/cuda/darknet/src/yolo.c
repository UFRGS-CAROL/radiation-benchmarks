#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "yolo.h"
#include <stdlib.h>
#include "list.h"
#include "log_processing.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

#ifdef LOGS
#include "log_helper.h"
#include "helpful.h"
#endif

char *voc_names[] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
		"car", "cat", "chair", "cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant", "sheep", "sofa", "train",
		"tvmonitor" };
image voc_labels[20];

void train_yolo(char *cfgfile, char *weightfile) {
	char *train_images = "/data/voc/train.txt";
	char *backup_directory = "/home/pjreddie/backup/";
	srand(time(0));
	data_seed = time(0);
	char *base = basecfg(cfgfile);
	printf("%s\n", base);
	float avg_loss = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate,
			net.momentum, net.decay);
	int imgs = net.batch * net.subdivisions;
	int i = *net.seen / imgs;
	data train, buffer;

	layer l = net.layers[net.n - 1];

	int side = l.side;
	int classes = l.classes;
	float jitter = l.jitter;

	list *plist = get_paths(train_images);
	//int N = plist->size;
	char **paths = (char **) list_to_array(plist);

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	args.jitter = jitter;
	args.num_boxes = side;
	args.d = &buffer;
	args.type = REGION_DATA;

	pthread_t load_thread = load_data_in_thread(args);
	clock_t time;
	//while(i*imgs < N*120){
	while (get_current_batch(net) < net.max_batches) {
		i += 1;
		time = clock();
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data_in_thread(args);

		printf("Loaded: %lf seconds\n", sec(clock() - time));

		time = clock();
		float loss = train_network(net, train);
		if (avg_loss < 0)
			avg_loss = loss;
		avg_loss = avg_loss * .9 + loss * .1;

		printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss,
				avg_loss, get_current_rate(net), sec(clock() - time), i * imgs);
		if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
	}
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}

void convert_detections(float *predictions, int classes, int num, int square,
		int side, int w, int h, float thresh, float **probs, box *boxes,
		int only_objectness) {
	int i, j, n;
	//int per_cell = 5*num+classes;
	for (i = 0; i < side * side; ++i) {
		int row = i / side;
		int col = i % side;
		for (n = 0; n < num; ++n) {
			int index = i * num + n;
			int p_index = side * side * classes + i * num + n;
			float scale = predictions[p_index];
			int box_index = side * side * (classes + num) + (i * num + n) * 4;
			boxes[index].x = (predictions[box_index + 0] + col) / side * w;
			boxes[index].y = (predictions[box_index + 1] + row) / side * h;
			boxes[index].w = pow(predictions[box_index + 2], (square ? 2 : 1))
					* w;
			boxes[index].h = pow(predictions[box_index + 3], (square ? 2 : 1))
					* h;
			for (j = 0; j < classes; ++j) {
				int class_index = i * classes;
				float prob = scale * predictions[class_index + j];
				probs[index][j] = (prob > thresh) ? prob : 0;
			}
			if (only_objectness) {
				probs[index][0] = scale;
			}
		}
	}
}

void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs,
		int total, int classes, int w, int h) { //Args parameter added to generate or check gold output
	int i, j;
	for (i = 0; i < total; ++i) {
		float xmin = boxes[i].x - boxes[i].w / 2.;
		float xmax = boxes[i].x + boxes[i].w / 2.;
		float ymin = boxes[i].y - boxes[i].h / 2.;
		float ymax = boxes[i].y + boxes[i].h / 2.;

		if (xmin < 0)
			xmin = 0;
		if (ymin < 0)
			ymin = 0;
		if (xmax > w)
			xmax = w;
		if (ymax > h)
			ymax = h;

		for (j = 0; j < classes; ++j) {
			if (probs[i][j]) {
				//saving the gold files
				fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j], xmin,
						ymin, xmax, ymax);
			}
		}

	}
}

void allocate_yolo_arrays(const long plist_size, const long total_size,
		int classes, box* boxes, box* boxes_gold, float*** probs,
		float*** probs_gold) {
	int j;
	if (boxes == NULL || boxes_gold == NULL || probs == NULL
			|| probs_gold == NULL) {
		fprintf(stderr, "ERROR ON ALLOCATING MEMORY\n");
		exit(EXIT_FAILURE);
	}
	int z;
	//man this shit sucks, I love C++ and JAVA
	for (z = 0; z < plist_size; z++) {
		probs[z] = calloc(total_size, sizeof(float*));
		probs_gold[z] = calloc(total_size, sizeof(float*));
		if (probs[z] == NULL || probs_gold[z] == NULL) {
			fprintf(stderr, "ERROR ON ALLOCATING MEMORY\n");
			exit(EXIT_FAILURE);
		}
		for (j = 0; j < total_size; j++) {
			probs[z][j] = calloc(classes, sizeof(float));
			probs_gold[z][j] = calloc(classes, sizeof(float));
			if (probs[z][j] == NULL || probs_gold[z][j] == NULL) {
				fprintf(stderr, "ERROR ON ALLOCATING MEMORY\n");
				exit(EXIT_FAILURE);
			}
		}
	}
}

void free_yolo_memory(const long plist_size, const long total_size, FILE* gold,
		float*** probs, float*** probs_gold, box* boxes, box* boxes_gold) {
	int j;
	//closing the gold input/output
	fclose(gold);
	int z = 0;
	for (z = 0; z < plist_size; z++) {
		for (j = 0; j < total_size; j++) {
			free(probs[z][j]);
			free(probs_gold[z][j]);
		}
		free(probs[z]);
		free(probs_gold[z]);
	}
	free(probs);
	free(probs_gold);
	free(boxes);
	free(boxes_gold);
}

void validate_yolo(const Args arg) { //char *cfgfile, char *weightfile, char *img_list_path, char *base_result_out) {
	network net = parse_network_cfg(arg.config_file);
	if (arg.weights) {
		load_weights(&net, arg.weights);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n",
			net.learning_rate, net.momentum, net.decay);
	srand(time(0));

//	char *base = "results/comp4_det_test_";
	char *base = arg.base_result_out;
	//list *plist = get_paths("data/voc.2007.test");
	//list *plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");
	//list *plist = get_paths("data/voc.2012.test");
	list *plist = get_paths(arg.img_list_path);
	char **paths = (char **) list_to_array(plist);

	layer l = net.layers[net.n - 1];
	int classes = l.classes;
	int square = l.sqrt;
	int side = l.side;

	//for gold generation or test
//	Gold gold;
//	gold_malloc(&gold, classes);

	int j;
	FILE **fps = calloc(classes, sizeof(FILE *));
	FILE *gold;

	for (j = 0; j < classes; ++j) {
		char buff[1024];

		snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
		fps[j] = fopen(buff, "w");
	}

	const long total_size = side * side * l.n;
	const long plist_size = plist->size;
	const long boxes_size = plist_size * total_size;
	box *boxes = calloc(boxes_size, sizeof(box)); //= calloc(side * side * l.n, sizeof(box));
	box *boxes_gold = calloc(boxes_size, sizeof(box));
	float ***probs = calloc(plist_size, sizeof(float**)); // = calloc(side * side * l.n, sizeof(float *));
	float ***probs_gold = calloc(plist_size, sizeof(float));
	allocate_yolo_arrays(plist_size, total_size, classes, boxes, boxes_gold,
			probs, probs_gold);
	int m = plist->size;
	int i = 0;
	int t;

	float thresh = .001;
	int nms = 1;
	float iou_thresh = .5;

//gold vars
	int total_gold, classes_gold, w_gold, h_gold;
//to catch all ids correctly
	char id_gold[plist->size][100]; //its the name of image

	int gold_sizes = side * side * l.n;

//if generating a gold
	if (arg.generate_flag) {
		if ((gold = fopen(arg.gold_output, "wb"))) {
			printf("generating gold file\n");
		} else {
			printf("error in opening output file\n");
			exit(EXIT_FAILURE);
		}
	} else {
		if (gold = fopen(arg.gold_input, "rb")) {
			printf("reading gold file\n");
			read_yolo_gold(gold, boxes_gold, boxes_size, probs_gold, plist_size,
					total_size, classes);
			printf("Gold file read\n");
		} else {
			printf("error in opening input file\n");
			exit(EXIT_FAILURE);
		}
	}
	int nthreads = 2;
	image *val = calloc(nthreads, sizeof(image));
	if (val == NULL) {
		printf("erro val\n");
		exit(EXIT_FAILURE);
	}
	image *val_resized = calloc(nthreads, sizeof(image));
	if (val_resized == NULL) {
		printf("Erro val resized\n");
		exit(EXIT_FAILURE);
	}
	image *buf = calloc(nthreads, sizeof(image));
	if (buf == NULL) {
		printf("Erro no buf\n");
		exit(EXIT_FAILURE);
	}
	image *buf_resized = calloc(nthreads, sizeof(image));
	if (buf_resized == NULL) {
		printf("Erro buf resised\n");
		exit(EXIT_FAILURE);
	}
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.type = IMAGE_DATA;

	for (t = 0; t < nthreads; ++t) {
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}

	int it;
	int gold_iterator = 0;
	for (it = 0; it < arg.iterations; it++) {
		time_t start = time(0);
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
				float *X = val_resized[t].data;

				float *predictions = network_predict(net, X);
				int w = val[t].w;
				int h = val[t].h;

				convert_detections(predictions, classes, l.n, square, side, w,
						h, thresh, probs[gold_iterator],
						(boxes + (gold_iterator * total_size)), 0);

				if (nms)
					do_nms_sort((boxes + (gold_iterator * total_size)),
							probs[gold_iterator], side * side * l.n, classes,
							iou_thresh);
				//now will save gold or keep running with log files

				if (arg.generate_flag) {
					print_yolo_detections(fps, id,
							(boxes + (gold_iterator * total_size)),
							probs[gold_iterator], side * side * l.n, classes, w,
							h);

					write_yolo_gold(gold, boxes, boxes_size, probs, plist_size,
							total_size, classes);
					printf("gold it %d id %s writen t = %d i = %d\n",
							gold_iterator, id, t, i);
				} else {
					print_yolo_detections(fps, id,
							boxes_gold + (gold_iterator * total_size),
							probs_gold[gold_iterator], side * side * l.n, classes, w,
							h);
					printf("gold it %d id %s writen t = %d i = %d\n",
							gold_iterator, id, t, i);
				}
				gold_iterator++;
				free(id);
				free_image(val[t]);
				free_image(val_resized[t]);
			}
		}

		fprintf(stderr, "Total Iteration %d Detection Time: %f Seconds\n", it,
				(double) (time(0) - start));
	}

//closing the gold input/output
	free_yolo_memory(plist_size, total_size, gold, probs, probs_gold, boxes,
			boxes_gold);
}

void validate_yolo_recall(char *cfgfile, char *weightfile) {
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n",
			net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	char *base = "results/comp4_det_test_";
	list *plist = get_paths("data/voc.2007.test");
	char **paths = (char **) list_to_array(plist);

	layer l = net.layers[net.n - 1];
	int classes = l.classes;
	int square = l.sqrt;
	int side = l.side;

	int j, k;
	FILE **fps = calloc(classes, sizeof(FILE *));
	for (j = 0; j < classes; ++j) {
		char buff[1024];
		snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
		fps[j] = fopen(buff, "w");
	}
	box *boxes = calloc(side * side * l.n, sizeof(box));
	float **probs = calloc(side * side * l.n, sizeof(float *));
	for (j = 0; j < side * side * l.n; ++j)
		probs[j] = calloc(classes, sizeof(float *));

	int m = plist->size;
	int i = 0;

	float thresh = .001;
	float iou_thresh = .5;
	float nms = 0;

	int total = 0;
	int correct = 0;
	int proposals = 0;
	float avg_iou = 0;

	for (i = 0; i < m; ++i) {
		char *path = paths[i];
		image orig = load_image_color(path, 0, 0);
		image sized = resize_image(orig, net.w, net.h);
		char *id = basecfg(path);
		float *predictions = network_predict(net, sized.data);
		convert_detections(predictions, classes, l.n, square, side, 1, 1,
				thresh, probs, boxes, 1);
		if (nms)
			do_nms(boxes, probs, side * side * l.n, 1, nms);

		char *labelpath = find_replace(path, "images", "labels");
		labelpath = find_replace(labelpath, "JPEGImages", "labels");
		labelpath = find_replace(labelpath, ".jpg", ".txt");
		labelpath = find_replace(labelpath, ".JPEG", ".txt");

		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		for (k = 0; k < side * side * l.n; ++k) {
			if (probs[k][0] > thresh) {
				++proposals;
			}
		}
		for (j = 0; j < num_labels; ++j) {
			++total;
			box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
			float best_iou = 0;
			for (k = 0; k < side * side * l.n; ++k) {
				float iou = box_iou(boxes[k], t);
				if (probs[k][0] > thresh && iou > best_iou) {
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
				correct, total, (float) proposals / (i + 1),
				avg_iou * 100 / total, 100. * correct / total);
		free(id);
		free_image(orig);
		free_image(sized);
	}
}

void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh) {

	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	detection_layer l = net.layers[net.n - 1];
	set_batch_network(&net, 1);
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms = .5;
	box *boxes = calloc(l.side * l.side * l.n, sizeof(box));
	float **probs = calloc(l.side * l.side * l.n, sizeof(float *));
	for (j = 0; j < l.side * l.side * l.n; ++j)
		probs[j] = calloc(l.classes, sizeof(float *));
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
		image sized = resize_image(im, net.w, net.h);
		float *X = sized.data;
		time = clock();
		float *predictions = network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
		convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1,
				thresh, probs, boxes, 0);
		if (nms)
			do_nms_sort(boxes, probs, l.side * l.side * l.n, l.classes, nms);
		//draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
		draw_detections(im, l.side * l.side * l.n, thresh, boxes, probs,
				voc_names, voc_labels, 20);
		save_image(im, "predictions");
		show_image(im, "predictions");

		show_image(sized, "resized");
		free_image(im);
		free_image(sized);
#ifdef OPENCV
		cvWaitKey(0);
		cvDestroyAllWindows();
#endif
		if (filename)
			break;
	}
}

//void run_yolo(int argc, char **argv) {
//	int i;
//	for (i = 0; i < 20; ++i) {
//		char buff[256];
//		sprintf(buff, "data/labels/%s.png", voc_names[i]);
//		voc_labels[i] = load_image_color(buff, 0, 0);
//	}
//
//	float thresh = find_float_arg(argc, argv, "-thresh", .2);
//	int cam_index = find_int_arg(argc, argv, "-c", 0);
//	int frame_skip = find_int_arg(argc, argv, "-s", 0);
//	if (argc < 4) {
//		fprintf(stderr,
//				"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n",
//				argv[0], argv[1]);
//		return;
//	}
//
//	char *cfg = argv[3];
//	char *weights = (argc > 4) ? argv[4] : 0;
//	char *filename = (argc > 5) ? argv[5] : 0;
//	if (0 == strcmp(argv[2], "test"))
//		test_yolo(cfg, weights, filename, thresh);
//	else if (0 == strcmp(argv[2], "train"))
//		train_yolo(cfg, weights);
//	else if (0 == strcmp(argv[2], "valid"))
//		validate_yolo(cfg, weights);
//	else if (0 == strcmp(argv[2], "recall"))
//		validate_yolo_recall(cfg, weights);
//	else if (0 == strcmp(argv[2], "demo"))
//		demo(cfg, weights, thresh, cam_index, filename, voc_names, voc_labels,
//				20, frame_skip);
//}

void run_yolo(const Args args) {
	int i;
	for (i = 0; i < 20; ++i) {
		char buff[256];
		sprintf(buff, "data/labels/%s.png", voc_names[i]);
		voc_labels[i] = load_image_color(buff, 0, 0);
	}

//need to be programed
	float thresh = args.thresh; //find_float_arg(argc, argv, "-thresh", .2);
	int cam_index = args.cam_index; //find_int_arg(argc, argv, "-c", 0);
	int frame_skip = args.frame_skip; //find_int_arg(argc, argv, "-s", 0);
//	if (argc < 4) {
//		fprintf(stderr,
//				"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n",
//				argv[0], argv[1]);
//		return;
//	}

//char *cfg = args.config_file; //argv[3];
//char *weights = args.weights; //(argc > 4) ? argv[4] : 0;
//test need be configured on parameters
//char *filename = args.test_filename; //(argc > 5) ? argv[5] : 0;

	if (0 == strcmp(args.execution_model, "test")) {
		//test_yolo(cfg, weights, filename, thresh);
		printf("function call not done yet\n");

	}
	if (0 == strcmp(args.execution_model, "train")) {
		//train_yolo(cfg, weights);
	}
	if (0 == strcmp(args.execution_model, "valid")) {
		validate_yolo(args);
	}
	if (0 == strcmp(args.execution_model, "recall")) {
//		validate_yolo_recall(cfg, weights);
		printf("function call not done yet\n");
	}
	if (0 == strcmp(args.execution_model, "demo")) {
//		demo(cfg, weights, thresh, cam_index, filename, voc_names, voc_labels,
//				20, frame_skip);
		printf("function call not done yet\n");
	}
}
