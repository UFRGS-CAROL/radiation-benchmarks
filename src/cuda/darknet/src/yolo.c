#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"

#include "args.h"
#include "log_processing.h"


#define min(X,Y) (((X) < (Y)) ? (X) : (Y))
#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
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

	args.angle = net.angle;
	args.exposure = net.exposure;
	args.saturation = net.saturation;
	args.hue = net.hue;

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
		int total, int classes, int w, int h) {
//	printf("passou2\n");
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
//			printf("%f", )
			if (probs[i][j]) {
				fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j], xmin,
						ymin, xmax, ymax);
//				printf("passou3\n");
			}
		}

	}
}

void free_yolo_test_memory(const Args* parameters, GoldPointers* current_ptr,
		GoldPointers* gold_ptr, int classes, image* val, image* val_resized,
		image* buf, image* buf_resized, FILE** fps) {
	//save gold values
	if (parameters->generate_flag) {
		gold_pointers_serialize(*current_ptr);
	}
	//for normal execution
	free_gold_pointers(&*current_ptr);
	if (!parameters->generate_flag)
		free_gold_pointers(&*gold_ptr);

	free(val);
	free(val_resized);
	free(buf);
	free(buf_resized);
	int cf;
	//	printf("passou antes do fclose\n");
	for (cf = 0; cf < classes; ++cf) {
		if (fps[cf] != NULL)
			fclose(fps[cf]);
	}
}

void validate_yolo(Args parameters) {
	network net = parse_network_cfg(parameters.config_file);
	if (parameters.weights) {
		load_weights(&net, parameters.weights);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n",
			net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	//result output and image list file
	char *base = parameters.base_result_out; //"gold/comp4_det_test_";
	list *plist = get_paths(parameters.img_list_path); //"voc.2012.test");
	char **paths = (char **) list_to_array(plist);

	//neural network stuff
	layer l = net.layers[net.n - 1];
	int classes = l.classes;
	int square = l.sqrt;
	int side = l.side;

	int j;
	//classes outputs files
	FILE **fps = calloc(classes, sizeof(FILE *));
	if (parameters.generate_flag) {
		for (j = 0; j < classes; ++j) {
			char buff[1024];
			snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
			fps[j] = fopen(buff, "w");
		}
	}

	//boxes and probabilities arrays
	GoldPointers current_ptr, gold_ptr;
	int gold_iterator = 0;

	if (parameters.generate_flag) {
		current_ptr = new_gold_pointers(classes, side * side * l.n, plist->size,
				parameters.gold_output, "wb");
	} else {
		//only gold_ptr need open a file
		current_ptr = new_gold_pointers(classes, side * side * l.n, plist->size,
				"not_open", "not_open");

		gold_ptr = new_gold_pointers(classes, side * side * l.n, plist->size,
				parameters.gold_input, "rb");
		//now we can already load gold values
		read_yolo_gold(&gold_ptr);
	}

	int m = plist->size;
	int i = 0;
	int t;

	float thresh = .001;
	int nms = 1;
	float iou_thresh = .5;

	int nthreads = 1;
	if (m > 1 && m <= 4) {
		nthreads = min(4, m);
	}

	image *val = calloc(nthreads, sizeof(image));
	image *val_resized = calloc(nthreads, sizeof(image));
	image *buf = calloc(nthreads, sizeof(image));
	image *buf_resized = calloc(nthreads, sizeof(image));
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));
	long iterator;
	long it = 0;

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

	printf("Images opening\n");
//	for (iterator = 0; iterator < parameters.iterations; iterator++) {
//#ifdef LOGS
//		if(!parameters.generate_flag) {
//			start_iteration();
//		}
//#endif
//		time_t start = time(0);

//	for (i = nthreads; i < m + nthreads; i += nthreads) {
	for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
		pthread_join(thr[t], 0);
		val[t] = buf[t];
		val_resized[t] = buf_resized[t];
	}

	printf("Images opening\n");
	for (t = 0; t < nthreads && i + t < m; ++t) {
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}

//	}
	for (iterator = 0; iterator < parameters.iterations; iterator++) {
		long max_err_per_iteration = 0;
//		printf("passou\n");
		double det_start = mysecond();
		for (i = nthreads; i < m + nthreads; i += nthreads) {

			for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
#ifdef LOGS
				if(!parameters.generate_flag) {
					start_iteration();
				}
#endif

				//for abft, because it is easier use an input parameter than a gcc macro
				if(parameters.abft){
					shared_errors.row_detected_errors = 0;
					shared_errors.col_detected_errors = 0;
					use_abft = 1;
				}
				double begin2 = mysecond();
				char *path = paths[i + t - nthreads];
				char *id = basecfg(path);
				float *X = val_resized[t].data;

				float *predictions;

				if (parameters.generate_flag) {
					predictions = network_predict(net, X, 1);
				} else {
					predictions = network_predict(net, X, 0);
				}

				//float *predictions = network_predict(net, X,0);

				int w = val[t].w;
				int h = val[t].h;
				ProbArray gold, current = current_ptr.pb_gold[gold_iterator];
				if (!parameters.generate_flag)
					gold = gold_ptr.pb_gold[gold_iterator];

				float **probs_curr = current.probs;
				box *boxes_curr = current_ptr.pb_gold[gold_iterator].boxes;

				convert_detections(predictions, classes, l.n, square, side, w,
						h, thresh, probs_curr, boxes_curr, 0);
				if (nms) {
					do_nms_sort(boxes_curr, probs_curr, side * side * l.n,
							classes, iou_thresh);
				}

//				printf("%f %f\n")
				if (parameters.generate_flag) {
					//	print_yolo_detections(fps, id,
					//			current_ptr.pb_gold[gold_iterator].boxes,
					//			current_ptr.pb_gold[gold_iterator].probs,
					//			side * side * l.n, classes, w, h);
				}

				//---------------------------------

#ifdef LOGS
				if(!parameters.generate_flag) {
					end_iteration();
				}
#endif
				unsigned long cmp = 0;
				//I need compare things here not anywhere else
				if (!parameters.generate_flag) {
					double begin = mysecond();
					if ((cmp = prob_array_comparable_and_log(gold, current,
							gold_iterator))) {
						fprintf(stderr,
								"%d errors found in the computation, run to the hills\n",
								cmp);
						saveLayer(net);
						max_err_per_iteration += cmp;
						if (max_err_per_iteration > 500) {
							free_yolo_test_memory(&parameters, &current_ptr,
									&gold_ptr, classes, val, val_resized, buf,
									buf_resized, fps);

#ifdef LOGS
						if(parameters.abft == 1){
							if(shared_errors.row_detected_errors || shared_errors.col_detected_errors) {
								char abft_string[500];
								fprintf(abft_string, "dumb_abft row_detected_errors: %ll col_detected_errors: %ll",
										shared_errors.row_detected_errors, shared_errors.col_detected_errors);
								log_error_detail(abft_string);
							}
						}

							if (!parameters.generate_flag) {
								log_error_count(cmp);
							}
#endif
						}

					}

					if ((i % 10) == 0) {
						fprintf(stdout,
								"Partial it %ld Gold comp Time: %fs Iteration done %3.2f\n",
								iterator, mysecond() - begin,
								((float) i / (float) m) * 100.0);
					}
//					printf("antes do clean");
					clear_vectors(&current_ptr);
					//			printf("passou\n");
				}
//				printf("passou %d %d\n");
#ifdef LOGS
				if (!parameters.generate_flag) {
					log_error_count(cmp);
				}
#endif
//				printf("passou %d %d\n");
//				printf("passou %d %d\n", gold_iterator, it++);
				gold_iterator = (gold_iterator + 1) % plist->size;

				//---------------------------------
				//printf("it %d seconds %f\n", iterator, mysecond() - begin2);
				if (iterator == parameters.iterations - 1 && (i >= m)) {
					//	printf("aqui\n");
					free(id);
					free_image(val[t]);
					free_image(val_resized[t]);
				}
			}
		}
		fprintf(stdout, "Total Detection Time: %f Seconds\n",
				(double) (mysecond() - det_start));

//		unsigned long cmp = 0;
//		//I need compare things here not anywhere else
//		if (!parameters.generate_flag) {
//			double begin = mysecond();
//			if ((cmp = comparable_and_log(gold_ptr, current_ptr)))
//				fprintf(stderr,
//						"%d errors found in the computation, run to the hills\n",
//						cmp);
//			fprintf(stdout,
//					"Iteration %ld Total Gold comparison Time: %f Seconds\n",
//					iterator, mysecond() - begin);
////			clear_vectors(&current_ptr);
////			printf("passou\n");
//
//		}

		//-----------------------------------------------
		for (t = 0; t < nthreads; ++t)
			pthread_join(thr[t], 0);
	}

//save gold values
	free_yolo_test_memory(&parameters, &current_ptr, &gold_ptr, classes, val,
			val_resized, buf, buf_resized, fps);
//
//	printf("passou depois do fclose\n");
//	printf("Yolo finished\n");
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
		float *predictions = network_predict(net, sized.data, 0);
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
	float nms = .4;
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
		float *predictions = network_predict(net, X, 0);
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

void run_yolo(Args args) {
	int i;
	for (i = 0; i < 20; ++i) {
		char buff[1000];
		sprintf(buff, "%s/data/labels/%s.png", args.base_result_out,
				voc_names[i]);
		voc_labels[i] = load_image_color(buff, 0, 0);
	}

//float thresh = find_float_arg(argc, argv, "-thresh", .2);
//int cam_index = find_int_arg(argc, argv, "-c", 0);
//int frame_skip = find_int_arg(argc, argv, "-s", 0);
//if(argc < 4){
//    fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
//    return;
//}

//    char *cfg = args.config_file;
//    char *weights = args.weights;//(argc > 4) ? argv[4] : 0;
//    char *filename = a//(argc > 5) ? argv[5]: 0;
//if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
//else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights);
	if (0 == strcmp(args.execution_model, "valid"))
		validate_yolo(args);
//else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
//else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, voc_labels, 20, frame_skip);
}
