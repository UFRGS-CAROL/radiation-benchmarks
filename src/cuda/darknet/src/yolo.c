#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"

#include "args.h"
#include "log_processing.h"

#include "abft.h"

#include "helpful.h"
#ifdef LOGS
#include "log_helper.h"
//#include "gemm.h"
#endif

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
//  printf("passou2\n");
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
//          printf("%f", )
            if (probs[i][j]) {
                fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j], xmin,
                        ymin, xmax, ymax);
//              printf("passou3\n");
            }
        }

    }
}

void validate_yolo(char *cfgfile, char *weightfile) {
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n",
            net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **) list_to_array(plist);

    layer l = net.layers[net.n - 1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
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
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 2;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
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
            convert_detections(predictions, classes, l.n, square, side, w, h,
                    thresh, probs, boxes, 0);
            if (nms)
                do_nms_sort(boxes, probs, side * side * l.n, classes,
                        iou_thresh);
            print_yolo_detections(fps, id, boxes, probs, side * side * l.n,
                    classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n",
            (double) (time(0) - start));
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

void test_yolo_generate(Args *arg) {
//-------------------------------------------------------------------------------
    // first I nee to treat all image files
    int img_list_size = 0;
    char **img_list = get_image_filenames(arg->img_list_path, &img_list_size);
//-------------------------------------------------------------------------------
    network net = parse_network_cfg(arg->config_file);
    if (arg->weights) {
        load_weights(&net, arg->weights);
    }
    detection_layer l = net.layers[net.n - 1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
//  char buff[256];
//  char *input = buff;
    int j;
    float nms = .4;
    box *boxes = calloc(l.side * l.side * l.n, sizeof(box));
    float **probs = calloc(l.side * l.side * l.n, sizeof(float *));
    for (j = 0; j < l.side * l.side * l.n; ++j)
        probs[j] = calloc(l.classes, sizeof(float *));

//  printf("total %d and other %d\n", l.side * l.side * l.n, l.h * l.n * l.w);
//-------------------------------------------------------------------------------
    FILE *output_file = fopen(arg->gold_inout, "w+");
    int classes = l.classes;
    int total = l.side * l.side * l.n;

    if (output_file) {
//      writing all parameters for test execution
//      thresh hier_tresh img_list_size img_list_path config_file config_data model weights total classes
        fprintf(output_file, "%f;%f;%d;%s;%s;%s;%s;%s;%d;%d;\n", arg->thresh,
                arg->hier_thresh, img_list_size, arg->img_list_path,
                arg->config_file, arg->cfg_data, arg->model, arg->weights,
                total, classes);
    } else {
        fprintf(stderr, "GOLD OPENING ERROR");
        exit(-1);
    }
    detection gold_to_save;
    gold_to_save.network_name = "darknet_v1";
    if (arg->save_layers)
        alloc_gold_layers_arrays(&gold_to_save, &net);

    //  set abft
    if (arg->abft >= 0 && arg->abft < MAX_ABFT_TYPES) {
        printf("passou no if %d\n\n", arg->abft);
#ifdef GPU
        switch (arg->abft) {
            case 1:
            set_abft_gemm(arg->abft);
            break;
            case 2:
            set_abft_smartpool(arg->abft);
            break;
        }
#endif
    }

//-------------------------------------------------------------------------------

    int i;
    for (i = 0; i < img_list_size; i++) {
        printf("generating gold for: %s\n", img_list[i]);
        image im = load_image_color(img_list[i], 0, 0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time = clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", img_list[i],
                sec(clock() - time));
        convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1,
                arg->thresh, probs, boxes, 0);
        if (nms)
            do_nms_sort(boxes, probs, l.side * l.side * l.n, l.classes, nms);

        //      must do the same thing that draw_detections
        //      but the output will be a gold file (old draw_detections)
        //      first write a filename
        fprintf(output_file, "%s;%d;%d;%d;\n", img_list[i], im.h, im.w, im.c);
        //      after writes all detection information
        //      each box is described as class number, left, top, right, bottom, prob (confidence)
        //      save_gold(FILE *fp, char *img, int num, int classes, float **probs,
        //              box *boxes)
        save_gold(output_file, img_list[i], l.side * l.side * l.n, l.classes,
                probs, boxes);

        if (arg->save_layers)
            save_layer(&gold_to_save, i, 0, "gold", 1, arg->img_list_path);

#ifdef GEN_IMG
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        draw_detections(im, l.side * l.side * l.n, arg->thresh, boxes, probs,
                voc_names, voc_labels, 20);
        char temp[100];
        sprintf(temp, "predictions_it_%d", i);
        save_image(im, temp);
        show_image(im, temp);
#endif
        free_image(im);
        free_image(sized);
    }

    //free char** memory
    for (i = 0; i < img_list_size; i++) {
        free(img_list[i]);
    }
    free(img_list);

    //close gold file
    fclose(output_file);
}

/**
 * support functions
 *resize_image(im, net.w, net.h);
 * -------------------------------------------------------------------------------------
 */
image *load_all_images_sized(image *img_array, int net_w, int net_h,
        int list_size) {
//      image sized = letterbox_image(im, net.w, net.h);
    int i;
    image *ret = (image*) malloc(sizeof(image) * list_size);
    for (i = 0; i < list_size; i++) {
        ret[i] = resize_image(img_array[i], net_w, net_h);//letterbox_image(img_array[i], net_w, net_h);
    }
    return ret;
}

//load_image_color(img_list[i], 0, 0);
image *load_all_images(detection det) {
//  image im = load_image_color(input, 0, 0);
    int i;
    image *ret = (image*) malloc(sizeof(image) * det.plist_size);
    for (i = 0; i < det.plist_size; i++) {
        ret[i] = load_image_color(det.img_names[i], 0, 0);
    }
    return ret;
}

void free_all_images(image *array, int list_size) {
    //          free_image(im);
    int i;
    for (i = 0; i < list_size; i++) {
        free_image(array[i]);
    }
}
//-------------------------------------------------------------------------------------

/**
 * Test yolo: radiation test case
 */
void test_yolo_radiation_test(Args *arg) {
//-------------------------------------------------------------------------------
    //radiation test case needs load gold first
    detection gold = load_gold(arg);
    printf("\nArgs inside detector_radiation\n");
    print_args(*arg);
    gold.network_name = "darknet_v1";
    //if abft is set these parameters will also be set
    error_return max_pool_errors;
    init_error_return(&max_pool_errors);
    //  set abft
    if (arg->abft >= 0 && arg->abft < MAX_ABFT_TYPES) {
#ifdef GPU
        switch (arg->abft) {
            case 1:
            set_abft_gemm(arg->abft);
            break;
            case 2:
            set_abft_smartpool(arg->abft);
            break;
            case 3:
            printf("%s ABFT not implemented yet\n", ABFT_TYPES[arg->abft]);
            exit(-1);
            break;
            case 4:
            printf("%s ABFT not implemented yet\n", ABFT_TYPES[arg->abft]);
            exit(-1);
            break;
            case 5:
            printf("%s ABFT not implemented yet\n", ABFT_TYPES[arg->abft]);
            exit(-1);
            break;
            default:
                printf("No ABFT was set\n");
                break;
        }
#endif
    }

//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
    network net = parse_network_cfg(arg->config_file);
    if (arg->weights) {
        load_weights(&net, arg->weights);
    }
    detection_layer l = net.layers[net.n - 1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;

    int j;
    float nms = .4;
    box *boxes = calloc(l.side * l.side * l.n, sizeof(box));
    float **probs = calloc(l.side * l.side * l.n, sizeof(float *));
    for (j = 0; j < l.side * l.side * l.n; ++j)
        probs[j] = calloc(l.classes, sizeof(float *));

//-------------------------------------------------------------------------------
    //load all images
    const image *im_array = load_all_images(gold);

    const image *im_array_sized = load_all_images_sized(im_array, net.w, net.h,
            gold.plist_size);

    //need to allocate layers arrays
    alloc_gold_layers_arrays(&gold, &net);
//  int classes = l.classes;
//  int total = l.side * l.side * l.n;
//-------------------------------------------------------------------------------

    int i, it;
    for (it = 0; it < arg->iterations; it++) {
        for (i = 0; i < gold.plist_size; i++) {

            image im = im_array[i]; //load_image_color(img_list[i], 0, 0);
            image sized = im_array_sized[i]; //resize_image(im, net.w, net.h);
            float *X = sized.data;
            time = clock();

            double time = mysecond();
            //This is the detection
            start_iteration_app();

            float *predictions = network_predict(net, X);

            convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1,
                    1, arg->thresh, probs, boxes, 0);
            if (nms)
                do_nms_sort(boxes, probs, l.side * l.side * l.n, l.classes,
                        nms);

            end_iteration_app();
            time = mysecond() - time;
//      here we test if any error happened
//          if shit happened we log
            double time_cmp = mysecond();

#ifdef GPU
            //before compare copy maxpool err detection values
            //smart pooling
            if (arg->abft == 2) {
                get_and_reset_error_detected_values(max_pool_errors);
            }
#endif
            compare(&gold, probs, boxes, l.w * l.h * l.n, l.classes, i,
                    arg->save_layers, it, arg->img_list_path, max_pool_errors);
            time_cmp = mysecond() - time_cmp;

            printf(
                    "Iteration %d - image %d predicted in %f seconds. Comparisson in %f seconds.\n",
                    it, i, time, time_cmp);

//########################################

#ifdef GEN_IMG
            //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
            draw_detections(im, l.side * l.side * l.n, arg->thresh, boxes, probs,
                    voc_names, voc_labels, 20);
            char temp[100];
            sprintf(temp, "predictions_it_%d", i);
            save_image(im, temp);
            show_image(im, temp);
#endif
            clear_boxes_and_probs(boxes, probs, l.w * l.h * l.n, l.classes);

        }
    }

    //free the memory
    free_ptrs((void **) probs, l.w * l.h * l.n);
    free(boxes);
    delete_detection_var(&gold, arg);

    free_all_images(im_array, gold.plist_size);
    free_all_images(im_array_sized, gold.plist_size);

    //free smartpool errors
    free_error_return(&max_pool_errors);
#ifdef GPU
    free_err_detected();
#endif
}

void run_yolo_rad(Args args) {
    if (args.generate_flag) {
        test_yolo_generate(&args);
    } else {
        test_yolo_radiation_test(&args);
    }
}

void run_yolo(int argc, char **argv) {
    int i;
    for (i = 0; i < 20; ++i) {
        char buff[256];
        sprintf(buff, "data/labels/%s.png", voc_names[i]);
        voc_labels[i] = load_image_color(buff, 0, 0);
    }

    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n",
                argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5] : 0;
    if (0 == strcmp(argv[2], "test")) {
        printf("passou aqui\n\n");
        test_yolo(cfg, weights, filename, thresh);
    } else if (0 == strcmp(argv[2], "train"))
        train_yolo(cfg, weights);
    else if (0 == strcmp(argv[2], "valid"))
        validate_yolo(cfg, weights);
    else if (0 == strcmp(argv[2], "recall"))
        validate_yolo_recall(cfg, weights);
    else if (0 == strcmp(argv[2], "demo"))
        demo(cfg, weights, thresh, cam_index, filename, voc_names, voc_labels,
                20, frame_skip);
}
