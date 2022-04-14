/*
 * detection_gold.c
 *
 *  Created on: 02/10/2018
 *      Author: fernando
 */
#include <cstdlib>
#include "detection_gold_w.h"
#include "detection_gold.h"
#include "darknet.h"
#include "log_processing.h"


struct detection_gold {
    void *obj;
};

detection_gold_t *create_detection_gold(int argc, char **argv, real_t thresh,
                                        real_t hier_thresh, char *img_list_path, char *config_file,
                                        char *config_data, char *model, char *weights) {
    detection_gold_t *m;
    DetectionGold *obj;

    m = (__typeof__(m)) malloc(sizeof(*m));
    obj = new DetectionGold(argc, argv, thresh, hier_thresh, img_list_path,
                            config_file, config_data, model, weights);
    m->obj = obj;
    return m;
}

void destroy_detection_gold(detection_gold_t *m) {
    if (m == nullptr)
        return;
    delete static_cast<DetectionGold *>(m->obj);
    free(m);
}

int run(detection_gold_t *m, detection *dets, int nboxes, int img_index,
        int classes, int img_w, int img_h) {
    DetectionGold *obj;
    if (m == nullptr)
        return 0;

    obj = static_cast<DetectionGold *>(m->obj);
    return obj->run(dets, nboxes, img_index, classes, img_w, img_h);
}

void start_iteration_wrapper(detection_gold_t *m) {
//	DetectionGold *obj;
    if (m == nullptr)
        return;
//	obj = static_cast<DetectionGold *>(m->obj);
    Log::start_iteration_app();
}

void end_iteration_wrapper(detection_gold_t *m) {
//	DetectionGold *obj;
    if (m == nullptr)
        return;

//	obj = static_cast<DetectionGold *>(m->obj);
    Log::end_iteration_app();

}

int get_iterations(detection_gold_t *m) {
    DetectionGold *obj;
    if (m == nullptr)
        return 0;

    obj = static_cast<DetectionGold *>(m->obj);
    int it = obj->iterations;
    return it;
}

int get_img_num(detection_gold_t *m) {
    DetectionGold *obj;
    if (m == nullptr)
        return 0;

    obj = static_cast<DetectionGold *>(m->obj);
    int it = obj->plist_size;
    return it;
}

unsigned char get_use_tensor_cores(detection_gold_t *m) {
    DetectionGold *obj;
    if (m == nullptr)
        return 0;

    obj = static_cast<DetectionGold *>(m->obj);
    unsigned char it = obj->tensor_core_mode;
    return it;
}

//int get_smx_redundancy(detection_gold_t *m) {
//    DetectionGold *obj;
//    if (m == nullptr)
//        return 0;
//
//    obj = static_cast<DetectionGold *>(m->obj);
////    int it = obj->stream_mr;
////    return it;
//    return 0;
//}
