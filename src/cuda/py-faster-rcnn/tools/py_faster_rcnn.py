#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import pickle
import math
import traceback
import csv
import sys

#threshold for radiation test
THRESHOLD = 0.005
#threshold for  configuration and nonmaxsupression
CONF_THRESH = 0.0 #0.8
NMS_THRESH = 0.3

DATASETS = ['CALTECH_CRITICAL', 'CALTECH', 'VOC2012']

import time, calendar

# import log helper
sys.path.insert(0, '/home/carol/radiation-benchmarks/src/include/log_helper_python/')

import log_helper as lh

CLASSES = ['__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               'ZF_faster_rcnn_final.caffemodel')}

"""
originaly the thresh was 0.5, but I want get all results
so it becames 0
this function will return a dict composed by:
ret['boxes'] = a bbox list
ret['scores'] = a scores list
"""


def visDetections(dets, thresh):
    """Draw detected bounding boxes."""
    ret = {'boxes': [], 'scores': []}
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return ret

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ret['boxes'].append(bbox)
        ret['scores'].append(score)

    # ax.add_patch(
    # e(xy, width, height,
    #         plt.Rectangle((bbox[0], bbox[1]),
    #                       bbox[2] - bbox[0],
    #                       bbox[3] - bbox[1], fill=False,
    #                       edgecolor='red', linewidth=3.5)
    #         )
    #     ax.text(bbox[0], bbox[1] - 2,
    #             '{:s} {:.3f}'.format(class_name, score),
    #             bbox=dict(facecolor='blue', alpha=0.5),
    #             fontsize=14, color='white')
    #
    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()
    return ret


def detect(net, image_name, pr):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
    if pr:
        print image_name

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    if pr:
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    detectionResult = {}
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        detectionResult[cls] = visDetections(dets, thresh=CONF_THRESH)

    return detectionResult

def getDatasetImgName(imgName):
	return imgName
    #dataset = ''
    #for i in DATASETS:
    #    if i in imgName:
    #        dataset = i
    #        break
#    list = imgName.split(dataset)
#   return str(dataset + str(list[1]))

# def detect(net, image_name, pr):
#     """Detect object classes in an image using pre-computed object proposals."""
#     #will return a hash with boxes and scores
#
#     # Load the demo image
#     im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
#     if pr:
#         print im_file
#     im = cv2.imread(im_file)
#
#     # Detect all object classes and regress object bounds
#     timer = Timer()
#     timer.tic()
#     scores, boxes = im_detect(net, im)
#     timer.toc()
#
#     if pr:
#         print ('Detection took {:.3f}s for '
#            '{:d} object proposals').format(timer.total_time, boxes.shape[0])
#     return [scores, boxes]
#
# def generate(net, image_name):
#     """Detect object classes in an image using pre-computed object proposals."""
#
#     # Load the demo image
#     #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
#     im_file = os.path.join(image_name)
#     #print im_file
#     im = cv2.imread(im_file)
#
#     # Detect all object classes and regress object bounds
#     timer = Timer()
#     timer.tic()
#     scores, boxes = im_detect(net, im)
#     timer.toc()
#
#     print ('Detection took {:.3f}s for '
#            '{:d} object proposals').format(timer.total_time, boxes.shape[0])
#
#     return [scores, boxes]


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    # radiation logs
    parser.add_argument('--ite', dest='iterations', help="number of iterations", default='1')

    parser.add_argument('--gen', dest='generate_file', help="if this var is set the gold file will be generated",
                        default="")

    parser.add_argument('--log', dest='is_log', help="is to generate logs", choices=["no_logs", "daniel_logs"],
                        default="no_logs")

    parser.add_argument('--iml', dest='img_list', help='mg list data path <text file txt, csv..>',
                        default='py_faster_list.txt')

    parser.add_argument('--gld', dest='gold', help='gold file', default='')

    args = parser.parse_args()

    return args


# write gold for pot use
def serialize_gold(filename, data):
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    except:
        print "Error on writing file"


# open gold file
def load_file(filename):
    try:
        with open(filename, "rb") as f:
            ret = pickle.load(f)
    except:
        return None
    return ret


def write_to_csv(filename, data):
    with open(filename, 'wb') as csvfile:
        spwriter = csv.writer(csvfile, delimiter=' ',
                              quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for scores, boxes in data:
            scores_m = len(scores)
            boxes_m = len(boxes)

            spwriter.writerow([scores_m, boxes_m])
            for scores_i in scores:
                scores_n = len(scores_i)
                spwriter.writerow([scores_n, "--", scores_i])

            for boxes_i in boxes:
                boxes_n = len(boxes_i)
                spwriter.writerow([boxes_n, "--", boxes_i])


# compare and return the error count and error string detail
def compare_boxes(gold, current, cls, img_name):
    # compare boxes #####################################################
    error_count = 0
    goldSize = len(gold)
    currSize = len(current)
    bbDiff = goldSize - currSize
    min_m_range = goldSize
    if bbDiff != 0:
        min_m_range = min(goldSize, currSize)
        lh.log_error_detail("img_name: " + str(img_name) + " class: " + str(cls) + "wrong_boxes_size: " + str(bbDiff))
        error_count += abs(bbDiff)

    pos = ['x1', 'y1', 'x2', 'y2']

    for i in range(0, min_m_range):
        # e(xy, width, height,
        #         plt.Rectangle((bbox[0], bbox[1]),
        #                       bbox[2] - bbox[0],
        #                       bbox[3] - bbox[1], fill=False,
        logString ="img_name: " + str(img_name) + " class: " + str(cls) + " box: [" + str(i) + "] "
        error = False
        try:
            for iGold, iCurr, k in zip(gold[i], current[i], pos):
                iG = float(iGold)
                iC = float(iCurr)
                diff = math.fabs(iG - iC)
                logString += str(k) + "_e: " + str(iG) + " " + str(k) + "_r: " + str(iC) + " "
                if diff > THRESHOLD:
                    error = True
        except:
            logString = "img_name: " + str(img_name) + " class: " + str(cls) + " box: [" + str(i) + "] loop_error"

        if error:
            lh.log_error_detail(logString)
            error_count += 1

    return error_count


# compare scores and return error count and string error detail
def compare_scores(gold, current, cls, img_name):
    error_count = 0
    goldSize = len(gold)
    currSize = len(current)
    scrDiff = goldSize - currSize
    min_m_range = goldSize
    if scrDiff != 0:
        min_m_range = min(goldSize, currSize)
        lh.log_error_detail("img_name: " + str(img_name) + " class: " + str(cls) + "wrong_score_size: " + str(scrDiff))
        error_count += abs(scrDiff)

    for i in range(0, min_m_range):
        iGold = float(gold[i])
        iCurr = float(current[i])
        diff = math.fabs(iGold - iCurr)
        if diff > THRESHOLD:
            lh.log_error_detail("img_name: " + str(img_name) + " class: " + str(cls) + " score: [" + str(i) + "] e: " + str(iGold) + " r: " + str(iCurr))
            error_count += 1

    return error_count


# compare gold against current
"""
for each image there are n classes of objects
    for each class there are n boxes and n scores
        ret['boxes'] = a bbox list
        ret['scores'] = a scores list
the compare input is the output of a single image
so only the second for is compared

"""


def compare(gold, current, img_name):
    error_count = 0
    goldKeys = gold.keys()
    currKeys = current.keys()

    goldSize = len(goldKeys)
    currSize = len(currKeys)
    size_error_m = goldSize - currSize
    if size_error_m != 0:
        lh.log_error_detail("img_name: " + str(img_name) + " missing_classes_on_detected: " + str(size_error_m))
        error_count += abs(size_error_m)

    intersection = set(goldKeys) & set(currKeys)
    tempImgName = getDatasetImgName(img_name)
    for i in intersection:
        iGold = gold[i]
        iCurr = current[i]
        bbListGold = iGold['boxes']
        bbListCurr = iCurr['boxes']
        scrListGold = iGold['scores']
        scrListCurr = iCurr['scores']

        #errorBefore = error_count
        error_count += compare_scores(scrListGold, scrListCurr, i, tempImgName)
        error_count += compare_boxes(bbListGold, bbListCurr, i, tempImgName)

        # if errorBefore != error_count:
        #     lh.log_error_info(
        #         "img_name: " + str(img_name) + " class: " + str(i) + " total_errors: " + str(error_count - errorBefore))

    return error_count


def force_update_timestamp():
    fp = open("/var/radiation-benchmarks/timestamp.txt", "w")
    fp.write(str(calendar.timegm(time.gmtime())))
    fp.close()
    return


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    force_update_timestamp()
    if "no_logs" not in args.is_log:

        string_info = "iterations: " + str(args.iterations) + " img_list: " + str(args.img_list) + " board: "
        if "X1" in args.img_list:
            string_info += "X1"
        else:
            string_info += "K40"
        lh.start_log_file("PyFasterRcnn", string_info)

    # object for gold file
    gold_file = {}
    ###################################################################################
    # only load network
    force_update_timestamp()
    try:
        # to make sure that the models and cfg will be with absolute path
        prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                                'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
        caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                                  NETS[args.demo_net][1])

        if not os.path.isfile(caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                           'fetch_faster_rcnn_models.sh?').format(caffemodel))

        if args.cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(args.gpu_id)
            cfg.GPU_ID = args.gpu_id
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        ##open gold
        if args.gold != "":
            gold_file = load_file(args.gold)
        print '\n\nLoaded network {:s}'.format(caffemodel)
    except Exception as e:
        if "no_logs" not in args.is_log:
            lh.log_error_detail("exception: error_loading_network error_info:" +
                                str(traceback.format_exception(*sys.exc_info())) + " XX " + str(
                e.__doc__) + " XX " + str(e.message))
            lh.end_log_file()
            raise
        else:
            print " XX " + str(e.__doc__) + " XX " + str(e.message)

    force_update_timestamp()
    ##after loading net we start
    try:
        ##################################################################################
        # device Warmup on a dummy image
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _ = im_detect(net, im)

            ##################################################################################
        in_names = []
        iterations = int(args.iterations)

        in_names = [line.strip() for line in open(args.img_list, 'r')]

        if args.generate_file != "":
            # execute only once
            # even in python you need initializate
            gold_file = {}
            print "Generating gold for Py-faster-rcnn"
            for im_name in in_names:
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                print 'Demo for {}'.format(im_name)
                gold_file[im_name] = detect(net, im_name, True)
                #print gold_file[im_name]
            print "Gold generated, saving file"
            serialize_gold(args.generate_file, gold_file)
            # write_to_csv(args.generate_file, gold_file)
            print "Gold save sucess"

        else:
            i = 0
            # j = 0
            im_size = len(in_names)
            while (i < iterations):
                # iterator
                # iterator = iter(gold_file)
                # total_iteration_errors = 0
                it = 0
                for im_name in in_names:
                    # item = iterator.next()
                    ###Log
                    # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                    it += 1
                    if it % 10 == 0:
                        print 'PyFaster for {}'.format(im_name)
                    if "no_logs" not in args.is_log:
                        ##start
                        lh.start_iteration()
                        ret = detect(net, im_name, (it % 10 == 0))
                        lh.end_iteration()

                        # check gold
                        timer = Timer()
                        timer.tic()
                        error_count = compare(gold_file[im_name], ret, im_name)
                        timer.toc()

                        # if error_count != 0:
                        iteration_file_pos = iterations + it
                        # scores_name = lh.get_log_file_name() +"_"+ str(iteration_file_pos) + ".scores"
                        # print scores_name
                        # if error_count != 0:
                        #     serialize_gold(scores_name,ret[0])
                        if it % 10 == 0:
                            print "Compare time ", timer.total_time, " errors ", error_count

                        lh.log_error_count(int(error_count))
                        force_update_timestamp()
                    ##end log
                    # j += 1
                    # if j == im_size:
                    #     break

                i += 1
    except Exception as e:
        if "no_logs" not in args.is_log:
            lh.log_error_detail("exception: error_network_exection error_info:" +
                                str(traceback.format_exception(*sys.exc_info())) + " XX " + str(
                e.__doc__) + " XX " + str(e.message))
            lh.end_log_file()
            raise
        else:
            print " XX " + str(e.__doc__) + " XX " + str(e.message)
    ##################################################################################
    # finish ok
    if "no_logs" not in args.is_log:
        lh.end_log_file()



        # def compare(gold, current, img_name):
        #     # scores_gold = gold[0]
        #     # boxes_gold = gold[1]
        #     # error_count = 0
        #     # #iterator for current, i need it because generate could be smaller than gold, so python will throw an exception
        #     # scores_curr = current[0]
        #     # boxes_curr = current[1]
        #     #
        #     #
        #     # CONF_THRESH = 0.8
        #     # NMS_THRESH = 0.3
        #     # error_detail_final = ""
        #     # ########################
        #     # #real compare
        #     #
        #     #
        #     # for cls_ind, cls in enumerate(CLASSES[1:]):
        #     #     cls_ind += 1 # because we skipped background
        #     #
        #     #     #for gold
        #     #     cls_boxes_gold = boxes_gold[:, 4*cls_ind:4*(cls_ind + 1)]
        #     #     cls_scores_gold = scores_gold[:, cls_ind]
        #     #     dets_gold = np.hstack((cls_boxes_gold,
        #     #                       cls_scores_gold[:, np.newaxis])).astype(np.float32)
        #     #     keep_gold = nms(dets_gold, NMS_THRESH)
        #     #     dets_gold = dets_gold[keep_gold, :]
        #     #
        #     #     #for current
        #     #     cls_boxes_curr = boxes_curr[:, 4*cls_ind:4*(cls_ind + 1)]
        #     #     cls_scores_curr = scores_curr[:, cls_ind]
        #     #     dets_curr = np.hstack((cls_boxes_curr,
        #     #                       cls_scores_curr[:, np.newaxis])).astype(np.float32)
        #     #     keep_curr = nms(dets_curr, NMS_THRESH)
        #     #     dets_curr = dets_curr[keep_curr, :]
        #     #
        #     #     #compare all results
        #     #
        #     #     error_count += compare_boxes(cls_boxes_gold,cls_boxes_curr)
        #     #     inds_gold = np.where(dets_gold[:, -1] >= 0)[0]
        #     #     inds_curr = np.where(dets_curr[:, -1] >= 0)[0]
        #     #     for i in inds_curr:
        #     #         bbox_curr = dets_curr[i, :4]
        #     #         scores_curr = dets_curr[i, -1]
        #     #         print "Scores curr\n" , bbox_curr , " " , scores_curr
        #     #
        #     #     for i in inds_gold:
        #     #         bbox_gold = dets_gold[i, :4]
        #     #         scores_gold = dets_gold[i, -1]
        #     #         print "Scores gold\n" , bbox_gold , " " , scores_gold
        #     #     error_count += compare_scores(cls_scores_gold,cls_scores_curr)
        #     #     inds_curr = np.where(dets_curr[:, -1] >=  0)[0]
        #     #     inds_gold = np.where(dets_gold[:, -1] >=  0)[0]
        #     #
        #     #     for (i_g, i_c) in zip(inds_gold, inds_curr):
        #     #         scores_gold = dets_gold[i_g, -1]
        #     #         scores_curr = dets_curr[i_c, -1]
        #     #         print "csl scores" , scores_curr
        #     #         print "csl scores gold" , scores_gold
        #
        #     if error_count > 0:
        #         lh.log_error_detail("input_img: "+img_name)
        #
        #     return error_count
