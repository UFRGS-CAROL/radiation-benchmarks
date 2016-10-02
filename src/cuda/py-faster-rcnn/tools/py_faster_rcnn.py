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

THRESHOLD = 0.0000001

#import log helper
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


def detect(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    #will return a hash with boxes and scores
   
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    print im_file
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    return [scores, boxes]

def generate(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_file = os.path.join(image_name)
    #print im_file
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
 
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    return [scores, boxes]


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

    #radiation logs
    parser.add_argument('--ite', dest='iterations', help="number of iterations", default='1')

    parser.add_argument('--gen', dest='generate_file', help="if this var is set the gold file will be generated", default="")

    parser.add_argument('--log', dest='is_log', help="is to generate logs", choices=["no_logs", "daniel_logs"], default="no_logs")

    parser.add_argument('--iml', dest='img_list', help='mg list data path <text file txt, csv..>', default='py_faster_list.txt')

    parser.add_argument('--gld',  dest='gold', help='gold file', default='')



    args = parser.parse_args()

    return args

#write gold for pot use
def serialize_gold(filename,data):
    try:
        with open(filename, "wb") as f:
                pickle.dump(data, f)
    except:
        print "Error on writing file"

#open gold file
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
        for scores,boxes in data:
            scores_m = len(scores)
            boxes_m = len(boxes)

            spwriter.writerow([scores_m, boxes_m])
            for scores_i in scores:
                scores_n =len(scores_i)
                spwriter.writerow([scores_n, "--", scores_i])

            for boxes_i in boxes:
                boxes_n = len(boxes_i)
                spwriter.writerow([boxes_n, "--", boxes_i])

##in the py-faster-original
#     for cls_ind, cls in enumerate(CLASSES[1:]):
#         cls_ind += 1 # because we skipped background
#         cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
#         cls_scores = scores[:, cls_ind]
#         dets = np.hstack((cls_boxes,
#                           cls_scores[:, np.newaxis])).astype(np.float32)
#         keep = nms(dets, NMS_THRESH)
#         dets = dets[keep, :]
# vis_detections(im, cls, dets, thresh=CONF_THRESH)


# compare gold against current
def compare(gold, current, img_name):
    scores_gold = gold[0]
    boxes_gold = gold[1]
    error_count = 0
    #iterator for current, i need it because generate could be smaller than gold, so python will throw an exception
    scores_curr = current[0]
    boxes_curr = current[1]

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background

        #for gold
        cls_boxes_gold = boxes_gold[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores_gold = scores_gold[:, cls_ind]
        dets_gold = np.hstack((cls_boxes_gold,
                          cls_scores_gold[:, np.newaxis])).astype(np.float32)
        keep_gold = nms(dets_gold, NMS_THRESH)
        dets_gold = dets_gold[keep_gold, :]

        #for current
        cls_boxes_curr = boxes_curr[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores_curr = scores_curr[:, cls_ind]
        dets_curr = np.hstack((cls_boxes_curr,
                          cls_scores_curr[:, np.newaxis])).astype(np.float32)
        keep_curr = nms(dets_curr, NMS_THRESH)
        dets_curr = dets_curr[keep_curr, :]

        print "\n\n\ndets gold\n\n"
        print cls_scores_curr
        print "\ndets curr\n"
        print cls_scores_curr


    #compare boxes #####################################################         
    min_m_range = boxes_m_gold = len(boxes_gold)
    boxes_m_curr = len(boxes_curr)
    #diff size
    size_error_m = abs(boxes_m_gold - boxes_m_curr)
    if size_error_m != 0:
        min_m_range = min(boxes_m_gold, boxes_m_curr)
        lh.log_error_detail("boxes_missing_lines: " + size_error_m)
        error_count += size_error_m
        

    for i in range(0,min_m_range):
        min_n_range = boxes_n_gold = len(boxes_gold[i])
        boxes_n_curr = len(boxes_curr[i])
        size_error_n = abs(boxes_n_gold - boxes_n_curr)
        if size_error_n != 0:
            min_n_range = min(boxes_n_gold, boxes_n_curr)
            lh.log_error_detail("boxes_missing_collumns: " + size_error_n + " line: " + i)
            error_count += size_error_m

        for j in range(0, min_n_range):
            gold_ij = float(boxes_gold[i][j])
            curr_ij = float(boxes_curr[i][j])
            diff = math.fabs(gold_ij -  curr_ij)
            if diff > THRESHOLD:
                error_detail = "boxes: [" + str(i) + "," + str(j) + "] e: " +  str(gold_ij) + " r: " + str(curr_ij)
                error_count += 1
                lh.log_error_detail(error_detail)
        
    if error_count > 0:
        lh.log_error_detail(img_name)

    return error_count

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if "no_logs" not in args.is_log:
        
        string_info = "iterations: " + str(args.iterations) + " img_list: " + str(args.img_list) + " board: "
        if "X1" in args.img_list:
            string_info += "X1"
        else:
            string_info += "K40"
        lh.start_log_file("PyFasterRcnn", string_info)
    
    #object for gold file
    gold_file = []
###################################################################################
#only load network
    try:
        #to make sure that the models and cfg will be with absolute path
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
                str(traceback.format_exception(*sys.exc_info())) + " XX " + str(e.__doc__) + " XX "+ str(e.message))
            lh.end_log_file()
            raise
        else:
            print " XX " + str(e.__doc__) + " XX "+ str(e.message)
            
    ##after loading net we start
    try:
    ##################################################################################
    #device Warmup on a dummy image
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _= im_detect(net, im)

    ##################################################################################
        in_names=[]
        iterations = 1
        
        in_names = [line.strip() for line in open(args.img_list, 'r')]

        if args.generate_file != "":
            #execute only once
            #even in python you need initializate
            gold_file = {}
            print "Generating gold for Py-faster-rcnn"
            for im_name in in_names:
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                print 'Demo for {}'.format(im_name)
                gold_file[im_name] = generate(net, im_name)

            print "Gold generated, saving file"
            serialize_gold(args.generate_file, gold_file)
            #write_to_csv(args.generate_file, gold_file)
            print "Gold save sucess"

        else:
            i = 0
            while(i < iterations):
                #iterator
                # iterator = iter(gold_file)
                for im_name in in_names:
                    # item = iterator.next()
                    ###Log
                    #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                    print 'PyFaster for data/demo/{}'.format(im_name)
                    if "no_logs" not in args.is_log:
                        ##start
                        lh.start_iteration()
                        ret=detect(net, im_name)
                        lh.end_iteration()

                        #check gold
                        timer = Timer()
                        timer.tic()
                        error_count = compare(gold_file[im_name], ret, im_name)
                        timer.toc()
                        print "Compare time " , timer.total_time , " errors " , error_count
                        lh.log_error_count(int(error_count))
                    ##end log
                i += 1
    except Exception as e:
        if "no_logs" not in args.is_log:
            lh.log_error_detail("exception: error_network_exection error_info:" +
                str(traceback.format_exception(*sys.exc_info())) + " XX " + str(e.__doc__) + " XX "+ str(e.message))
            lh.end_log_file()
            raise
        else:
            print " XX " + str(e.__doc__) + " XX "+ str(e.message)
    ##################################################################################
    #finish ok
    if "no_logs" not in args.is_log:
        lh.end_log_file()



