#!/usr/bin/env python

import re

import numpy as np
import copy
import PrecisionAndRecall
import math
import os

# class for reading Darknet gold file
# from IPython.utils.py3compat import which

# need read onc
# darknet_gold_content = {}
# pyfaster_gold_contend = {}
import time

from PIL import Image

GOLD_DIR = "/home/fernando/Dropbox/UFRGS/Pesquisa/LANSCE_2016_PARSED/Gold_CNNs/"
THRESHOLD = 0.5

CONF_THRESH = 0.8
NMS_THRESH = 0.3

CLASSES = ['__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


def getClassBoxes(cls_boxes,cls_scores):
    NMS_THRESH = 0.3
    finalClassBoxes = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        # cls_ind += 1 # because we skipped background
        # for gold
        # cls_boxes_gold = boxes_gold[:, 4*cls_ind:4*(cls_ind + 1)]
        # cls_scores_gold = scores_gold[:, cls_ind]
        dets = np.hstack((cls_boxes[cls_ind],
                          cls_scores[cls_ind][:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        finalClassBoxes.append(dets)

    return finalClassBoxes


def getCls(gold):
    scores = gold[0]
    boxes = gold[1]

    classBoxes = []
    classScores = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        # dets = np.hstack((cls_boxes,
        #                   cls_scores[:, np.newaxis])).astype(np.float32)
        # keep = py_cpu_nms(dets, NMS_THRESH)
        # dets = dets[keep, :]
        classBoxes.append(cls_boxes)
        classScores.append(cls_scores)

    # for cls_ind, cls in enumerate(CLASSES[1:]):
    #     cls_ind += 1  # because we skipped background
    #
    #     # for gold
    #     cls_boxes_gold = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    #     dets_gold = np.hstack((cls_boxes_gold,
    #                            scores[:, np.newaxis])).astype(np.float32)
    #     keep_gold = py_cpu_nms(dets_gold, NMS_THRESH)
    #     dets_gold = dets_gold[keep_gold, :]
    #     classBoxes.append(dets_gold)

    return classBoxes,classScores

# parse Darknet
# returns a dictionary
def parseErrDarknet(errString):
    # image_list_position: [6] boxes: [41]  x_r: 4.7665621948242188e+02 x_e: 4.7665621948242188e+02 x_diff: 0.0000000000000000e+00 y_r: 1.1905993652343750e+02
    # y_e: 1.1905993652343750e+02 y_diff: 0.0000000000000000e+00 w_r: 3.5675010681152344e+01 w_e: 3.5675010681152344e+01 w_diff: 0.0000000000000000e+00 h_r: 8.6992042541503906e+01
    # h_e: 8.6992027282714844e+01 h_diff: 1.5258789062500000e-05
    # ERR probs: [41,4]  prob_r: 1.8973023397848010e-03 prob_e: 1.8973024562001228e-03
    ret = {}
    if 'image_list_position' in errString:
        image_err = re.match(
            ".*image_list_position\: \[(\d+)\].*boxes\: \[(\d+)\].*x_r\: (\S+).*x_e\: (\S+).*x_diff\: (\S+).*y_r\: (\S+).*y_e\: (\S+).*y_diff\: (\S+).*w_r\: (\S+).*w_e\: (\S+).*w_diff\: (\S+).*h_r\: (\S+).*h_e\: (\S+).*h_diff\: (\S+).*",
            errString)
    else:
        image_err = re.match(".*(\S+).*boxes\: \[(\d+)\].*x_r\: (\S+).*x_e\: (\S+).*x_diff\: (\S+).*y_r\: (\S+).*y_e\: (\S+).*y_diff\: (\S+).*w_r\: (\S+).*w_e\: (\S+).*w_diff\: (\S+).*h_r\: (\S+).*h_e\: (\S+).*h_diff\: (\S+).*",
            errString)


    if image_err:
        ret["type"] = "boxes"
        ret["image_list_position"] = image_err.group(1)
        ret["boxes"] = image_err.group(2)
        # x
        ret["x_r"] = image_err.group(3)
        ret["x_e"] = image_err.group(4)
        ret["x_diff"] = image_err.group(5)
        try:
            long(float(ret["x_r"]))
        except:
            ret["x_r"] = 1e30

        try:
            long(float(ret["x_e"]))
        except:
            ret["x_e"] = 1e30

        try:
            long(float(ret["x_diff"]))
        except:
            ret["x_diff"] = 1e30

        # y
        ret["y_r"] = image_err.group(6)
        ret["y_e"] = image_err.group(7)
        ret["y_diff"] = image_err.group(8)
        try:
            long(float(ret["y_r"]))
        except:
            ret["y_r"] = 1e30

        try:
            long(float(ret["y_e"]))
        except:
            ret["y_e"] = 1e30

        try:
            long(float(ret["y_diff"]))
        except:
            ret["y_diff"] = 1e30


        # w
        ret["w_r"] = image_err.group(9)
        ret["w_e"] = image_err.group(10)
        ret["w_diff"] = image_err.group(11)
        try:
            long(float(ret["w_r"]))
        except:
            ret["w_r"] = 1e30

        try:
            long(float(ret["w_e"]))
        except:
            ret["w_e"] = 1e30

        try:
            garbage = long(float(ret["w_diff"]))
        except:
            ret["w_diff"] = 1e30



        # h
        ret["h_r"] = image_err.group(12)
        ret["h_e"] = image_err.group(13)
        ret["h_diff"] = image_err.group(14)
        try:
            garbage = long(float(ret["h_r"]))
        except:
            ret["h_r"] = 1e30

        try:
            garbage = long(float(ret["h_e"]))
        except:
            ret["h_e"] = 1e30

        try:
            garbage = long(float(ret["h_diff"]))
        except:
            ret["h_diff"] = 1e30

        return ret if len(ret) > 0 else None


    if 'image_list_position' in errString:
        image_err = re.match(".*image_list_position\: \[(\d+)\].*probs\: \[(\d+),(\d+)\].*prob_r\: ([0-9e\+\-\.]+).*prob_e\: ([0-9e\+\-\.]+).*",
                         errString)
    else:
        image_err = re.match(
            ".*(\S+).*probs\: \[(\d+),(\d+)\].*prob_r\: ([0-9e\+\-\.]+).*prob_e\: ([0-9e\+\-\.]+).*",
            errString)

    if image_err:
        ret["type"] = "probs"
        ret["image_list_position"] = image_err.group(1)
        ret["probs_x"] = image_err.group(2)
        ret["probs_y"] = image_err.group(3)
        ret["prob_r"] = image_err.group(4)
        ret["prob_e"] = image_err.group(5)


    return ret if len(ret) > 0 else None


# parse PyFaster
def parseErrPyFaster(errString,imgIndex):
    # ERR boxes: [27,4] e: 132.775177002 r: 132.775024414
    ret = {}
    if 'boxes' in errString:
        image_err = re.match(
            ".*boxes\: \[(\d+),(\d+)\].*e\: (\S+).*r\: (\S+).*",
            errString)
        if image_err:
            ret["type"] = "boxes"
            ret["imgindex"] = imgIndex
            ###########
            ret["boxes_x"] = image_err.group(1)
            try:
                long((ret["boxes_x"]))
            except:
                ret["boxes_x"] = 1e30
            ###########
            ret["boxes_y"] = image_err.group(2)
            try:
                long((ret["boxes_y"]))
            except:
                ret["boxes_y"] = 1e30
            ###########
            ret["e"] = image_err.group(3)
            try:
                long(float(ret["e"]))
            except:
                ret["e"] = 1e30
            ############
            ret["r"] = image_err.group(4)
            try:
                long(float(ret["r"]))
            except:
                ret["r"] = 1e30


        # try:
        #     long(float(ret["x_r"]))
        # except:
        #     ret["x_r"] = 1e30
    # else:
    #     image_err = re.match(".*probs\: \[(\d+),(\d+)\].*e\: ([0-9e\+\-\.]+).*r\: ([0-9e\+\-\.]+).*",
    #                          errString)
    #     if image_err:
    #         ret["type"] = "probs"
    #         ret["probs_x"] = image_err.group(1)
    #         ret["probs_y"] = image_err.group(2)
    #         ret["prob_e"] = image_err[3]
    #         ret["prob_r"] = image_err[4]

    return (ret if len(ret) > 0 else None)


# def readGold(gold_path):
#     gold_obj = GoldContent(gold_path)
#     return gold_obj



"""I used this function since deepcopy is so slow"""


def copyList(objList, nn='darknet'):
    # list = []
    # # print "Passou aqui"
    # done = True
    # for i in objList:
    #     if done:
    #         lenght = len(i)
    #         done = False
    #
    #     temp = np.empty(lenght, dtype=object)
    #     it = 0
    #     for j in i:
    #         temp[it] = j.deepcopy()
    #         it += 1
    #     list.append(temp)
    # return list
    temp = []
    if nn == 'darknet':
        for i in objList:
            temp.append(i.deepcopy())
    else:
        for i in objList:
            temp.append(copy.deepcopy(i))

    return temp


"""python precision recall was so slow that I needed use C++"""


def writeTempFileCXX(path, goldRect):
    f = open(path, "w")
    f.write("somenthing,something1\n")
    for i in goldRect:
        x = (i.left)
        y = (i.bottom)
        h = (i.height)
        w = (i.width)
        br_y = i.top
        br_x = i.right
        out = str(h) + "," + str(w) + "," + str(x) + "," + str(y) + "," + str(br_x) + "," + str(br_y) + "\n"
        # print out
        f.write(out)
    f.close()


"""

cPixel center_of_mass(vector<Rect> rectangles) {
cPixel pixel;
long x_total = 0, y_total = 0, pixels = 0;
long x_max, x_min, y_max, y_min;
for (long i = 0; i < rectangles.size(); i++) {
    x_max = rectangles[i].br_x;
    y_max = rectangles[i].br_y;
    x_min = x_max - rectangles[i].width;
    y_min = y_max - rectangles[i].height;

    for(long x = x_min; x <= x_max; x++) {
        for (long y = y_min; y <= y_max; y++) {
            x_total = x_total + x;
            y_total = y_total + y;
            pixels++;
        }
    }
}
pixel.x = x_total/pixels;
pixel.y = y_total/pixels;

return pixel;

}

return x and y pixel cordinates
"""


def centerOfMass(rectangles):
    xTotal = 0
    yTotal = 0
    pixels = 0
    for r in rectangles:
        # for x in xrange(r.left, r.right + 1):
        #  for y in xrange(r.bottom, r.top + 1):
        xTotal += ((r.bottom - r.top - 1) * (r.left - r.right - 1) * (r.left + r.right)) / 2
        yTotal += ((r.bottom - r.top - 1) * (r.bottom + r.top) * (r.left - r.right - 1)) / 2
        pixels += (r.top - r.bottom + 1) * (r.right - r.left + 1)

    return ((xTotal) / pixels, (yTotal) / pixels)

def centerOfMassGoldVsFound(gold,found, xSize, ySize):
    xGold,yGold = centerOfMass(gold)
    xFound,yFound = centerOfMass(found)

    #(test_center_of_mass.x - gold_center_of_mass.x)/x_size << "," << (double)(test_center_of_mass.y - gold_center_of_mass.y)/y_size
    return float(xFound - xGold)/xSize, float(yFound - yGold)/ySize

def getImageSize(imgPath):
    # print imgPath
    with Image.open(imgPath) as im:
        width, height = im.size
    return width, height


def generatePyFasterRectangles(dets, thresh=0):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]

    if len(inds) == 0:
        return

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    bboxList = []
    scoresList = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        scoresList.append(score)
        # left = int(math.floor(float(i["x_r"])))
        # bottom = int(math.floor(float(i["y_r"])))
        # h = int(math.ceil(float(i["h_r"])))
        # w = int(math.ceil(float(i["w_r"])))
        # tempBoxes[boxPos] = Rectangle(left, bottom, w, h)
        left = int(math.floor(float(bbox[0])))
        bottom = int(math.floor(float(bbox[1])))
        w = int(math.ceil(bbox[2] - bbox[0]))
        h = int(math.ceil(bbox[3] - bbox[1]))
        bboxList.append(Rectangle(left, bottom, w, h))

    #     ax.add_patch(
    #                           X       Y
    #         plt.Rectangle((bbox[0], bbox[1]),
    #                         Xmax   -   Xmin
    #                       bbox[2] - bbox[0],
    #                         Ymax   -  Ymin
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

    return scoresList,bboxList


"""
ret["type"] = "boxes"
ret["imgindex"] = imgIndex
###########
ret["boxes_x"] = image_err.group(1)

###########
ret["boxes_y"] = image_err.group(2)
###########
ret["e"] = image_err.group(3)
############
ret["r"] = image_err.group(4)
"""
def relativeErrorParserPyFaster(img_list_path, errList, gold_obj, sdcIte):
    if len(errList) <= 0:
        return ("errlist fucked",None,None,None,None,None,None,None,None,None)

    goldPyfaster = gold_obj.pyFasterGold
    img_list = open(img_list_path, "r").readlines()
    imgLPos =  getImgLPos(sdcit=sdcIte, maxsize=len(goldPyfaster.keys())) #getImgLPos(errList=errList, cnn="pyfaster", sdcit=sdcIte, maxsize=len(img_list))
    imgFile = img_list[imgLPos].rstrip()
    gold = goldPyfaster[imgFile]

    goldArray = getCls(gold)

    tempArray = copyList(goldArray, 'pyfaster')


    print "Gold array size ", len(tempArray)
    # for i in tempArray:
    #     print len(i)

    for i in errList:
        x = long(i["boxes_x"])
        y = long(i["boxes_y"])
        print i["boxes_x"] , i["boxes_y"]
        # print "vet size ", len(goldArray)
        # print "x size ", len (tempArray[x])
       # tempArray[x][y] = float(i["r"])

    goldRectangles = generatePyFasterRectangles(goldArray)
    tempRectangles      = generatePyFasterRectangles(tempArray)


    pR = PrecisionAndRecall(0.5)
    pR.precisionAndRecallParallel(goldRectangles, tempRectangles)

    return (
    len(gold), len(tempRectangles), 0, 0, pR.getPrecision(), pR.getRecall(), pR.getFalseNegative(), pR.getFalsePositive(),
    pR.getTruePositive(), imgFile)


"""
Compare two sets of boxes



       ret["type"] = "boxes"
        ret["image_list_position"] = image_err.group(1)
        ret["boxes"] = image_err.group(2)
        # x
        ret["x_r"] = image_err.group(3)
        ret["x_e"] = image_err.group(4)
        ret["x_diff"] = image_err.group(5)
        # y
        ret["y_r"] = image_err.group(6)
        ret["y_e"] = image_err.group(7)
        ret["y_diff"] = image_err.group(8)
        # w
        ret["w_r"] = image_err.group(9)
        ret["w_e"] = image_err.group(10)
        ret["w_diff"] = image_err.group(11)
        # h
        ret["h_r"] = image_err.group(12)
        ret["h_e"] = image_err.group(13)
        ret["h_diff"] = image_err.group(14)

"""

def relativeErrorParserDarknet(img_list_path, errList, gold_obj, sdcIt):
    if len(errList) <= 0:
        return ("errlist fucked",None,None,None,None,None,None,None,None,None)

    imgList = open(img_list_path, "r").readlines()

    imgLPos = getImgLPos(sdcit=sdcIt, maxsize=gold_obj.plist_size)

    #print "\nTamanho do plist " , gold_obj.plist_size , " tamanho do imgLPos" , imgLPos
    gold = gold_obj.prob_array["boxes"][imgLPos]
    tempBoxes = copyList(gold)
    # {'x_e': '2.3084202575683594e+02', 'h_r': '4.6537536621093750e+01', 'x_diff': '0.0000000000000000e+00',
    #  'w_diff': '3.8146972656250000e-06', 'y_r': '2.5291372680664062e+02', 'y_diff': '0.0000000000000000e+00',
    #  'w_e': '3.0895418167114258e+01', 'boxes': '94', 'h_e': '4.6537536621093750e+01', 'x_r': '2.3084202575683594e+02',
    #  'h_diff': '0.0000000000000000e+00', 'y_e': '2.5291372680664062e+02', 'w_r': '3.0895414352416992e+01',
    #  'type': 'boxes', 'image_list_position': '314'}
    for i in errList:
        if i["type"] == "boxes":
            boxPos = int(i['boxes'])
            left = int(math.floor(float(i["x_r"])))
            bottom = int(math.floor(float(i["y_r"])))
            h = int(math.ceil(float(i["h_r"])))
            w = int(math.ceil(float(i["w_r"])))
            tempBoxes[boxPos] = Rectangle(left, bottom, w, h)

    # writeTempFileCXX("/tmp/temp_test.log", tempBoxes)
    pR = PrecisionAndRecall(0.5)
    imgFile = imgList[imgLPos]
    #pR.precisionAndRecallSerial(gold, tempBoxes)
    p,r = pR.precisionRecall(gold, tempBoxes, 0.5)
    #sizX, sizY = getImageSize((GOLD_DIR + imgFile).rstrip("\n"))
    # start = time.clock()
    x,y = 0,0#centerOfMassGoldVsFound(gold,tempBoxes,sizX, sizY)
    # print time.clock() - start
    return (len(gold), len(tempBoxes), x, y, p, r, pR.getFalseNegative(), pR.getFalsePositive(), pR.getTruePositive(), imgFile)

def getImgLPos(**kwargs):
    sdcIte = kwargs.pop("sdcit")
    maxSize = kwargs.pop("maxsize")
    return (int(sdcIte) % int(maxSize))

# def relativeErrorPyFaster(img_list_path, errList, gold_obj, sdcIte):
#     img_list = open(img_list_path, "r").readlines()
#     imgLPos = getImgLPos(errList=errList, cnn="pyfaster", sdcit=sdcIte, maxsize=len(img_list))
#     imgFile = img_list[imgLPos]
#     gold = gold_obj[imgFile]
#     print getClassBoxes(gold)
# found = []
# gold = []
# with open("../../sassifi_hog/inst_hardened/logs/sdcs/1x_cuda02016_10_12_06_51_28_cudaHOG_carolk402.log", 'r') as sdc_file:
#     sdc_contents = sdc_file.readlines()
#     skip = 0
#     for i in sdc_contents:
#         if skip != 0:
#             temp_list = i.split(",")
#             #x, y, height, width, br_x, br_y)
#             h = long(temp_list[0])
#             w = long(temp_list[1])
#             x = long(temp_list[2])
#             y = long(temp_list[3])
#             i_found = Rectangle(x, y, w, h)
#             found.append(i_found)
#         else:
#             print i
#         skip += 1
#
# with open("../../sassifi_hog/GOLD_1x.data", 'r') as gold_file:
#     gold_contents = gold_file.readlines()
#     skip = 0
#     for i in gold_contents:
#
#         if skip != 0:
#             temp_list = i.split(",")
#             #x, y, height, width, br_x, br_y)
#             h = long(temp_list[0])
#             w = long(temp_list[1])
#             x = long(temp_list[2])
#             y = long(temp_list[3])
#             i_gold = Rectangle(x, y, w, h)
#             gold.append(i_gold)
#         else:
#             print i
#         skip += 1
#
#
# print "fazendo PR"
# pr = PrecisionAndRecall(0.5)
# pr.precisionAndRecallParallel(gold, found)
# print pr.getPrecision()
# print "centro de massa"
# x_size = 1500
# y_size = 1000
# print centerOfMassGoldVsFound(gold,found,x_size,y_size)

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
