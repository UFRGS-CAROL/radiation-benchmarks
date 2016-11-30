#!/usr/bin/env python

import re
from ctypes import *
import numpy as np
import copy
import multiprocessing
import subprocess
import math
import os

# class for reading Darknet gold file
# from IPython.utils.py3compat import which

# need read onc
# darknet_gold_content = {}
# pyfaster_gold_contend = {}
import time
import pickle
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

"""

"""

class PrecisionAndRecall(object):
    precision = 0
    recall = 0
    falsePositive = 0
    falseNegative = 0
    truePositive = 0
    threshold = 1

    def __init__(self, threshold):
        manager = multiprocessing.Manager()
        self.threshold = threshold
        self.precision = manager.Value('f', 0.0)
        self.recall = manager.Value('f', 0.0)
        self.falseNegative = manager.Value('i', 0)
        self.truePositive = manager.Value('i', 0)
        self.falsePositive =manager.Value('i',0)

    def __repr__(self):
        return "precision " + str(self.precision) + " recall:" + str(self.recall) + " false positive:" + str(self.falsePositive) \
                + " false negative:" + str(self.falseNegative) + " true positive:" + str(self.truePositive) + " threshold:" + str(self.threshold)

    def getPrecision(self): return self.precision.value
    def getFalsePositive(self): return self.falsePositive.value
    def getFalseNegative(self): return self.falseNegative.value
    def getTruePositive(self): return self.truePositive.value
    def getRecall(self): return self.precision.value

    """
    Calculates the precision an recall value, based on Lucas C++ function for HOG
    gold = List of Rectangles
    found = List o Rectangles
    """

    def precisionAndRecallSerial(self, gold, found):
        self.recallMethod(gold, found)

        self.precisionMethod(gold, found)
        self.precision.value = float(self.truePositive.value) / float(self.truePositive.value + self.falsePositive.value)

    def precisionAndRecallParallel(self, gold, found):
        # print "running in parallel"
        rp = multiprocessing.Process(target=self.recallMethod, args=(gold, found))
        pp = multiprocessing.Process(target=self.precisionMethod, args=(gold, found))
        rp.start()
        pp.start()
        rp.join(timeout=1)
        pp.join(timeout=1)
        # print "precision " + str(self.precision.value) + " recall:" + str(self.recall.value) + " false positive:" + str(self.falsePositive) \
        #         + " false negative:" + str(self.falseNegative) + " true positive:" + str(self.truePositive) + " threshold:" + str(self.threshold)
        self.precision.value = float(self.truePositive.value) / float(self.truePositive.value + self.falsePositive.value)

    """
        split precision for parallelism
    """

    def precisionMethod(self, gold, found):
        # print "passou precision"
        out_positive = 0
        for i in found:
            for g in gold:
                # calc the intersection
                intRect = g.intersection(i)
                intersectionArea = intRect.area()
                unionRect = g.union(i)
                unionArea = unionRect.area()

                if (float(intersectionArea) / float(unionArea)) >= self.threshold:
                    out_positive += 1
                    break

        self.falsePositive.value = len(found) - out_positive

    """
        split recall for parallelism
    """


    def recallMethod(self, gold, found):
        # print "passou recall"
        for i in gold:
            for z in found:
                intRect = z.intersection(i)
                unionRect = z.union(i)
                intersectionArea = intRect.area()
                unionArea = unionRect.area()
                if (float(intersectionArea) / float(unionArea)) >= self.threshold:
                    self.truePositive.value += 1
                    break

        self.falseNegative.value = len(gold) - self.truePositive.value
        self.recall.value = float(self.truePositive.value) / float(self.truePositive.value + self.falseNegative.value)


class Point(object):
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y




class GoldContent(object):
    plist_size = 0
    classe = 0
    total_size = 0
    prob_array = {}
    pyFasterGold = []
    pyFasterImgList = ""

    # return a dict that look like this
    # //to store all gold filenames
    # typedef struct gold_pointers {
    # //	box *boxes_gold;
    # //	ProbArray pb;
    # 	ProbArray *pb_gold;
    # 	long plist_size;
    # 	FILE* gold;
    # 	int has_file;
    # } GoldPointers;
    #
    def __init__(self, **kwargs):
        #use keyargs
        nn = kwargs.pop("nn")
        filePath = kwargs.pop("filepath")
        if "darknet" in nn:
            self.darknetConstructor(filePath)
        elif "pyfaster" in nn:
            self.pyFasterConstructor(filePath)

    def darknetConstructor(self, filePath):
        cc_file = open(filePath, 'rb')
        result = []
        # darknet write file in this order, so we need recover data in this order
        # long plist_size;
        # long classes;
        # long total_size;
        # for ( < plist_size times >){
        # -----pb_gold.boxes
        # -----pb_gold.probs
        # }
        plist_size = Long()
        classes = Long()
        total_size = Long()
        cc_file.readinto(plist_size)
        cc_file.readinto(classes)
        cc_file.readinto(total_size)
        plist_size = long(plist_size.l)
        classes = long(classes.l)
        total_size = long(total_size.l)
        i = 0
        self.plist_size = plist_size
        self.classes = classes
        self.total_size = total_size
        self.prob_array["boxes"] = []
        self.prob_array["probs"] = []
        while i < long(plist_size):
            # boxes has total_size size
            boxes = readBoxes(cc_file, total_size)
            probs = readProbs(cc_file, total_size, classes)
            self.prob_array["probs"].append(probs)
            self.prob_array["boxes"].append(boxes)
            i += 1
        cc_file.close()

    def pyFasterConstructor(self, filepath):
        try:
            f = open(filepath, "rb")
            tempGold = pickle.load(f)
            f.close()

        except:
            raise
        self.pyFasterGold = tempGold



    def __copy__(self):
        return copy.deepcopy(self)


def getClassBoxes(gold):
    scores = gold[0]
    boxes = gold[1]

    classBoxes = []

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = py_cpu_nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        classBoxes.append(dets)

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

    return classBoxes


class Float(Structure):
    _fields_ = [('f', c_float)]

    def __repr__(self):
        return str(self.f)


class Box(Structure):
    _fields_ = [('x', c_float), ('y', c_float), ('w', c_float), ('h', c_float)]

    def __repr__(self):
        return str(self.x) + " " + str(self.y) + " " + str(self.w) + " " + str(self.h)


class Long(Structure):
    _fields_ = [('l', c_long)]

    def __repr__(self):
        return str(self.l)


"""
  	left
The X coordinate of the left side of the box
  	right
The X coordinate of the right side of the box
  	top
The Y coordinate of the top edge of the box
  	bottom
The Y coordinate of the bottom edge of the box
"""


class Rectangle(object):
    left = 0
    bottom = 0
    top = 0
    right = 0

    height = 0
    width = 0

    # self._left, self._top, self._right, self._bottom
    def __init__(self, left, bottom, width, height):
        self.left = left
        self.bottom = bottom
        self.width = width
        self.height = height
        self.right()
        self.top()

    def right(self):
        """The width of the rectangle"""
        self.right = self.left + self.width

    def top(self):
        self.top = self.bottom + self.height

    def __repr__(self):
        return "left " + str(self.left) + " bottom " + str(self.bottom) + " width " + str(
            self.width) + " height " + str(self.height) + " right " + str(
            self.right) + " top " + str(self.top)

    def deepcopy(self):
        # print "passou no deepcopy"
        copy_obj = Rectangle(0, 0, 0, 0)
        copy_obj.left = self.left
        copy_obj.bottom = self.bottom
        copy_obj.height = self.height
        copy_obj.width = self.width
        copy_obj.right = self.right
        copy_obj.top = self.top
        return copy_obj

    def __deepcopy__(self):
        # print "passou no deepcopy"
        copy_obj = Rectangle(0, 0, 0, 0)
        copy_obj.left = self.left
        copy_obj.bottom = self.bottom
        copy_obj.height = self.height
        copy_obj.width = self.width
        copy_obj.right = self.right
        copy_obj.top = self.top
        return copy_obj

    def intersection(self, other):
        # if self.isdisjoint(other):
        #     return Rectangle(0, 0, 0, 0)
        left = max(self.left, other.left)
        bottom = max(self.bottom, other.bottom)
        right = min(self.right, other.right)
        top = min(self.top, other.top)
        w = right - left
        h = top - bottom
        if w > 0 and h > 0:
            return Rectangle(left, bottom, w, h)
        else:
            return Rectangle(0, 0, 0, 0)

    def union(self, other):
        left = min(self.left, other.left)
        bottom = min(self.bottom, other.bottom)
        top = max(self.top, other.top)
        right = max(self.right, other.right)
        w = right - left
        h = top - bottom
        return Rectangle(left, bottom, w, h)

    def isdisjoint(self, other):
        """Returns ``True`` if the two rectangles have no intersection."""
        return self.left > other.right or self.right < other.left or self.top > other.bottom or self.bottom < other.top

    def area(self):
        return self.width * self.height


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


def readBoxes(cc_file, n):
    i = 0
    boxes = np.empty(n, dtype=object)
    while i < n:
        box = Box()
        cc_file.readinto(box)
        # instead boxes I put the Rectangles to make it easy
        # (self, x, y, height, width, br_x, br_y):
        # x_min_gold = x_max_gold - gold[i].width;
        # y_min_gold = y_max_gold - gold[i].height;
        # left, top, right, bottom):
        left = int(math.floor(box.x))
        bottom = int(math.floor(box.y))
        h = int(math.ceil(box.h))
        w = int(math.ceil(box.w))

        boxes[i] = Rectangle(left, bottom, w, h)
        i += 1
    return boxes


def readProbs(cc_file, total_size, classes):
    i = 0
    prob = np.empty((total_size, classes), dtype=object)

    while i < total_size:
        j = 0
        while j < classes:
            pb_ij = Float()
            cc_file.readinto(pb_ij)
            prob[i][j] = pb_ij
            j += 1
        i += 1

    return prob



# def readGold(gold_path):
#     gold_obj = GoldContent(gold_path)
#     return gold_obj



"""I used this function since deepcopy is so slow"""


def copyList(objList):
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

    goldArray = getClassBoxes(gold)

    tempArray = copyList(goldArray)

    print "Gold array size ", len(tempArray)
    for i in tempArray:
        print len(i)

    for i in errList:
        x = long(i["boxes_x"])
        y = long(i["boxes_y"])
        print i["boxes_x"] , i["boxes_y"]

        #tempArray[x][y] = float(i["r"])
    exit(1)
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

    print "\nTamanho do plist " , gold_obj.plist_size , " tamanho do imgLPos" , imgLPos
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
    pR.precisionAndRecallParallel(gold, tempBoxes)
    #sizX, sizY = getImageSize((GOLD_DIR + imgFile).rstrip("\n"))
    # start = time.clock()
    x,y = 0,0#centerOfMassGoldVsFound(gold,tempBoxes,sizX, sizY)
    # print time.clock() - start
    return (len(gold), len(tempBoxes), x, y, pR.getPrecision(), pR.getRecall(), pR.getFalseNegative(), pR.getFalsePositive(), pR.getTruePositive(), imgFile)

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
