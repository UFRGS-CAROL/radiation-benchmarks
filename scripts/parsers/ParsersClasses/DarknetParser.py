#!/usr/bin/env python
from SupportClasses import Rectangle
import copy
import math
import os
import re

import numpy as np

from SupportClasses import PrecisionAndRecall as pr
from SupportClasses import GoldContent as gc
from Parser import Parser



# class for reading Darknet gold file
# from IPython.utils.py3compat import which

# need read onc
# darknet_gold_content = {}
# pyfaster_gold_contend = {}

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


class DarknetParser(Parser):

    __prThreshold = 0.5
    __precisionAndRecall = pr.PrecisionAndRecall(__prThreshold)
    __goldObj = gc.GoldContent()

    __rectangles = Rectangle.Rectangle(0, 0, 0, 0)

    def getBenchmark(self):
        return self._benchmark
    #overiding csvheader
    _csvHeader = ["logFileName", "Machine", "Benchmark", "imgFile", "SDC_Iteration", "#Accumulated_Errors",
                             "#Iteration_Errors", "gold_lines", "detected_lines", "x_center_of_mass",
                             "y_center_of_mass", "precision", "recall", "false_negative", "false_positive",
                             "true_positive"]


    __executionType  = None
    __executionModel = None
    __imgListPath    = None
    __imgListFile    = None
    __weights        = None
    __configFile     = None
    __iterations     = None
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

    def _relativeErrorParser(self, errList):
        imgListPath = None
        sdcIt = None

        if len(errList) <= 0:
            return ("errlist fucked", None, None, None, None, None, None, None, None, None)

        imgList = open(self.__imgListPath, "r").readlines()

        imgLPos = self.getImgLPos(sdcit= self._sdcIteration, maxsize=self.__goldObj.plist_size)

        # print "\nTamanho do plist " , gold_obj.plist_size , " tamanho do imgLPos" , imgLPos
        gold = self.__goldObj.prob_array["boxes"][imgLPos]
        tempBoxes = self.copyList(gold)
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
        pR = PrecisionAndRecall.PrecisionAndRecall(0.5)
        imgFile = imgList[imgLPos]
        # pR.precisionAndRecallSerial(gold, tempBoxes)
        p, r = pR.precisionRecall(gold, tempBoxes, 0.5)
        # sizX, sizY = getImageSize((GOLD_DIR + imgFile).rstrip("\n"))
        # start = time.clock()
        x, y = 0, 0  # centerOfMassGoldVsFound(gold,tempBoxes,sizX, sizY)
        # print time.clock() - start
        return (
            len(gold), len(tempBoxes), x, y, p, r, pR.getFalseNegative(), pR.getFalsePositive(), pR.getTruePositive(),
            imgFile)


    def buildImageMethod(self):
        return False

    def setSize(self, header):
        darknetM = re.match(
            ".*execution_type\:(\S+).*execution_model\:(\S+).*img_list_path\:(\S+).*weights\:(\S+).*config_file\:(\S+).*iterations\:(\d+).*",
            header)

        if darknetM:
            try:
                self.__executionType = darknetM.group(1)
                self.__executionModel = darknetM.group(2)
                self.__imgListPath = darknetM.group(3)
                self.__imgListFile = GOLD_DIR + os.path.basename(os.path.normpath(self.__imgListPath))
                self.__weights = darknetM.group(4)
                self.__configFile = darknetM.group(5)
                self.__iterations = darknetM.group(6)

            except:
                self.__imgListFile = None
        # return self.__imgListFile
        self._size = str(self.__imgListFile)


    # parse Darknet
    # returns a dictionary
    def parseErrMethod(self, errString):
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





    def copyList(self, gold):
        return None

    def getImgLPos(self, sdcIt, maxsize):
        return None

    def getClassBoxes(self, cls_boxes,cls_scores):
        NMS_THRESH = 0.3
        finalClassBoxes = []
        for cls_ind, cls in enumerate(CLASSES[1:]):
            # cls_ind += 1 # because we skipped background
            # for gold
            # cls_boxes_gold = boxes_gold[:, 4*cls_ind:4*(cls_ind + 1)]
            # cls_scores_gold = scores_gold[:, cls_ind]
            dets = np.hstack((cls_boxes[cls_ind],
                              cls_scores[cls_ind][:, np.newaxis])).astype(np.float32)
            keep = self.nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            finalClassBoxes.append(dets)

        return finalClassBoxes


    def getCls(self, gold):
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




    """I used this function since deepcopy is so slow"""


    def copyList(self,objList):
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
        # if nn == 'darknet':
        #     for i in objList:
        #         temp.append(i.deepcopy())
        # else:
        for i in objList:
            temp.append(copy.deepcopy(i))

        return temp



#
    # """python precision recall was so slow that I needed use C++"""
    # def writeTempFileCXX(path, goldRect):
    #     f = open(path, "w")
    #     f.write("somenthing,something1\n")
    #     for i in goldRect:
    #         x = (i.left)
    #         y = (i.bottom)
    #         h = (i.height)
    #         w = (i.width)
    #         br_y = i.top
    #         br_x = i.right
    #         out = str(h) + "," + str(w) + "," + str(x) + "," + str(y) + "," + str(br_x) + "," + str(br_y) + "\n"
    #         # print out
    #         f.write(out)
    #     f.close()
#
#
#
#
#
#
#
# def getImgLPos(**kwargs):
#     sdcIte = kwargs.pop("sdcit")
#     maxSize = kwargs.pop("maxsize")
#     return (int(sdcIte) % int(maxSize))

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

