#!/usr/bin/env python
import math

import numpy

from SupportClasses import Rectangle
import os
import re
import sys
from PIL import Image


from ObjectDetectionParser import ObjectDetectionParser
from SupportClasses import _GoldContent
from SupportClasses import PrecisionAndRecall

"""This section MUST, I WRITE MUST, BE SET ACCORDING THE GOLD PATHS"""
# GOLD_BASE_DIR = [
#     '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_K40',
#     #'/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_TITAN',
#      # '/home/familia/Dropbox/UFRGS/Pesquisa/fault_injections/sassifi_darknet'
# ]

GOLD_BASE_DIR = {
    'carol-k402': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_K40',
    'carol-tx': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_TITAN',
    #carolx1a
    #carolx1b
    #carolx1c

     # '/home/familia/Dropbox/UFRGS/Pesquisa/fault_injections/sassifi_darknet'
}
LOCAL_RADIATION_BENCH =  '/home/fernando/git_pesquisa' #'/mnt/4E0AEF320AEF15AD/PESQUISA/git_pesquisa'

# DATASETS = {
#     # normal
#     'gold.caltech.critical.1K.test': {
#         'caltech.pedestrians.critical.1K.txt': {
#             'gold': None, 'txt': None, 'obj': None}},
#     'gold.caltech.1K.test': {
#         'caltech.pedestrians.1K.txt': {'gold': None, 'txt': None, 'obj': None}},
#     # abft
#     'gold.caltech.abft.1K.test': {
#         'caltech.pedestrians.1K.txt': {'gold': None, 'txt': None, 'obj': None}},
#     'gold.caltech.critical.abft.1K.test': {'caltech.pedestrians.critical.1K.txt': {'gold': None, 'txt': None, 'obj': None}},
#
#     'gold.voc.2012.1K.test': {
#         'voc.2012.1K.txt': {'gold': None, 'txt': None, 'obj': None}},
#     'gold.voc.2012.abft.1K.test': {
#         'voc.2012.1K.txt': {'gold': None, 'txt': None, 'obj': None}},
#      }

DATASETS = {
    # normal
    'caltech.pedestrians.critical.1K.txt':{'dumb_abft':'gold.caltech.critical.abft.1K.test', 'no_abft':'gold.caltech.critical.1K.test'},
    'caltech.pedestrians.1K.txt':{'dumb_abft':'gold.caltech.abft.1K.test','no_abft':'gold.caltech.1K.test'},
    'voc.2012.1K.txt':{'dumb_abft':'gold.voc.2012.abft.1K.test', 'no_abft': 'gold.voc.2012.1K.test'}
}



# CURRENT_MACHINE_DIR ="/home/familia/Fernando/radiation-benchmarks/data/networks_img_list/"

"""___________________"""

import time


class ImageRaw():
    w = 0
    h = 0
    file = ""
    def __init__(self, file):
        self.w, self.h = self.getImageSize(file)
        self.file = file


    def getImageSize(self, imgPath):
        # print imgPath
        with Image.open(imgPath) as im:
            width, height = im.size
        return width, height


class DarknetParser(ObjectDetectionParser):
    __executionType = None
    __executionModel = None
    __imgListPath = None
    __weights = None
    __configFile = None
    __iterations = None
    __goldFileName = None

    # def __init__(self):
    #     start_time = time.time()
    #     ObjectDetectionParser.__init__(self)
    #     for kD, vD in DATASETS.iteritems():
    #         for kI, vI in vD.iteritems():
    #             for i in GOLD_BASE_DIR:
    #                 DATASETS[kD][kI]['gold'] = str(i) + '/darknet/' + str(kD)
    #                 DATASETS[kD][kI]['txt'] = str(
    #                     i) + '/networks_img_list/' + str(kI)
    #
    #                 if not os.path.isfile(DATASETS[kD][kI]['gold']):
    #                     sys.exit(str(DATASETS[kD][kI][
    #                                      'gold']) + " no such file or directory")
    #                 if not os.path.isfile(DATASETS[kD][kI]['txt']):
    #                     sys.exit(str(DATASETS[kD][kI][
    #                                      'txt']) + " no such file or directory")
    #
    #                 # if it pass, I will open all gold on memory
    #                 DATASETS[kD][kI]['obj'] = _GoldContent._GoldContent(
    #                     nn='darknet', filepath=DATASETS[kD][kI]['gold'])
    #     elapsed_time = time.time() - start_time
    #
    #     print "\n darknet open gold time ", elapsed_time
    #     # for gold object
    #
    # # each imglist has a gold, only need to keep then on the memory
    # goldObjects = {}

    def setSize(self, header):
        if "abft" in header:
            darknetM = re.match(
                ".*execution_type\:(\S+).*execution_model\:(\S+).*img_list_path\:"
                "(\S+).*weights\:(\S+).*config_file\:(\S+).*iterations\:(\d+).*abft: (\S+).*",
                header)
        else:
            darknetM = re.match(
                ".*execution_type\:(\S+).*execution_model\:(\S+).*img_list_path\:"
                "(\S+).*weights\:(\S+).*config_file\:(\S+).*iterations\:(\d+).*",
                header)

        if darknetM:
            try:
                self.__executionType = darknetM.group(1)
                self.__executionModel = darknetM.group(2)
                self.__imgListPath = darknetM.group(3)
                self.__weights = darknetM.group(4)
                self.__configFile = darknetM.group(5)
                self.__iterations = darknetM.group(6)
                if "abft" in header:
                    self._abftType = darknetM.group(7)

                self.__goldFileName = self.getGoldFileName(self.__imgListPath)
            except:
                self.__imgListPath = None

        # return self.__imgListFile
        # tempPath = os.path.basename(self.__imgListPath).replace(".txt","")
        self._size = str(self.__goldFileName)

    def __printYoloDetections(self, boxes, probs, total, classes):
        validRectangles = []
        validProbs = []
        validClasses = []
        for i in range(0, total):
            box = boxes[i]
            xmin = max(box.left - box.width / 2.0, 0)
            ymin = max(box.bottom - box.height / 2.0, 0)

            # if xmin < 0:
            #     xmin = 0
            # if ymin < 0:
            #     ymin = 0

            for j in range(0, classes):
                if probs[i][j] >= self._detectionThreshold:
                    validProbs.append(probs[i][j])
                    #check image bounds
                    rect = Rectangle.Rectangle(int(xmin), int(ymin), box.width, box.height)
                    validRectangles.append(rect)
                    validClasses.append(self._classes[j])
                    # print self._classes[j]

        return validRectangles, validProbs, validClasses

    def __setLocalFile(self, listFile, imgPos):
        tmp = (listFile[imgPos].rstrip()).split('radiation-benchmarks')[1]
        tmp = LOCAL_RADIATION_BENCH + '/radiation-benchmarks' + tmp
        return tmp

    def newRectArray(self, arr):
        ret = numpy.empty(len(arr), dtype=object)
        t = 0
        for i in arr:
            ret[t] = i.deepcopy()
            t += 1
        return ret


    def newMatrix(self, prob, n, m):
        ret = numpy.empty((n, m), dtype=float)
        for i in xrange(0,n):
            for j in xrange(0, m):
                ret[i][j] = prob[i][j]
        return ret

    def _relativeErrorParser(self, errList):
        if len(errList) <= 0:
            return

        # # parsing box array
        # print self.__goldFileName
        # print self.__imgListPath
        # currDataset = DATASETS[self.__goldFileName][os.path.basename(self.__imgListPath)]
        # goldCurrObj = currDataset['obj']
        # if goldCurrObj == None:
        #     sys.exit("Gold obj was not created")
        #
        # listFile = open(currDataset['txt']).readlines()
        # print currDataset['gold']
        #
        # imgPos = int(self._sdcIteration) % len(listFile)
        # print imgPos
        # imgFilename = self.__setLocalFile(listFile, imgPos)
        # imgObj = ImageRaw(imgFilename)
        #
        #
        #
        # # probs
        # goldProb = goldCurrObj.getProbArray()
        # goldProb = goldProb[imgPos]
        #
        # foundProb = self.newMatrix(goldProb, goldCurrObj.getTotalSize(), goldCurrObj.getClasses())#numpy.copy(goldProb)
        # # rects
        # goldRects = goldCurrObj.getRectArray()
        # goldRects = goldRects[imgPos]
        #
        #
        # foundRects = self.newRectArray(goldRects)#self.copyList(goldRects)
        #the gold key is build with machine + benchmark + dataset
        gold = None
        goldKey = self._machine + "_" + self._benchmark + "_" + self.__goldFileName
        goldPath = GOLD_BASE_DIR[self._machine] + "/darknet/" + self.__goldFileName
        txtPath = GOLD_BASE_DIR[self._machine] + '/networks_img_list/' + os.path.basename(self.__imgListPath)
        if goldKey not in self._goldDatasetArray:
            g = _GoldContent._GoldContent(nn='darknet', filepath=goldPath)
            self._goldDatasetArray[goldKey] = g

        gold = self._goldDatasetArray[goldKey]

        listFile = open(txtPath).readlines()

        imgPos = int(self._sdcIteration) % len(listFile)
        imgFilename = self.__setLocalFile(listFile, imgPos)
        imgObj = ImageRaw(imgFilename)
        goldPb = gold.getProbArray()[imgPos]
        goldRt = gold.getRectArray()[imgPos]

        foundPb = self.newMatrix(goldPb, gold.getTotalSize(), gold.getClasses())
        foundRt = self.newRectArray(goldRt)

        self._rowDetErrors = 0
        self._colDetErrors = 0
        self._wrongElements = 0

        # for y in errList:
        #     # print y['img_list_position'], imgPos
        #     # print y
        #     if y["type"] == "boxes":
        #         i = y['boxes']
        #         rectPos = i['box_pos']
        #         left = int(float(i["x_r"]))
        #         bottom = int((float(i["y_r"])))
        #         # done = False
        #         # if 1 < math.fabs(float(i['x_r']) - float(i['x_e'])):
        #         #     print "x" , float(i['x_r']) - float(i['x_e'])
        #         #     done = True
        #         # if math.fabs(float(i['y_r']) - float(i['y_e'])) > 1:
        #         #     print "y", float(i['y_r']) - float(i['y_e'])
        #         #     done = True
        #         # if math.fabs(float(i['h_r']) - float(i['h_e'])) > 1:
        #         #     print "h" , float(i['h_r']) - float(i['h_e'])
        #         #     done = True
        #         # if math.fa bs( float(i['w_r']) - float(i['w_e'])) > 1:
        #         #     print "w" , float(i['w_r']) - float(i['w_e'])
        #         #     done = True
        #         h = abs(int(float(i["h_r"])))
        #         w = abs(int(float(i["w_r"])))
        #         print "before", foundRects[rectPos]
        #         print i
        #         foundRects[rectPos] = Rectangle.Rectangle(left, bottom, w, h)
        #         # print "rect found " , foundRects[rectPos]
        #         # print "rect gold "  , goldRects[rectPos] , "\n"
        #         print float(i["x_r"]) , float(i["y_r"]), float(i["h_r"]), float(i["w_r"])
        #         print "after", foundRects[rectPos] , "\n"
        #         self._wrongElements += 1
        #     elif y["type"] == "abft" and self._abftType != 'no_abft':
        #
        #         i = y['abft_det']
        #         self._rowDetErrors += i["row_detected_errors"]
        #         self._colDetErrors += i["col_detected_errors"]
        #
        #     elif y["type"] == "probs":
        #         i = int(y["probs"]["probs_x"])
        #         j = int(y["probs"]["probs_y"])
        #
        #         foundProb[i][j] = float(y["probs"]["prob_r"])



        if self._rowDetErrors > 1e6:
            self._rowDetErrors /= long(1e15)
        if self._colDetErrors > 1e6:
            self._colDetErrors /= long(1e15)

        #############
        # before keep going is necessary to filter the results
        gValidRects, gValidProbs, gValidClasses = self.__printYoloDetections(goldRt, goldPb, gold.getTotalSize(),
                                                                             len(self._classes) -1)
        fValidRects, fValidProbs, fValidClasses = self.__printYoloDetections(foundRt, foundPb, gold.getTotalSize(),
                                                                             len(self._classes) - 1)



        if self._logFileName == '2016_12_11_14_50_35_cudaDarknet_carol-k402.log':
            print goldRt[31]
            sys.exit()
        #############
        # imgFilename = listFile[imgPos].rstrip()
        # if len(gValidRects) > 1:
        #     imgFilename = imgFilename.split("data", 1)[1]
        #     imgFilename = "/home/carol/radiation-benchmarks/data" + imgFilename
        #     if imgFilename not in self.test:
        #         self.test.append(imgFilename)
        #         os.system("echo \""  + str(self.test) + '\" > last_line.txt')
        #     print gValidRects

        # self.buildImageMethod(listFile[imgPos].rstrip(), fValidRects,
        #                           fValidClasses, fValidProbs, "found")
        # sys.exit()

        precisionRecallObj = PrecisionAndRecall.PrecisionAndRecall(0.5)
        gValidSize = len(gValidRects)
        fValidSize = len(fValidRects)

        precisionRecallObj.precisionAndRecallParallel(gValidRects, fValidRects)
        self._precision = precisionRecallObj.getPrecision()
        self._recall = precisionRecallObj.getRecall()
        print "\n", self._precision , self._recall
        # self.buildImageMethod(listFile[imgPos].rstrip(), gValidRects, fValidRects)

        self._falseNegative = precisionRecallObj.getFalseNegative()
        self._falsePositive = precisionRecallObj.getFalsePositive()
        self._truePositive = precisionRecallObj.getTruePositive()
        # set all
        self._goldLines = gValidSize
        self._detectedLines = fValidSize
        self._xCenterOfMass, self._yCenterOfMass = precisionRecallObj.centerOfMassGoldVsFound(gValidRects, fValidRects, imgObj.w, imgObj.h)


    # parse Darknet
    # returns a dictionary
    def parseErrMethod(self, errString):
        # parse errString for darknet
        ret = {}
        imgListPosition = ""
        if 'boxes' in errString:
            dictBox, imgListPosition = self.__processBoxes(errString)
            if len(dictBox) > 0:
                ret["boxes"] = dictBox
                ret["type"] = "boxes"
        elif 'probs' in errString:
            dictProbs, imgListPosition = self.__processProbs(errString)
            if len(dictProbs) > 0:
                ret["probs"] = dictProbs
                ret["type"] = "probs"
        elif 'INF' in errString:
            dictAbft, imgListPosition = self.__processAbft(errString)
            if len(dictAbft) > 0:
                ret["abft_det"] = dictAbft
                ret["type"] = "abft"

        if imgListPosition != "":
            ret["img_list_position"] = int(imgListPosition)

        return ret if len(ret) > 0 else None

    def __processBoxes(self, errString):
        ret = {}
        imgListPosition = ""
        # ERR image_list_position: [823] boxes: [0]  x_r:
        # 6.7088904380798340e+00 x_e: 6.7152066230773926e+00 x_diff:
        # 6.3161849975585938e-03 y_r: 4.9068140983581543e+00 y_e:
        # 4.9818339347839355e+00 y_diff: 7.5019836425781250e-02 w_r:
        # 2.1113674640655518e+00 w_e: 2.1666510105133057e+00 w_diff:
        # 5.5283546447753906e-02 h_r: 3.4393796920776367e+00 h_e:
        # 3.4377186298370361e+00 h_diff: 1.6610622406005859e-03
        image_err = re.match(
            ".*image_list_position\: \[(\d+)\].*boxes\: \[(\d+)\].*x_r\: (\S+).*x_e\: (\S+).*x_diff\:"
            " (\S+).*y_r\: (\S+).*y_e\: (\S+).*y_diff\: (\S+).*w_r\: (\S+).*w_e\: (\S+).*w_diff\:"
            " (\S+).*h_r\: (\S+).*h_e\: (\S+).*h_diff\: (\S+).*", errString)

        if image_err:
            try:
                imgListPosition = image_err.group(1)
                ret["box_pos"] = int(image_err.group(2))
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
                    long(float(ret["w_diff"]))

                except:
                    ret["w_diff"] = 1e30

                # h
                ret["h_r"] = image_err.group(12)
                ret["h_e"] = image_err.group(13)
                ret["h_diff"] = image_err.group(14)
                try:
                    long(float(ret["h_r"]))
                except:
                    ret["h_r"] = 1e30

                try:
                    long(float(ret["h_e"]))
                except:
                    ret["h_e"] = 1e30

                try:
                    long(float(ret["h_diff"]))
                except:
                    ret["h_diff"] = 1e30
            except:
                print "Error on parsing boxes"
                raise

            # if float(ret['x_r']) - float(ret['x_e']) > 0.01:
            #     print float(ret['x_r']) - float(ret['x_e'])
            # if float(ret['y_r']) - float(ret['y_e']) > 0.01:
            #     print float(ret['y_r']) - float(ret['y_e'])
            #
            # if float(ret["h_r"]) - float(ret['h_e']) > 0.01:
            #     print float(ret["h_r"]) - float(ret['h_e'])
            # if float(ret["w_r"]) - float(ret["w_e"]) > 0.01:
            #     print float(ret["w_r"]) - float(ret["w_e"])

        return ret, imgListPosition

    def __processProbs(self, errString):
        ret = {}
        imgListPosition = ""
        image_err = re.match(
            ".*image_list_position\: \[(\d+)\].*probs\: \[(\d+),"
            "(\d+)\].*prob_r\: ([0-9e\+\-\.]+).*prob_e\: ([0-9e\+\-\.]+).*",
            errString)
        if image_err:
            try:
                imgListPosition = image_err.group(1)
                ret["probs_x"] = image_err.group(2)
                ret["probs_y"] = image_err.group(3)
                ret["prob_r"] = image_err.group(4)
                ret["prob_e"] = image_err.group(5)
                # if math.fabs(float(ret["prob_r"]) - float(ret["prob_e"])) > 0.1:
                #     print float(ret["prob_r"]) - float(ret["prob_e"])
            except:
                print "Error on parsing probs"
                raise

        return ret, imgListPosition

    def __processAbft(self, errString):
        # INF abft_type: dumb image_list_position: [151] row_detected_errors: 1 col_detected_errors: 1
        m = re.match(
            ".*abft_type\: (\S+).*image_list_position\: \[(\d+)\].*row_detected_errors\:"
            " (\d+).*col_detected_errors\: (\d+).*", errString)
        ret = {}
        imgListPosition = ""
        if m:
            try:
                imgListPosition = str(m.group(2))
                ret["row_detected_errors"] = int(m.group(3))
                ret["col_detected_errors"] = int(m.group(4))
            except:
                print "Error on parsing abft info"
                raise

        return ret, imgListPosition

    def getGoldFileName(self, imgListPath):
        imgListPath = os.path.basename(imgListPath)
        # for k, v in DATASETS.iteritems():
        #     for kI, vI in v.iteritems():
        #         if imgListPath == kI:
        #             k = os.path.basename(k)
        #             if self._abftType == 'dumb_abft' and 'abft' in k:
        #                 return k
        #             elif self._abftType == 'no_abft' and 'abft' not in k:
        #                 return k
        return DATASETS[imgListPath][self._abftType]

    # def __maxIndex(self, a):
    #     n = len(a)
    #     if (n <= 0):
    #         return -1
    #     maxI = 0
    #     max = a[0]
    #     for i in range(1, n):
    #         if (a[i] > max):
    #             max = a[i]
    #             maxI = i
    #     return maxI


    # def __drawDetections(self, image, num, boxes,probs):
    #     validRectangles = []
    #     validProbs = []
    #     validClasses = []
    #     for i in range(0, num):
    #         class_ = self.__maxIndex(probs[i])
    #         prob = probs[i][class_]
    #         if(prob > self._detectionThreshold):
    #             # width = 8
    #             #print names[class_], prob*100
    #             # offset = class_*1 % classesN
    #             # red = self.__getColor(2,offset,classesN)
    #             # green = self.__getColor(1,offset,classesN)
    #             # blue = self.__getColor(0,offset,classesN)
    #             # rgb = [red, green, blue]
    #             b = boxes[i]
    #             left = (b.left - b.width / 2.)  * image.w
    #             right = (b.left + b.width / 2.) * image.w
    #             top = (b.bottom - b.height / 2.) * image.h
    #             bot = (b.bottom + b.height / 2.) * image.h
    #
    #             if (left < 0): left = 0
    #             if (right > image.w - 1): right = image.w-1
    #             if (top < 0): top = 0
    #             if (bot > image.h - 1): bot = image.h-1;
    #
    #             # draw_box_width(im, left, top, right, bot, width, red, green, blue)
    #             # if (labels) draw_label(im, top + width, left, labels[class], rgb)
    #             validProbs.append(prob)
    #             rect = Rectangle.Rectangle(int(left),
    #                                        int(bot), (int(right - left)),
    #                                        (int(top - bot)))
    #             validRectangles.append(rect)
    #             validClasses.append(self._classes[class_])
    #
    #
    #     return validRectangles, validProbs, validClasses

