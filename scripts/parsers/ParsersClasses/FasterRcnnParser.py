import math
import os
import re
from SupportClasses import Rectangle
import numpy as np
from SupportClasses import _GoldContent as gc
from SupportClasses import PrecisionAndRecall as pr
from ObjectDetectionParser import  ObjectDetectionParser

# GOLD_DIR = "/home/fernando/Dropbox/UFRGS/Pesquisa/LANSCE_2016_PARSED/Gold_CNNs/"

class FasterRcnnParser(ObjectDetectionParser):
    __iterations = None
    __imgListPath = None
    __board = None

    __rectangles = Rectangle.Rectangle(0, 0, 0, 0)

    def getBenchmark(self):
        return self._benchmark

    # parse PyFaster
    def parseErrMethod(self,errString):
        ret = {}
        if 'box' in errString:
            dictBox = self._processBoxes(errString)
            if len(dictBox) > 0:
                ret["boxes"] = dictBox
                ret["type"] = "boxes"
        elif 'score' in errString:
            dictScore = self._processScores(errString)
            if len(dictScore) > 0:
                ret["scores"] = dictScore
                ret["type"] = "scores"

        return (ret if len(ret) > 0 else None)


    def _processScores(self, errString):
        ret = {}


        return ret

    def _processBoxes(self, errString):
        ##ERR img_name: /home/carol/radiation-benchmarks/data/CALTECH/set10/V000/193.jpg
        # class: sheep
        # box: [8]
        #  x1_e: 435.740264893 x1_r: 435.782531738
        # y1_e: 244.744735718 y1_r: 244.746307373
        # x2_e: 610.136474609 x2_r: 610.124450684
        # y2_e: 326.088867188 y2_r: 326.093597412
        ret = {}
        if 'wrong' in errString:

            image_err = re.match(".*img_name\: (\S+).*"
                                 "class\: (\S+).*box\: \[(\d+)\].*"
                                 "x1_r\: (\S+).*"
                                 "y1_r\: (\S+).*"
                                 "x2_r\: (\S+).*"
                                 "y2_r\: (\S+).*", errString)
            if image_err:
                ret["image_path"] = image_err.group(1)
                ret["class"] = image_err.group(2)
                ret["box"] = image_err.group(3)

                #x1
                ret["x1"] = image_err.group(4)
                try:
                    long(float(ret["x1"]))
                except:
                    ret["x1"] = 1e30
                ###########

                #y1
                ret["y1"] = image_err.group(5)
                try:
                    long(float(ret["y1"]))
                except:
                    ret["y1"] = 1e30
                ###########

                #x2
                ret["x2"] = image_err.group(6)
                try:
                    long(float(ret["x2"]))
                except:
                    ret["x2"] = 1e30
                ############

                #y2
                ret["y2"] = image_err.group(7)
                try:
                    long(float(ret["y2"]))
                except:
                    ret["y2"] = 1e30

            return ret

    def _relativeErrorParser(self, errList):
        if len(errList) <= 0:
            return
        goldRects = []
        foundRects = []

        precisionRecallObj = pr.PrecisionAndRecall(self.__prThreshold)
        precisionRecallObj.precisionAndRecallParallel(goldRects, foundRects)

        self._goldLines = len(goldRects)
        self._detectedLines = len(foundRects)
        self._xCenterOfMass = None
        self._yCenterOfMass = None
        self._precision = precisionRecallObj.getPrecision()
        self._recall = precisionRecallObj.getRecall()
        self._falseNegative = precisionRecallObj.getFalseNegative()
        self._falsePositive = precisionRecallObj.getFalsePositive()
        self._truePositive = precisionRecallObj.getTruePositive()



    def setSize(self, header):
        # pyfaster
        py_faster_m = re.match(".*iterations\: (\d+).*img_list\: (\S+).*board\: (\S+).*", header)
        if py_faster_m:
            self.__iterations = py_faster_m.group(1)
            self.__imgListPath = py_faster_m.group(2)
            self.__board = py_faster_m.group(3)
        self._size = self.__imgListPath



