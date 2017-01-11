import os
import re

import time

import sys

from ObjectDetectionParser import  ObjectDetectionParser
from SupportClasses import PrecisionAndRecall as pr
from SupportClasses import _GoldContent

GOLD_BASE_DIR = [
    #'/home/familia/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_K40',
    #'/home/familia/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_TITAN',
     '/home/familia/Dropbox/UFRGS/Pesquisa/fault_injections/sassifi_darknet'
]

DATASETS = {
    'gold.caltech.critical.1K.test': {
        'caltech.pedestrians.critical.1K.txt': {
            'gold': None, 'txt': None, 'obj': None}},
    'gold.caltech.1K.test': {
        'caltech.pedestrians.1K.txt': {'gold': None, 'txt': None, 'obj': None}},
    'gold.voc.2012.1K.test': {
       'voc.2012.1K.txt': {'gold': None, 'txt': None, 'obj': None}},
}

class FasterRcnnParser(ObjectDetectionParser):
    __iterations = None
    __imgListPath = None
    __board = None

    def __init__(self):
        start_time = time.time()
        ObjectDetectionParser.__init__(self)
        for kD, vD in DATASETS.iteritems():
            for kI, vI in vD.iteritems():
                for i in GOLD_BASE_DIR:
                    DATASETS[kD][kI]['gold'] = str(i) + '/py_faster_rcnn/' + str(kD)
                    DATASETS[kD][kI]['txt'] = str(
                        i) + '/networks_img_list/' + str(kI)

                    if not os.path.isfile(DATASETS[kD][kI]['gold']):
                        sys.exit(str(DATASETS[kD][kI][
                                         'gold']) + " no such file or directory")
                    if not os.path.isfile(DATASETS[kD][kI]['txt']):
                        sys.exit(str(DATASETS[kD][kI][
                                         'txt']) + " no such file or directory")

                    # if it pass, I will open all gold on memory
                    DATASETS[kD][kI]['obj'] = _GoldContent._GoldContent(
                        nn='darknet', filepath=DATASETS[kD][kI]['gold'])
        elapsed_time = time.time() - start_time

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
        #ERR img_name: /home/carol/radiation-benchmarks/data/VOC2012/2011_004360.jpg class: horse wrong_score_size: -17
        #ERR img_name: /home/carol/radiation-benchmarks/data/VOC2012/2011_004360.jpg class: horse score: [0] e: 0.0158654786646 r: 0.00468954769894
        scoreErr = re.match(".*img_name\: (\S+).*"
                                 "class\: (\S+).*wrong_score_size\: (\S+).*", errString)

        ret["wrong_score_size"] = -1
        if scoreErr:
            try:
                ret["wrong_score_size"] = abs(int(scoreErr.group(3)))
            except:
                print "\nerror on parsing wrong_score_size"
                raise

        else:
            scoreErr =  re.match(".*img_name\: (\S+).*"
                                 "class\: (\S+).*score\: \[(\d+)\].*e\: (\S+).*r\: (\S+).*", errString)

            try:
                ret["score_pos"] = int(scoreErr.group(3))
            except:
                print "\nerror on parsing score pos"
                raise

            try:
                ret["score_r"] = float(scoreErr.group(5))
            except:
                print "\nerror on parsing score read"
                raise

        if scoreErr:
            try:
                ret["img_path"] = scoreErr.group(1)
                ret["class"] = scoreErr.group(2)
            except:
                print "\nerror on parsing img_path and class"
                raise

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

            imageErr = re.match(".*img_name\: (\S+).*"
                                 "class\: (\S+).*box\: \[(\d+)\].*"
                                 "x1_r\: (\S+).*"
                                 "y1_r\: (\S+).*"
                                 "x2_r\: (\S+).*"
                                 "y2_r\: (\S+).*", errString)
            if imageErr:
                ret["image_path"] = imageErr.group(1)
                ret["class"] = imageErr.group(2)
                ret["box"] = imageErr.group(3)

                #x1
                ret["x1"] = imageErr.group(4)
                try:
                    long(float(ret["x1"]))
                except:
                    ret["x1"] = 1e30
                ###########

                #y1
                ret["y1"] = imageErr.group(5)
                try:
                    long(float(ret["y1"]))
                except:
                    ret["y1"] = 1e30
                ###########

                #x2
                ret["x2"] = imageErr.group(6)
                try:
                    long(float(ret["x2"]))
                except:
                    ret["x2"] = 1e30
                ############

                #y2
                ret["y2"] = imageErr.group(7)
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



