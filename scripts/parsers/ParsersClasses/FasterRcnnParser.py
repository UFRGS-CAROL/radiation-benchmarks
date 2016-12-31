import math
import os
import re
from SupportClasses import Rectangle
import numpy as np
from SupportClasses import GoldContent as gc
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

    # def generatePyFasterRectangles(self, dets, thresh=0):
    #     """Draw detected bounding boxes."""
    #     inds = np.where(dets[:, -1] >= thresh)[0]
    #
    #     if len(inds) == 0:
    #         return
    #
    #     # im = im[:, :, (2, 1, 0)]
    #     # fig, ax = plt.subplots(figsize=(12, 12))
    #     # ax.imshow(im, aspect='equal')
    #     bboxList = []
    #     scoresList = []
    #     for i in inds:
    #         bbox = dets[i, :4]
    #         score = dets[i, -1]
    #         scoresList.append(score)
    #         # left = int(math.floor(float(i["x_r"])))
    #         # bottom = int(math.floor(float(i["y_r"])))
    #         # h = int(math.ceil(float(i["h_r"])))
    #         # w = int(math.ceil(float(i["w_r"])))
    #         # tempBoxes[boxPos] = Rectangle(left, bottom, w, h)
    #         left = int(math.floor(float(bbox[0])))
    #         bottom = int(math.floor(float(bbox[1])))
    #         w = int(math.ceil(bbox[2] - bbox[0]))
    #         h = int(math.ceil(bbox[3] - bbox[1]))
    #         bboxList.append(Rectangle(left, bottom, w, h))
        # ax.add_patch(
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
        # return scoresList, bboxList


    # parse PyFaster
    def parseErrMethod(self,errString):
        # ERR boxes: [27,4] e: 132.775177002 r: 132.775024414
        ret = {}
        if 'boxes' in errString:
            dictBox = self._processBoxes(errString)
            if len(dictBox) > 0:
                ret["boxes"] = dictBox
                ret["type"] = "boxes"
        elif 'scores' in errString:
            self._processScores(errString)

        return (ret if len(ret) > 0 else None)

    def _processBoxes(self, errString):
        ret = {}
        image_err = re.match(
            ".*boxes\: \[(\d+),(\d+)\].*e\: (\S+).*r\: (\S+).*", errString)
        if image_err:
            ret["type"] = "boxes"
            # ret["imgindex"] = imgIndex
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



    def buildImageMethod(self):
        return False
    # def generatePyFasterRectangles(self, goldArray): return None
    def getCls(self, gold): return None

    def getImgLPos(self, sdcit): return None





    def py_cpu_nms(self, dets, thresh):
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
