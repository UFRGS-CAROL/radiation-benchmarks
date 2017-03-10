import os
import re
import numpy
from ObjectDetectionParser import ObjectDetectionParser
from ObjectDetectionParser import ImageRaw
from SupportClasses import PrecisionAndRecall
from SupportClasses import _GoldContent
from SupportClasses import Rectangle

class FasterRcnnParser(ObjectDetectionParser):
    # __iterations = None
    __board = None

    def __init__(self, **kwargs):
        ObjectDetectionParser.__init__(self, **kwargs)
        self._detectionThreshold = 0.3


    def getBenchmark(self):
        return self._benchmark

    # parse PyFaster
    def parseErrMethod(self, errString):
        # print errString
        ret = {}
        if 'box' in errString:
            dictBox,imgPath = self._processBoxes(errString)
            if dictBox:
                ret["boxes"] = dictBox
                ret["img_path"] = imgPath
        elif 'score' in errString:
            dictScore,imgPath = self._processScores(errString)
            if dictScore:
                ret["scores"] = dictScore
                ret["img_path"] = imgPath

        return (ret if len(ret) > 0 else None)

    def _processScores(self, errString):
        ret = {}
        # ERR img_name: /home/carol/radiation-benchmarks/data/VOC2012/2011_004360.jpg class: horse wrong_score_size: -17
        # ERR img_name: /home/carol/radiation-benchmarks/data/VOC2012/2011_004360.jpg class: horse score: [0] e: 0.0158654786646 r: 0.00468954769894
        scoreErr = re.match(".*img_name\: (\S+).*"
                            "class\: (\S+).*wrong_score_size\: (\S+).*", errString)
        imgPath = ''
        if scoreErr:
            try:
                ret["wrong_score_size"] = abs(int(scoreErr.group(3)))
            except:
                print "\nerror on parsing wrong_score_size"
                raise

        else:
            scoreErr = re.match(".*img_name\: (\S+).*"
                                "class\: (\S+).*score\: \[(\d+)\].*e\: (\S+).*r\: (\S+).*", errString)

            try:
                ret["score_pos"] = int(scoreErr.group(3))
            except:
                print "\nerror on parsing score pos"
                raise

            try:
                ret["score_e"] = float(scoreErr.group(4))
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
                imgPath = scoreErr.group(1)
                ret["class"] = scoreErr.group(2)
            except:
                print "\nerror on parsing img_path and class"
                raise

        return (ret if len(ret) > 0 else None), imgPath

    def _processBoxes(self, errString):
        ##ERR img_name: /home/carol/radiation-benchmarks/data/CALTECH/set10/V000/193.jpg
        # class: sheep
        # box: [8]
        #  x1_e: 435.740264893 x1_r: 435.782531738
        # y1_e: 244.744735718 y1_r: 244.746307373
        # x2_e: 610.136474609 x2_r: 610.124450684
        # y2_e: 326.088867188 y2_r: 326.093597412
        ret = {}
        imageErr = re.match(".*img_name\: (\S+).*"
                            "class\: (\S+).*box\: \[(\d+)\].*"
                            "x1_e\: (\S+).*x1_r\: (\S+).*"
                            "y1_e\: (\S+).*y1_r\: (\S+).*"
                            "x2_e\: (\S+).*x2_r\: (\S+).*"
                            "y2_e\: (\S+).*y2_r\: (\S+).*", errString)
        imgPath = ''
        if imageErr:
            # ret['generation'] = 1
            imgPath = imageErr.group(1)
            ret["class"] = imageErr.group(2)
            ret["box"] = imageErr.group(3)

            # x1
            ret["x1_e"] = imageErr.group(4)
            try:
                long(float(ret["x1_e"]))
            except:
                ret["x1_e"] = 1e30

            ret["x1_r"] = imageErr.group(5)
            try:
                long(float(ret["x1_r"]))
            except:
                ret["x1_r"] = 1e30
            ###########

            # y1
            ret["y1_e"] = imageErr.group(6)
            try:
                long(float(ret["y1_e"]))
            except:
                ret["y1_e"] = 1e30

            ret["y1_r"] = imageErr.group(7)
            try:
                long(float(ret["y1_r"]))
            except:
                ret["y1_r"] = 1e30
            ###########

            # x2
            ret["x2_e"] = imageErr.group(8)
            try:
                long(float(ret["x2_e"]))
            except:
                ret["x2_e"] = 1e30

            ret["x2_r"] = imageErr.group(9)
            try:
                long(float(ret["x2_r"]))
            except:
                ret["x2_r"] = 1e30
            ############

            # y2
            ret["y2_e"] = imageErr.group(10)
            try:
                long(float(ret["y2_e"]))
            except:
                ret["y2_e"] = 1e30

            ret["y2_r"] = imageErr.group(11)
            try:
                long(float(ret["y2_r"]))
            except:
                ret["y2_r"] = 1e30

        # ERR boxes: [98,12] e: 13.7840118408 r: 13.7840270996
        imageErr = re.match(".*boxes\: \[(\d+),(\d+)\].*e\: (\S+).*r\: (\S+).*", errString)

        # if imageErr:
        #     ret['generation'] = 2
        #     ret["i"] = int(imageErr.group(1))
        #     ret["j"] = int(imageErr.group(2))
        #
        #     ret["e"] = imageErr.group(3)
        #     try:
        #         long(float(ret["e"]))
        #     except:
        #         ret["e"] = 1e30
        #
        #     ret["r"] = imageErr.group(4)
        #     try:
        #         long(float(ret["r"]))
        #     except:
        #         ret["r"] = 1e30

        return (ret if len(ret) > 0 else None), imgPath

    def _relativeErrorParser(self, errList):
        if len(errList) <= 0:
            return

        goldKey = self._machine + "_" + self._benchmark + "_" + self._goldFileName

        if self._machine in self._goldBaseDir:
            goldPath = self._goldBaseDir[self._machine] + "/py_faster_rcnn/" + self._goldFileName
            # txtPath = self._goldBaseDir[self._machine] + '/networks_img_list/' + os.path.basename(self._imgListPath)
        else:
            print "\n not indexed machine ", self._machine
            return

        if goldKey not in self._goldDatasetArray:
            g = _GoldContent._GoldContent(nn='pyfaster', filepath=goldPath)
            self._goldDatasetArray[goldKey] = g
        # else:
        #     print '\nnao passou para ', goldKey
        gold = self._goldDatasetArray[goldKey].getPyFasterGold()


        imgPos = errList[0]['img_path']



        imgFilename = self.__setLocalFile(imgPos)
        imgObj = ImageRaw(imgFilename)

        # to get from gold
        imgFilenameRaw = imgPos.rstrip() if 'radiation-benchmarks' in imgPos else '/home/carol/radiation-benchmarks/data/' + imgPos.rstrip()
        goldImg = gold[imgFilenameRaw]
        # print goldImg
        foundImg = self.__copyGoldImg(goldImg)


        self._wrongElements = 0
        for y in errList:
            #boxes
            if 'boxes' in y:
                cl = str(y['boxes']['class'])
                box = int(y['boxes']['box'])
                x1R = float(y['boxes']['x1_r'])
                x2R = float(y['boxes']['x2_r'])
                y1R = float(y['boxes']['y1_r'])
                y2R = float(y['boxes']['y2_r'])
                r = [x1R, y1R, x2R, y2R]
                # x1E = float(y['boxes']['x1_e'])
                # x2E = float(y['boxes']['x2_e'])
                # y1E = float(y['boxes']['y1_e'])
                # y2E = float(y['boxes']['y2_e'])
                # e = [x1E, y1E, x2E, y2E]
                for i in xrange(0,4): foundImg[cl]['boxes'][box][i] = r[i]
                # # if math.fabs(x1E - goldImg[cl]['boxes'][box][0]) > 0.1:
                # print "\npau no box ", goldImg[cl]['boxes'][box][0], x1E
                #     # sys.exit()
                # # if math.fabs(y1E - goldImg[cl]['boxes'][box][1]) > 0.1:
                # print "\npau no box ", goldImg[cl]['boxes'][box][1], y1E
                #     # sys.exit()
                # # if math.fabs(x2E - goldImg[cl]['boxes'][box][2]) > 0.1:
                # print "\npau no box ", goldImg[cl]['boxes'][box][2] , x2E
                #     # sys.exit()
                # # if math.fabs(y2E - goldImg[cl]['boxes'][box][3]) > 0.1:
                # print "\npau no box ", goldImg[cl]['boxes'][box][3], y2E
                    # sys.exit()
            #scores
            if 'scores' in y:
                if 'score_pos' in y['scores']:
                    sR = float(y['scores']['score_r'])
                    sE = float(y['scores']['score_e'])
                    sP = int(y['scores']['score_pos'])
                    cl = str(y['scores']['class'])
                    foundImg[cl]['scores'][sP] = sR
                    # if 0.1 < math.fabs(goldImg[cl]['scores'][sP] - sE):
                    # print "\npau no score ", goldImg[cl]['scores'][sP], sE
                        # sys.exit()

        gValidRects, gValidProbs, gValidClasses = self.__generatePyFasterDetection(goldImg)
        fValidRects, fValidProbs, fValidClasses = self.__generatePyFasterDetection(foundImg)
        # print gValidRects
        self._abftType = self._rowDetErrors = self._colDetErrors = 'pyfaster'
        precisionRecallObj = PrecisionAndRecall.PrecisionAndRecall(self._prThreshold)
        gValidSize = len(gValidRects)
        fValidSize = len(fValidRects)

        precisionRecallObj.precisionAndRecallParallel(gValidRects, fValidRects)
        if len(gValidRects) == 0 and len(fValidRects) == 0:
            self._precision = 1
            self._recall = 1
        else:
            self._precision = precisionRecallObj.getPrecision()
            self._recall = precisionRecallObj.getRecall()

        if self._imgOutputDir and (self._precision != 1 or self._recall != 1):
            self.buildImageMethod(imgFilename.rstrip(), gValidRects, fValidRects, str(self._sdcIteration)
                                  + '_' + self._logFileName, self._imgOutputDir)

        self._falseNegative = precisionRecallObj.getFalseNegative()
        self._falsePositive = precisionRecallObj.getFalsePositive()
        self._truePositive = precisionRecallObj.getTruePositive()
        # set all
        self._goldLines = gValidSize
        self._detectedLines = fValidSize

        self._xCenterOfMass, self._yCenterOfMass = precisionRecallObj.centerOfMassGoldVsFound(gValidRects, fValidRects,
                                                                                              imgObj.w, imgObj.h)


    #like printYoloDetection
    def __generatePyFasterDetection(self, detection):
        rects = []
        probs = []
        classes = []
        for cls_ind, cls in enumerate(self._classes[1:]):
            valueDet = detection[cls]
            scores = valueDet['scores']
            box  = valueDet['boxes']
            for pb, bbox in zip(scores, box):
                if float(pb) >= self._detectionThreshold:
                    probs.append(float(pb))
                    l = int(float(bbox[0]))
                    b = int(float(bbox[1]))
                    w = int(float(bbox[2]) - l)
                    h = int(float(bbox[3]) - b)
                    rect = Rectangle.Rectangle(l, b, w, h)
                    rects.append(rect)
                    classes.append(cls)

        return rects, probs, classes

    def __copyGoldImg(self, goldImg):
        ret = {}
        for cls_ind, cls in enumerate(self._classes[1:]):
            d = goldImg[cls]
            scores = d['scores']
            box = d['boxes']
            ret[cls] = {'boxes': [], 'scores': []}
            for pb, bbox in zip(scores, box):
                newBbox = numpy.empty(4, dtype=float)
                for i in xrange(0,4): newBbox[i] = bbox[i]
                ret[cls]['boxes'].append(newBbox)
                ret[cls]['scores'].append(float(pb))

        return ret


    def setSize(self, header):
        # pyfaster
        # HEADER iterations: 1000 img_list: /home/carol/radiation-benchmarks/data/networks_img_list/caltech.pedestrians.1K.txt board: K40
        m = re.match(".*iterations\: (\d+).*img_list\: (\S+).*board\: (\S+).*", header)
        if m:
            self._iterations = m.group(1)
            self._imgListPath = m.group(2)
            self.__board = m.group(3)

            self._goldFileName = self._datasets[os.path.basename(self._imgListPath)]

        self._size = 'py_faster_' + os.path.basename(self._imgListPath) + '_' + str(self.__board)


    def __setLocalFile(self, imgPath):
        tmp = ''
        if 'radiation-benchmarks' in imgPath:
            splited = (imgPath.rstrip()).split('radiation-benchmarks/')[1]
        else:
            splited = 'data/' + imgPath

        tmp = self._localRadiationBench + '/radiation-benchmarks/' + splited

        tmp.replace('//', '/')
        return tmp
