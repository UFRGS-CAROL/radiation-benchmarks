import os
import re

from ObjectDetectionParser import  ObjectDetectionParser
from SupportClasses import PrecisionAndRecall
from SupportClasses import _GoldContent

GOLD_BASE_DIR = {
    'carol-k402': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_K40',
    'carol-tx': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_TITAN',
    #carolx1a
    'carolx1a': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_X1/tx1b',
    #carolx1b
    'carolx1b': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_X1/tx1b',
    #carolx1c
    'carolx1c': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_X1/tx1c',
     # '/home/familia/Dropbox/UFRGS/Pesquisa/fault_injections/sassifi_darknet'
}

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

    # def __init__(self):
    #     start_time = time.time()
    #     ObjectDetectionParser.__init__(self)
    #     for kD, vD in DATASETS.iteritems():
    #         for kI, vI in vD.iteritems():
    #             for i in GOLD_BASE_DIR:
    #                 DATASETS[kD][kI]['gold'] = str(i) + '/py_faster_rcnn/' + str(kD)
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


        goldKey = self._machine + "_" + self._benchmark + "_" + self.__goldFileName

        if self._machine in GOLD_BASE_DIR:
            goldPath = GOLD_BASE_DIR[self._machine] + "/darknet/" + self.__goldFileName
            txtPath = GOLD_BASE_DIR[self._machine] + '/networks_img_list/' + os.path.basename(self.__imgListPath)
        else:
            print self._machine
            return

        if goldKey not in self._goldDatasetArray:
            g = _GoldContent._GoldContent(nn='darknet', filepath=goldPath)
            self._goldDatasetArray[goldKey] = g

        gold = self._goldDatasetArray[goldKey]

        listFile = open(txtPath).readlines()

        imgPos = int(self._sdcIteration) % len(listFile)
        imgFilename = self.__setLocalFile(listFile, imgPos)
        imgObj = ObjectDetectionParser.ImageRaw(imgFilename)



        self._wrongElements = 0
        for y in errList:
            print y




        precisionRecallObj = PrecisionAndRecall.PrecisionAndRecall(self._prThreshold)
        gValidSize = len(gValidRects)
        fValidSize = len(fValidRects)

        precisionRecallObj.precisionAndRecallParallel(gValidRects, fValidRects)
        self._precision = precisionRecallObj.getPrecision()
        self._recall = precisionRecallObj.getRecall()

        # if self._logFileName == '2016_02_11_16_28_06_cudaDarknet_carolx1b.log':
        #     print "\n", gValidRects
        #     print "\n", fValidRects
        #     # print goldPb[46][14] #0.591782
            # print goldPb[31][14] #0.461120
            # print goldPb[54][11] #0.579033
            # print goldPb[95][11] #0.411528
        # if 0< self._precision < 1.0 or 0< self._recall < 1.0:
        #     print "\n", self._precision , self._recall
        #     print self._logFileName
            # print "\n", gValidRects
            # print "\n", fValidRects
            #sys.exit()

        # self.buildImageMethod(listFile[imgPos].rstrip(), gValidRects, fValidRects)

        self._falseNegative = precisionRecallObj.getFalseNegative()
        self._falsePositive = precisionRecallObj.getFalsePositive()
        self._truePositive = precisionRecallObj.getTruePositive()
        # set all
        self._goldLines = gValidSize
        self._detectedLines = fValidSize
        self._xCenterOfMass, self._yCenterOfMass = precisionRecallObj.centerOfMassGoldVsFound(gValidRects, fValidRects, imgObj.w, imgObj.h)



    def setSize(self, header):
        # pyfaster
        #HEADER iterations: 1000 img_list: /home/carol/radiation-benchmarks/data/networks_img_list/caltech.pedestrians.1K.txt board: K40
        m = re.match(".*iterations\: (\d+).*img_list\: (\S+).*board\: (\S+).*", header)
        if m:
            self.__iterations = m.group(1)
            self.__imgListPath = m.group(2)
            self.__board = m.group(3)

        self._size = 'py_faster_' + os.path.basename(self.__imgListPath) + '_' + str(self.__board)



