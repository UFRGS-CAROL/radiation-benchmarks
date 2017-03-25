import numpy
import csv
from math import *
from SupportClasses import Rectangle
import os
import re
import glob, struct
from ObjectDetectionParser import ObjectDetectionParser
from SupportClasses import _GoldContent
from ObjectDetectionParser import ImageRaw
from SupportClasses import PrecisionAndRecall


class DarknetParser(ObjectDetectionParser):
    __executionType = None
    __executionModel = None

    __weights = None
    __configFile = None
    _extendedHeader = False

    __errorTypes = ['allLayers', 'filtered2', 'filtered5', 'filtered50']
    __infoNames = ['smallestError', 'biggestError', 'numErrors', 'errorsAverage', 'errorsStdDeviation']
    __filterNames = ['allErrors', 'newErrors', 'propagatedErrors']
    
    _smallestError = None
    _biggestError = None
    _numErrors = None
    _errorsAverage = None
    _errorsStdDeviation = None
    
    __layerDimentions = {
        0: [224, 224, 64],
        1: [112, 112, 64],
        2: [112, 112, 192],
        3: [56, 56, 192],
        4: [56, 56, 128],
        5: [56, 56, 256],
        6: [56, 56, 256],
        7: [56, 56, 512],
        8: [28, 28, 512],
        9: [28, 28, 256],
        10: [28, 28, 512],
        11: [28, 28, 256],
        12: [28, 28, 512],
        13: [28, 28, 256],
        14: [28, 28, 512],
        15: [28, 28, 256],
        16: [28, 28, 512],
        17: [28, 28, 512],
        18: [28, 28, 1024],
        19: [14, 14, 1024],
        20: [14, 14, 512],
        21: [14, 14, 1024],
        22: [14, 14, 512],
        23: [14, 14, 1024],
        24: [14, 14, 1024],
        25: [7, 7, 1024],
        26: [7, 7, 1024],
        27: [7, 7, 1024],
        28: [7, 7, 256],
        29: [12544],
        30: [1175],
        31: [1175]}

    _csvHeader = ["logFileName", "Machine", "Benchmark", "SDC_Iteration", "#Accumulated_Errors", "#Iteration_Errors",
                  "gold_lines", "detected_lines", "wrong_elements", "x_center_of_mass", "y_center_of_mass", "precision",
                  "recall", "false_negative", "false_positive", "true_positive", "abft_type", "row_detected_errors",
                  "col_detected_errors", "failed_layer", "header"]
    
    # it is only for darknet for a while
    _parseLayers = False
    __layersGoldPath = ""
    __layersPath = ""

    def __init__(self, **kwargs):
        ObjectDetectionParser.__init__(self, **kwargs)
        self._parseLayers = bool(kwargs.pop("parseLayers"))

        try:
            if self._parseLayers:
                self.__layersGoldPath = str(kwargs.pop("layersGoldPath"))
                self.__layersPath = str(kwargs.pop("layersPath"))
                self._extendedHeader = True
                self._csvHeader.extend(self.getLayerHeaderName(layerNum, infoName, filterName)
                                       	for filterName in self.__filterNames
                                	for infoName in self.__infoNames
                        	        for layerNum in xrange(32))
		self._csvHeader.extend(self.getLayerHeaderNameErrorType(layerNum)
					for layerNum in xrange(32))
        except:
            print "\n Crash on create parse layers parameters"
            sys.exit(-1)

    _failed_layer = None


    def getLayerHeaderNameErrorType(self, layerNum):
        # layer3<layerNum>ErrorType
        layerHeaderName = 'layer' + str(layerNum) + 'ErrorType'
        return layerHeaderName

    def getLayerHeaderName(self, layerNum, infoName, filterName):
        # layerHeaderName :: layer<layerNum><infoName>_<filterName>
        # <infoName> :: 'smallestError', 'biggestError', 'numErrors', 'errorsAverage', 'errorsVariance'
        # <filterName> :: 'allErrors', 'newErrors', 'propagatedErrors'
        layerHeaderName = 'layer' + str(layerNum) + infoName + '_' + filterName
        return layerHeaderName

    def errorTypeToString(self, errorType):
        if (errorType[0] == 1):
            return "cubic"
        elif (errorType[1] == 1):
            return "square"
        elif (errorType[2] == 1):
            return "colORrow"
        elif (errorType[3] == 1):
            return "single"
        elif (errorType[4] == 1):
            return "random"
        else:
            return "no errors"

    def _writeToCSV(self, csvFileName):
        self._writeCSVHeader(csvFileName)

        try:
            csvWFP = open(csvFileName, "a")
            writer = csv.writer(csvWFP, delimiter=';')
            # ["logFileName", "Machine", "Benchmark", "imgFile", "SDC_Iteration",
            #     "#Accumulated_Errors", "#Iteration_Errors", "gold_lines",
            #     "detected_lines", "x_center_of_mass", "y_center_of_mass",
            #     "precision", "recall", "false_negative", "false_positive",
            #     "true_positive", "abft_type", "row_detected_errors",
            #     "col_detected_errors", "failed_layer", "header"]
            outputList = [self._logFileName,
                          self._machine,
                          self._benchmark,
                          self._sdcIteration,
                          self._accIteErrors,
                          self._iteErrors,
                          self._goldLines,
                          self._detectedLines,
                          self._wrongElements,
                          self._xCenterOfMass,
                          self._yCenterOfMass,
                          self._precision,
                          self._recall,
                          self._falseNegative,
                          self._falsePositive,
                          self._truePositive,
                          self._abftType,
                          self._rowDetErrors,
                          self._colDetErrors,
                          self._failed_layer,
                          self._header]

            if (self._parseLayers):
                for filterName in self.__filterNames:
                    outputList.extend([(self._smallestError[filterName][i] if self._smallestError[filterName][i] else "")
                                      for i in xrange(32)])
                    outputList.extend([(self._biggestError[filterName][i] if self._biggestError[filterName][i] else "")
                                      for i in xrange(32)])
                    outputList.extend([(self._numErrors[filterName][i] if self._numErrors[filterName][i] else "")
                                      for i in xrange(32)])
                    outputList.extend([(self._errorsAverage[filterName][i] if self._errorsAverage[filterName][i] else "")
                                      for i in xrange(32)])
                    outputList.extend([(self._errorsStdDeviation[filterName][i] if self._errorsStdDeviation[filterName][i] else "")
                                      for i in xrange(32)])
		outputList.extend(self.errorTypeToString(self.errorTypeList[i]) for i in xrange(32))

                # if self._abftType != 'no_abft' and self._abftType != None:
                #     outputList.extend([])

            writer.writerow(outputList)
            csvWFP.close()

        except:
            print "\n PAU NO CSV\n"
            raise
            
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
                self._imgListPath = darknetM.group(3)
                self.__weights = darknetM.group(4)
                self.__configFile = darknetM.group(5)
                self._iterations = darknetM.group(6)
                if "abft" in header:
                    self._abftType = darknetM.group(7)

                self._goldFileName = self._datasets[os.path.basename(self._imgListPath)][self._abftType]

            except:
                self._imgListPath = None

        # return self.__imgListFile
        # tempPath = os.path.basename(self.__imgListPath).replace(".txt","")
        self._size = str(self._goldFileName)

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
                    # check image bounds
                    rect = Rectangle.Rectangle(int(xmin), int(ymin), box.width, box.height)
                    validRectangles.append(rect)
                    validClasses.append(self._classes[j])
                    # print self._classes[j]

        return validRectangles, validProbs, validClasses

    def __setLocalFile(self, listFile, imgPos):
        tmp = (listFile[imgPos].rstrip()).split('radiation-benchmarks')[1]
        tmp = self._localRadiationBench + '/radiation-benchmarks' + tmp
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
        for i in xrange(0, n):
            for j in xrange(0, m):
                ret[i][j] = prob[i][j]
        return ret

    def getSizeOfLayer(self, layerNum):
        # retorna o numero de valores de uma layer
        dim = self.__layerDimentions[layerNum]
        layerSize = 0
        if (layerNum < 29):
            layerSize = dim[0] * dim[1] * dim[2]
        else:
            layerSize = dim[0]
        return layerSize

    def tupleToArray(self, layerContents, layerNum):
        size = self.__layerDimentions[layerNum][0]
        layer = [0 for i in xrange(0, size)]
        for i in range(0, size):
            layer[i] = layerContents[i]
        return layer

    def tupleTo3DMatrix(self, layerContents, layerNum):
        dim = self.__layerDimentions[layerNum]  # width,height,depth
        layer = [[[0 for k in xrange(dim[2])] for j in xrange(dim[1])] for i in xrange(dim[0])]
        for i in range(0, dim[0]):
            for j in range(0, dim[1]):
                for k in range(0, dim[2]):
                    contentsIndex = (i * dim[1] + j) * dim[2] + k
                    layer[i][j][k] = layerContents[contentsIndex]
        return layer

    def printLayersSizes(self):
        layerSize = [0 for k in xrange(32)]
        for i in range(0, 32):
            contentsLen = 0
            for filename in glob.glob(self.__layersPath + '*_' + str(i)):
                if (layerSize[i] == 0):
                    layerSize[i] = self.getSizeOfLayer(filename)

                layerFile = open(filename, "rb")
                numItens = layerSize[i] / 4  # float size = 4bytes
                layerContents = struct.unpack('f' * numItens, layerFile.read(4 * numItens))

                # print(filename + " :")
                # layerSize[i] = len(layerContents)
                # print("len: " + str(layerSize[i]))
                # for item in layerContents:
                # print(str(type(item)))
                layerFile.close()
                contentsLen = len(layerContents)
            print("layer " + str(i) + " size=" + str(layerSize[i]) + " contentSize=" + str(contentsLen))

    def getRelativeError(self, expected, read):
        absoluteError = abs(expected - read)
        relativeError = abs(absoluteError / expected) * 100
        return relativeError

    def get1DLayerErrorList(self, layerArray, goldArray, size):
    #sem filtered2 e etc
        # layerError :: xPos, yPos, zPos, found(?), expected(?)
        # layerErrorList :: [layerError]
        layerErrorList = []
        for i in range(0, size):
            if (layerArray[i] != goldArray[i]):
                relativeError = self.getRelativeError(goldArray[i], layerArray[i])
                layerError = [i, -1, -1, layerArray[i], goldArray[i]]
                layerErrorList.append(layerError)

        return layerErrorList

    def get1DLayerErrorLists(self, layerArray, goldArray, size):
    #funcao otimizada (com filtered2 e etc)
        # layerError :: xPos, yPos, zPos, found(?), expected(?)
        # layerErrorLists :: {[allLlayerErrors], [filtered2LayerErrors], [filtered5LayerErrors], [filtered50LayerErrors]}
        layerErrorLists =  {errorTypeString:[] for errorTypeString in self.__errorTypes}
        for i in range(0, size):
            if (layerArray[i] != goldArray[i]):
                relativeError = self.getRelativeError(goldArray[i], layerArray[i])
                layerError = [i, -1, -1, layerArray[i], goldArray[i]]
                layerErrorLists['allLayers'].append(layerError)
                if(relativeError > 2):
                    layerErrorLists['filtered2'].append(layerError)
                if(relativeError > 5):
                    layerErrorLists['filtered5'].append(layerError)
                if(relativeError > 50):
                    layerErrorLists['filtered50'].append(layerError)

        return layerErrorLists

    def get3DLayerErrorList(self, layer, gold, width, height, depth):
        # sem filtered2 e etc
        # layerError :: xPos, yPos, zPos, found(?), expected(?)
        layerErrorList = []
        for i in range(0, width):
            for j in range(0, height):
                for k in range(0, depth):
                    if (layer[i][j][k] != gold[i][j][k]):
                        relativeError = self.getRelativeError(gold[i][j][k], layer[i][j][k])
                        layerError = [i, j, k, layer[i][j][k], gold[i][j][k]]
                        layerErrorList.append(layerError)

        return layerErrorList

    def get3DLayerErrorLists(self, layer, gold, width, height, depth):
    #funcao otimizada (com filtered2 e etc)
        # layerError :: xPos, yPos, zPos, found(?), expected(?)
        # layerErrorLists :: {[allLayerErrors], [filtered2LayerErrors], [filtered5LayerErrors], [filtered50LayerErrors]}
        layerErrorLists =  {errorTypeString:[] for errorTypeString in self.__errorTypes}
        for i in range(0, width):
            for j in range(0, height):
                for k in range(0, depth):
                    if (layer[i][j][k] != gold[i][j][k]):
                        relativeError = self.getRelativeError(gold[i][j][k], layer[i][j][k])
                        layerError = [i, j, k, layer[i][j][k], gold[i][j][k]]
                        layerErrorLists['allLayers'].append(layerError)
                        if(relativeError > 2):
                            layerErrorLists['filtered2'].append(layerError)
                        if(relativeError > 5):
                            layerErrorLists['filtered5'].append(layerError)
                        if(relativeError > 50):
                            layerErrorLists['filtered50'].append(layerError)
        return layerErrorLists

    def getLayerDimentions(self, layerNum):
        width = 0
        height = 0
        depth = 0
        isArray = False
        if (len(self.__layerDimentions[layerNum]) == 3):
            width = self.__layerDimentions[layerNum][0]
            height = self.__layerDimentions[layerNum][1]
            depth = self.__layerDimentions[layerNum][2]
        elif (len(self.__layerDimentions[layerNum]) == 1):
            # as camadas 29, 30 e 31 sao apenas arrays
            width = self.__layerDimentions[layerNum][0]
            isArray = True
        else:
            print("erro: dicionario ta bugado")

        return isArray, width, height, depth

    def loadLayer(self, layerNum):
        # carrega de um log para uma matriz
        # print('_logFileName: ' + self._logFileName[])
        # print('_sdcIteration: ' + self._sdcIteration)
        sdcIteration = self._sdcIteration
        if self._isFaultInjection:
            sdcIteration = str(int(sdcIteration) + 1)
            # print 'debug' + sdcIteration
        layerFilename = self.__layersPath + self._logFileName + "_it_" + sdcIteration + "_layer_" + str(layerNum)
        #layerFilename = self.__layersPath  + '2017_03_15_04_15_52_cudaDarknet_carolk402.log_it_0_layer_' + str(layerNum)
        # layerFilename = self.__layersGoldPath + '2017_02_22_09_08_51_cudaDarknet_carol-k402.log_it_64_layer_' + str(layerNum)
        # print self.__layersPath   + layerFilename
        filenames = glob.glob(layerFilename)
        # print '_logFilename: ' + self._logFileName
        # print str(filenames)
        if (len(filenames) == 0):
            return None
        elif (len(filenames) > 1):
            print('+de 1 log encontrado para \'' + layerFilename + '\'')

        filename = filenames[0]
        layerSize = self.getSizeOfLayer(layerNum)

        layerFile = open(filename, "rb")
        numItens = layerSize  # float size = 4bytes

        layerContents = struct.unpack('f' * numItens, layerFile.read(4 * numItens))
        # botar em matriz 3D
        if (layerNum < 29):
            layer = self.tupleTo3DMatrix(layerContents, layerNum)
        else:
            layer = self.tupleToArray(layerContents, layerNum)
        layerFile.close()
        # print("load layer " + str(layerNum) + " size = " + str(layerSize) + " filename: " + filename + " len(layer) = " + str(len(layer)))
        return layer

    def getDatasetName(self):
        if self._goldFileName == 'gold.caltech.1K.test' or self._goldFileName == 'gold.caltech.abft.1K.test':
            return '_caltech1K'
        elif self._goldFileName == 'gold.voc.2012.1K.test' or self._goldFileName == 'gold.voc.2012.abft.1K.test':
            return '_voc2012'
        elif self._goldFileName == 'gold.caltech.critical.1K.test' or self._goldFileName == 'gold.caltech.critical.abft.1K.test':
            return '_caltechCritical'
        else:
            print 'erro getDatasetName: ' + self._goldFileName + ' nao classificado'
            return ''

    def loadGoldLayer(self, layerNum):
        # carrega de um log para uma matriz
        datasetName = self.getDatasetName()
        goldIteration = str(int(self._sdcIteration) % self._imgListSize)
        # print 'dataset? ' + self._goldFileName + '  it ' + self._sdcIteration + '  abft: ' + self._abftType
        layerFilename = self.__layersGoldPath + "gold_" + self._machine + datasetName + '_it_' + goldIteration + '_layer_' + str(
            layerNum)
        # layerFilename = self.__layersGoldPath + '2017_02_22_09_08_51_cudaDarknet_carol-k402.log_it_64_layer_' + str(layerNum)
        #print layerFilename
        filenames = glob.glob(layerFilename)
        #print str(filenames)
        if (len(filenames) == 0):
            return None
        elif (len(filenames) > 1):
            print('+de 1 gold encontrado para \'' + layerFilename + str(layerNum) + '\'')

        layerSize = self.getSizeOfLayer(layerNum)

        layerFile = open(filenames[0], "rb")
        numItens = layerSize  # float size = 4bytes

        layerContents = struct.unpack('f' * numItens, layerFile.read(4 * numItens))

        # botar em matriz 3D
        if (layerNum < 29):
            layer = self.tupleTo3DMatrix(layerContents, layerNum)
        else:
            layer = self.tupleToArray(layerContents, layerNum)
        layerFile.close()
        # print("load layer " + str(layerNum) + " size = " + str(layerSize) + " filename: " + filename + " len(layer) = " + str(len(layer)))
        return layer

    def _localityParser1D(self, layerErrorList):
        # errorType :: cubic, square, colOrRow, single, random
        # layerError :: xPos, yPos, zPos, found(?), expected(?)
        # layerErrorLists :: {[allLlayerErrors], [filtered2LayerErrors], [filtered5LayerErrors]}
        if len(layerErrorList) < 1:
            return [0, 0, 0, 0, 0]
        elif len(layerErrorList) == 1:
            return [0, 0, 0, 1, 0]
        else:
            errorInSequence = False
            lastErrorPos = -2
            for layerError in layerErrorList:
                if (layerError[0] == lastErrorPos + 1):
                    errorInSequence = True
                lastErrorPos = layerError[0]
            if errorInSequence:
                return [0, 0, 1, 0, 0]
            else:
                return [0, 0, 0, 0, 1]

    def getLayerErrorList(self, layer, gold, layerNum):
        # layerError :: xPos, yPos, zPos, found(?), expected(?)
        # layerErrorLists :: [layerError]
        isArray, width, height, depth = self.getLayerDimentions(layerNum)
        errorList  = []
        if (isArray):
            errorList = self.get1DLayerErrorList(layer, gold, width)
        else:
            errorList = self.get3DLayerErrorList(layer, gold, width, height, depth)
        return errorList

    def printErrorType(self, errorType):
        if (errorType[0] == 1):
            print("cubic error")
        elif (errorType[1] == 1):
            print("square error")
        elif (errorType[2] == 1):
            print("column or row error")
        elif (errorType[3] == 1):
            print("single error")
        elif (errorType[4] == 1):
            print("random error")
        else:
            print("no errors")

    def jaccard_similarity(self, x, y):

        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality / float(union_cardinality)

    def layer3DToArray(self, layer, layerNum):
        width, height, depth = self.__layerDimentions[layerNum]
        layerSize = self.getSizeOfLayer(layerNum)
        layerArray = [0 for k in xrange(layerSize)]
        for i in range(0, width):
            for j in range(0, height):
                for k in range(0, depth):
                    arrayIndex = (i * height + j) * depth + k
                    layerArray[arrayIndex] = layer[i][j][k]
        return layerArray

    def getErrorListInfos(self, layerErrorList):
        # layerError :: xPos, yPos, zPos, found(?), expected(?)
        smallest = 0.0
        biggest = 0.0
        average = 0.0
        stdDeviation = 0.0
        #totalSum = long(0)
        relativeErrorsList = []
        for layerError in layerErrorList:
            relativeError = self.getRelativeError(layerError[4], layerError[3])
            relativeErrorsList.append(relativeError)
            #totalSum = totalSum + relativeError
            average += relativeError/len(layerErrorList)

            if(smallest == 0.0 and biggest == 0.0):
                smallest = relativeError
                biggest = relativeError
            else:
                if(relativeError < smallest):
                    smallest = relativeError
                if(relativeError > biggest):
                    biggest = relativeError
	#print('\ndebug totalSum: ' + str(totalSum))
        #average = totalSum/len(layerErrorList)
	#average = numpy.average(relativeErrorsList)
        if(average != 0.0):
            stdDeviation = numpy.std(relativeErrorsList)

        return smallest, biggest, average, stdDeviation


    def parseLayers(self):
        ### faz o parsing de todas as camadas de uma iteracao
        # errorType :: [cubic, square, colOrRow, single, random]
        # layerError :: xPos, yPos, zPos, found(?), expected(?)
        # layerErrorLists :: {[allLlayerErrors], [filtered2LayerErrors], [filtered5LayerErrors]}
        # print ('\n' + self._logFileName + ' :: ' + self._goldFileName + ' :: ' + self._imgListPath)
        #kernelTime = int(self._accIteErrors) // int(self._sdcIteration)
        #print '\nkerneltime: ' + str(kernelTime)

        self._smallestError = {filterName:[0.0 for i in xrange(32)] for filterName in self.__filterNames}
        self._biggestError = {filterName:[0.0 for i in xrange(32)] for filterName in self.__filterNames}
        self._numErrors = {filterName:[0 for i in xrange(32)] for filterName in self.__filterNames}
        self._errorsAverage = {filterName:[0.0 for i in xrange(32)] for filterName in self.__filterNames}
        self._errorsStdDeviation = {filterName:[0.0 for i in xrange(32)] for filterName in self.__filterNames}
        self._failed_layer = ""
        logsNotFound = False
        goldsNotFound = False
        self._errorFound = False
        self.errorTypeList = [ [] for i in range(0,32)]

        for i in range(0, 32):
            # print '\n----layer ' + str(i) + ' :'
            layer = self.loadLayer(i)
            gold = self.loadGoldLayer(i)
            if (layer is None):
                print(self._machine + ' it: ' + str(self._sdcIteration) + ' layer ' + str(i) + ' log not found')
                logsNotFound = True
                break
            elif (gold is None):
                print('gold ' + str(i) + ' log not found')
                goldsNotFound = True
                break
            else:
                layerErrorList = self.getLayerErrorList(layer, gold, i)
		if(len(layerErrorList)>0):
                    self._numErrors['allErrors'][i] = len(layerErrorList)
                smallest, biggest, average, stdDeviation = self.getErrorListInfos(layerErrorList)
                self._smallestError['allErrors'][i] = smallest
                self._biggestError['allErrors'][i] = biggest
                self._errorsAverage['allErrors'][i] = average
                self._errorsStdDeviation['allErrors'][i] = stdDeviation
                if(self._errorFound):
                    #ja tinha erros em alguma camada anterior
                    self._numErrors['propagatedErrors'][i] = self._numErrors['allErrors'][i]
                    self._smallestError['propagatedErrors'][i] = self._smallestError['allErrors'][i]
                    self._biggestError['propagatedErrors'][i] = self._biggestError['allErrors'][i]
                    self._errorsAverage['propagatedErrors'][i] = self._errorsAverage['allErrors'][i]
                    self._errorsStdDeviation['propagatedErrors'][i] = self._errorsStdDeviation['allErrors'][i]
                else:
                    self._numErrors['newErrors'][i] = self._numErrors['allErrors'][i]
                    self._smallestError['newErrors'][i] = self._smallestError['allErrors'][i]
                    self._biggestError['newErrors'][i] = self._biggestError['allErrors'][i]
                    self._errorsAverage['newErrors'][i] = self._errorsAverage['allErrors'][i]
                    self._errorsStdDeviation['newErrors'][i] = self._errorsStdDeviation['allErrors'][i]
                if(False): #i == 31):
                    print('\nlogName : ' + self._logFileName)
                    print('numErrors camada ' + str(i) + ' :: ' + str(len(layerErrorList)))
                    print('smallestError camada ' + str(i) + ' :: ' + str(smallest))
                    print('biggestError camada ' + str(i) + ' :: ' + str(biggest))
                    print('errorsAverage camada ' + str(i) + ' :: ' + str(average))
                    print('errorsStdDeviation camada ' + str(i) + ' :: ' + str(stdDeviation))
                    print('ja deu erro? ' + str(self._errorFound))
                    print('Precision: ' + str(self._precision) + '  Recall: ' + str(self._recall) + '\n')
                if (i < 29):
                    # layer 3D
                    self.errorTypeList[i] = self._localityParser3D(layerErrorList)
                    if (self.errorTypeList[i] != [0, 0, 0, 0, 0]):
                        # aconteceu algum tipo de erro
                        if (not self._errorFound):
                            self._failed_layer = str(i)
                            self._errorFound = True
                            # layerArray = self.layer3DToArray(layer, i)
                            # goldArray = self.layer3DToArray(gold, i)
                            # jaccardCoef = self.jaccard_similarity(layerArray,goldArray)
                    else:
                        # nao teve nenhum erro
                        jaccardCoef = 1
                else:
                    # layer 1D
                    self.errorTypeList[i] = self._localityParser1D(layerErrorList)
                    if (self.errorTypeList[i] != [0, 0, 0, 0, 0]):
                        # aconteceu algum tipo de erro
                        if (not self._errorFound):
                            self._failed_layer = str(i)
                            self._errorFound = True
                            # jaccardCoef = self.jaccard_similarity(layer,gold)
                    else:
                        # nao teve nenhum erro
                        jaccardCoef = 1
                        # print('jaccard = ' + str(jaccardCoef))

        if logsNotFound and goldsNotFound:
            self._failed_layer += 'golds and logs not found'
        elif logsNotFound:
            self._failed_layer += 'logs not found'
        elif goldsNotFound:
            self._failed_layer += 'golds not found'
        # print('failed_layer: ' + self._failed_layer + '\n')
        pass


    def _relativeErrorParser(self, errList):
        if len(errList) <= 0:  # or '2017_03_06_12_51_03_cudaDarknet_carolk402.log' not in self._logFileName:
            return

        if self._abftType == 'dumb_abft' and 'abft' not in self._goldFileName:
            print "\n\n", self._goldFileName
        goldKey = self._machine + "_" + self._benchmark + "_" + self._goldFileName

        if self._machine in self._goldBaseDir:
            goldPath = self._goldBaseDir[self._machine] + "/darknet/" + self._goldFileName
            txtPath = self._goldBaseDir[self._machine] + '/networks_img_list/' + os.path.basename(self._imgListPath)
        else:
            # if self._machine == 'carolk402':  # errors_log_database_inst foi gerado com essa string
            #     self._machine = 'carol-k402'
            #     goldPath = self.__goldBaseDir [self._machine] + "/darknet/" + self._goldFileName
            #     txtPath = self.__goldBaseDir [self._machine] + '/networks_img_list/' + os.path.basename(self._imgListPath)
            #     self._isInstLayers = True
            # else:
            print 'not indexed machine: ' , self._machine , " set it on Parameters.py"
            return

        if goldKey not in self._goldDatasetArray:
            g = _GoldContent._GoldContent(nn='darknet', filepath=goldPath)
            self._goldDatasetArray[goldKey] = g

        gold = self._goldDatasetArray[goldKey]

        listFile = open(txtPath).readlines()

        # set imglist size for this SDC, it will be set again for other SDC
        self._imgListSize = len(listFile)
        imgPos = int(self._sdcIteration) % self._imgListSize
        imgFilename = self.__setLocalFile(listFile, imgPos)
        imgObj = ImageRaw(imgFilename)

        goldPb = gold.getProbArray()[imgPos]
        goldRt = gold.getRectArray()[imgPos]

        goldPb = self.newMatrix(goldPb, gold.getTotalSize(), gold.getClasses())
        goldRt = self.newRectArray(goldRt)

        foundPb = self.newMatrix(goldPb, gold.getTotalSize(), gold.getClasses())
        foundRt = self.newRectArray(goldRt)

        self._rowDetErrors = 0
        self._colDetErrors = 0
        self._wrongElements = 0

        for y in errList:
            # print y['img_list_position'], imgPos
            # print y
            if y["type"] == "boxes":
                i = y['boxes']
                rectPos = i['box_pos']
                # found
                lr = int(float(i["x_r"]))
                br = int((float(i["y_r"])))
                hr = abs(int(float(i["h_r"])))
                wr = abs(int(float(i["w_r"])))
                # gold correct
                le = int(float(i["x_e"]))
                be = int(float(i["y_e"]))
                he = int(float(i["h_e"]))
                we = int(float(i["w_e"]))

                foundRt[rectPos] = Rectangle.Rectangle(lr, br, wr, hr)
                # t = goldRt[rectPos].deepcopy()
                goldRt[rectPos] = Rectangle.Rectangle(le, be, we, he)
                # if not (t == goldRt[rectPos]):
                #     print "\n", goldRt[rectPos]
                #     print t
                self._wrongElements += 1
            elif y["type"] == "abft" and self._abftType != 'no_abft':

                i = y['abft_det']
                self._rowDetErrors += i["row_detected_errors"]
                self._colDetErrors += i["col_detected_errors"]

            elif y["type"] == "probs":
                i = int(y["probs"]["probs_x"])
                j = int(y["probs"]["probs_y"])

                # if math.fabs(float(y["probs"]["prob_e"]) - foundPb[i][j]) > 0: print "\n" , foundPb[i][j] , float(y["probs"]["prob_r"]) , y["probs"]["prob_e"]

                foundPb[i][j] = float(y["probs"]["prob_r"])
                goldPb[i][j] = float(y["probs"]["prob_e"])

        # if self._abftType != 'no_abft':
        #    print str(self._sdcIteration) + ' : ' + self._abftType 

        if self._rowDetErrors > 1e6:
            self._rowDetErrors /= long(1e15)
        if self._colDetErrors > 1e6:
            self._colDetErrors /= long(1e15)

        #############
        # before keep going is necessary to filter the results
        gValidRects, gValidProbs, gValidClasses = self.__printYoloDetections(goldRt, goldPb, gold.getTotalSize(),
                                                                             len(self._classes) - 1)
        fValidRects, fValidProbs, fValidClasses = self.__printYoloDetections(foundRt, foundPb, gold.getTotalSize(),
                                                                             len(self._classes) - 1)

        precisionRecallObj = PrecisionAndRecall.PrecisionAndRecall(self._prThreshold)
        gValidSize = len(gValidRects)
        fValidSize = len(fValidRects)

        precisionRecallObj.precisionAndRecallParallel(gValidRects, fValidRects)
        self._precision = precisionRecallObj.getPrecision()
        self._recall = precisionRecallObj.getRecall()

        if self._parseLayers: #and self.hasLayerLogs(self._sdcIteration):
            # print self._sdcIteration + 'debug'
            self.parseLayers()
            # print self._machine + self._abftType

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
