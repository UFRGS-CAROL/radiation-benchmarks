from abc import ABCMeta, abstractmethod
from PIL import Image
import struct
from sklearn.metrics import jaccard_similarity_score
import os
import errno
import collections
import csv
import warnings

"""Base class for parser, need be implemented by each benchmark"""


class Parser():
    __metaclass__ = ABCMeta
    #only for parser
    __keys = ["errorsParsed", "errListFiltered", "errListFiltered2"]
    _errors = {}
    #for localityParser2D
    _locality = {}
    #for jaccardCoefficient
    _jaccardCoefficientDict = {}

    def __init__(self):
        for i in self.__keys:
            # _errors["errorsParsed"] = []
            # _errors["errListFiltered"] = []
            # _errors["errListFiltered2"] = []
            self._errors[i] = []
            # _locality["errorsParsed"] = [0, 0, 0, 0, 0]
            # _locality["errListFiltered"] = [0, 0, 0, 0, 0]
            # _locality["errListFiltered2"] = [0, 0, 0, 0, 0]
            # cubic, square, colRow, single, random
            self._locality[i] = [0, 0, 0, 0, 0]
            # _jaccardCoefficinetDict["errorsParsed"] = 0
            # _jaccardCoefficinetDict["errListFiltered"] = 0
            # _jaccardCoefficinetDict["errListFiltered2"] = 0
            self._jaccardCoefficientDict[i] = 0

    _toleratedRelErr = 2.0  # minimum relative error to be considered, in percentage
    _toleratedRelErr2 = 5.0  # minimum relative error to be considered, in percentage
    _buildImages = False  # build locality images
    _headerWriten = False
    _errList = []
    _pureHeader = ""
    _logFileNameNoExt = ""
    _dirName = ""

    #size must be set on the child classes
    _size = ""

    # specific atributes for CSV write
    # logFileName,machine,benchmark,header,sdcIteration,accIteErrors,iteErrors,
    _logFileName = ""
    _machine = ""
    _benchmark = None
    _header = ""
    _sdcIteration = -1
    _accIteErrors = -1
    _iteErrors = -1
    _imageIndex = 0


    _csvHeader = ["logFileName", "Machine", "Benchmark", "Header", "SDC Iteration", "#Accumulated Errors",
                             "#Iteration Errors", "Relative Errors <= " + str(_toleratedRelErr) + "%",
                             "Relative Errors <= " + str(_toleratedRelErr2) + "%", "Jaccard",
                             "Jaccard > " + str(_toleratedRelErr) + "%", "Jaccard > " + str(_toleratedRelErr2) + "%",
                             "Cubic", "Square", "Line", "Single", "Random", "Cubic Err > " + str(_toleratedRelErr),
                  "Square Err > " + str(_toleratedRelErr), "Line Err > " + str(_toleratedRelErr),
                  "Single Err > " + str(_toleratedRelErr), "Random Err > " + str(_toleratedRelErr),
                  "Cubic Err > " + str(_toleratedRelErr2), "Square Err > " + str(_toleratedRelErr2),
                  "Line Err > " + str(_toleratedRelErr2), "Single Err > " + str(_toleratedRelErr2),
                  "Random Err > " + str(_toleratedRelErr2), "Max Relative Error", "Min Rel Error",
                  "Average Rel Err", "zeroOut", "zeroGold"]


    #for relativeErrorParser
    _maxRelErr = 0
    _minRelErr = 0
    _avgRelErr = 0
    _zeroOut  = 0
    _zeroGold = 0
    _relErrLowerLimit = 0
    _relErrLowerLimit2 = 0

    # for benchmarks which have a third dimention this attribute must be set on the child process
    _hasThirdDimention = False

    def debugAttPrint(self):
        print "*******Var values*******"
        print "log file" , self._logFileName
        print "machine" , self._machine
        print "acc ite errors" , self._accIteErrors
        print "bencharmk" , self._benchmark
        print "header" , self._header
        print "sdcIterators" , self._sdcIteration
        print "iteErrors" , self._iteErrors
        print "size" , self._size
        print "dir name", self._dirName
        print "third dimention", self._hasThirdDimention

    @abstractmethod
    def getBenchmark(self):
        raise NotImplementedError

    def setDefaultValues(self, logFileName, machine, benchmark, header, sdcIteration, accIteErrors, iteErrors,  errList, logFileNameNoExt, pureHeader):
        self._logFileName = logFileName
        self._machine = machine
        self._benchmark = benchmark
        self._header = header
        self._sdcIteration = sdcIteration
        self._accIteErrors = accIteErrors

        self._iteErrors = iteErrors
        self._errList = errList
        # print "\n\nerr list inside " , len(errList)
        self._pureHeader = pureHeader
        self._logFileNameNoExt = logFileNameNoExt

        #self._size = \
        self.setSize(self._pureHeader)

        self._makeDirName()

        #----------------


    def getHasThirdDimention(self):
        return self._hasThirdDimention

    def setBuildImage(self, val):
        self.__buildImage = True



    """call to the private methods"""
    def parseErr(self):
        # self._errors["errorsParsed"] = []
        # self._errors["errListFiltered"] = []
        # self._errors["errListFiltered2"] = []
        for errString in self._errList:
            err = self.parseErrMethod(errString)
            if err != None:
                self. _errors["errorsParsed"].append(err)

    """for almost all benchmarks this method must be ovirride, because it is application dependent"""
    def relativeErrorParser(self):
        [self._maxRelErr, self._minRelErr, self._avgRelErr, self._zeroOut, self._zeroGold,
            self._relErrLowerLimit,
         self._errors["errListFiltered"], self._relErrLowerLimit2,
         self._errors["errListFiltered2"]] = self._relativeErrorParser(self._errors["errorsParsed"])

    @abstractmethod
    def parseErrMethod(self, errString):
        raise NotImplementedError()

    # @abstractmethod
    # def __relativeErrorParser(self, errList):
    #     raise NotImplementedError()

    """
        build image, based on object parameters
        #currObj.buildImage(errorsParsed, size,
        #                            currObj.dirName + '/' + currObj.header + '/' + currObj.logFileNameNoExt + '_' + str(imageIndex))
    """
    @abstractmethod
    def buildImageMethod(self):
        raise NotImplementedError()

    @abstractmethod
    def setSize(self, header):
        raise NotImplementedError()

    """if the csvHeader must be different, the variable must be set to the other value, so getCSVHeader will return other constant"""
    def getCSVHeader(self):
        return self.csvHeader



    """
    if you want other relative error parser this method must be override
    return [highest relative error, lowest relative error, average relative error, # zeros in the output, #zero in the GOLD, #errors with relative errors lower than limit(toleratedRelErr), list of errors limited by toleratedRelErr, #errors with relative errors lower than limit(toleratedRelErr2), list of errors limited by toleratedRelErr2]
    assumes errList[2] is read valued and errList[3] is expected value
    """
    def _relativeErrorParser(self, errList):
        relErr = []
        zeroGold = 0
        zeroOut = 0
        relErrLowerLimit = 0
        relErrLowerLimit2 = 0
        errListFiltered = []
        errListFiltered2 = []
        for err in errList:
            read = float(err[2])
            expected = float(err[3])
            absoluteErr = abs(expected - read)
            if abs(read) < 1e-6:
                zeroOut += 1
            if abs(expected) < 1e-6:
                zeroGold += 1
            else:
                relError = abs(absoluteErr / expected) * 100
                relErr.append(relError)
                if relError < self._toleratedRelErr:
                    relErrLowerLimit += 1
                else:
                    errListFiltered.append(err)
                if relError < self._toleratedRelErr2:
                    relErrLowerLimit2 += 1
                else:
                    errListFiltered2.append(err)
        if len(relErr) > 0:
            maxRelErr = max(relErr)
            minRelErr = min(relErr)
            avgRelErr = sum(relErr) / float(len(relErr))
            # print "\n\n" , relErrLowerLimit , " > 5 ", relErrLowerLimit2 , "\n\n"
            return [maxRelErr, minRelErr, avgRelErr, zeroOut, zeroGold, relErrLowerLimit, errListFiltered,
                    relErrLowerLimit2, errListFiltered2]
        else:
            return [None, None, None, zeroOut, zeroGold, relErrLowerLimit, errListFiltered, relErrLowerLimit2,
                    errListFiltered2]

            # fileNameSuffix = "errorFilterTo-"+str(toleratedRelErr) # add a suffix to csv filename

    def _buildImage(self, errors, size, filename):
        # identifica em qual posicao da matriz ocorreram os erros
        # definindo as bordas [esquerda, cabeca, direita, pe]
        err_limits = [int(size), int(size), 0, 0]
        for error in errors:
            if int(error[0]) < err_limits[0]:
                err_limits[0] = int(error[0])
            if int(error[0]) > err_limits[2]:
                err_limits[2] = int(error[0])
            if int(error[1]) < err_limits[1]:
                err_limits[1] = int(error[1])
            if int(error[1]) > err_limits[3]:
                err_limits[3] = int(error[1])

        # adiciona 5 pontos em cada lado para visualizacao facilitada
        # verifica que isso nao ultrapassa os limites da matriz
        err_limits[0] -= 5
        err_limits[1] -= 5
        err_limits[2] += 5
        err_limits[3] += 5
        if err_limits[0] < 0:
            err_limits[0] = 0
        if err_limits[1] < 0:
            err_limits[1] = 0
        if err_limits[2] > size:
            err_limits[2] = size
        if err_limits[3] > size:
            err_limits[3] = size

        # define uma imagem com o dobro do tamanho, para poder adicionar as guias
        # (o quadriculado)
        size_x = (err_limits[2] - err_limits[0]) * 2 + 1
        size_y = (err_limits[3] - err_limits[1]) * 2 + 1
        img = Image.new("RGB", (size_x, size_y), "white")

        n = 0

        # adiciona os erros a imagem
        for error in errors:
            n += 1
            try:
                if (n < 499):
                    img.putpixel(((int(error[0]) - err_limits[0]) * 2, (int(error[1]) - err_limits[1]) * 2),
                                 (255, 0, 0))
                else:
                    img.putpixel(((int(error[0]) - err_limits[0]) * 2, (int(error[1]) - err_limits[1]) * 2),
                                 (0, 0, 255))
            except IndexError:
                print ("Index error: ", error[0], ";", err_limits[0], ";", error[1], ";", err_limits[1])

        # adiciona as guias (quadriculado)
        if (size_x < 512) and (size_y < 512):
            for y in range(size_y):
                for x in range(size_x):
                    if (x % 2) == 1 or (y % 2) == 1:
                        img.putpixel((x, y), (240, 240, 240))

        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        img.save(filename + '.png')

        ################# => build_image()


    def jaccardCoefficient(self):
        # self._jaccardCoefficientDict["errorsParsed"] = 0
        # self._jaccardCoefficientDict["errListFiltered"] = 0
        # self._jaccardCoefficientDict["errListFiltered2"] = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for keys, values in self._errors.iteritems():
                #         jaccard = currObj.jaccardCoefficientLavaMD(errorsParsed)
                #         jaccardF = currObj.jaccardCoefficientLavaMD(errListFiltered)
                #         jaccardF2 = currObj.jaccardCoefficientLavaMD(errListFiltered2)
                # print "\n" , self._benchmark
                self._jaccardCoefficientDict[keys] = self._jaccardCoefficient(values)


    def _jaccardCoefficient(self, errListJaccard):
        expected = []
        read = []
        for err in errListJaccard:
            try:
                readGStr = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[2]))
                expectedGStr = ''.join(
                    bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[3]))
            except OverflowError:
                readGStr = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[2]))
                expectedGStr = ''.join(
                    bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[3]))

            read.extend([n for n in readGStr])
            expected.extend([n for n in expectedGStr])

        try:
            jac = jaccard_similarity_score(expected, read)
            dissimilarity = float(1.0 - jac)
            return dissimilarity
        except:
            return None


    """locality parser 2d, and 3d if it's avaliable"""
    # (square, colRow, single, random) = localityParser2D(errorsParsed)
    # (squareF, colRowF, singleF, randomF) = localityParser2D(errListFiltered)
    # (squareF2, colRowF2, singleF2, randomF2) = localityParser2D(errListFiltered2)
    # (cubic, square, colRow, single, random) = localityParser3D(errorsParsed)
    # (cubicF, squareF, colRowF, singleF, randomF) = localityParser3D(errListFiltered)
    # (cubicF2, squareF2, colRowF2, singleF2, randomF2) = localityParser3D(errListFiltered2)
    def localityParser(self):
        # self._locality["errorsParsed"] = [0, 0, 0, 0, 0]
        # self._locality["errListFiltered"] = [0, 0, 0, 0, 0]
        # self._locality["errListFiltered2"] = [0, 0, 0, 0, 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for key, value in self._errors.iteritems():
                if self._hasThirdDimention:
                    self._locality[key] = self._localityParser3D(value)
                else:
                    self._locality[key] = self._localityParser2D(value)

    # return [square, col/row, single, random]
    # assumes errList[0] is posX and errList[1] is posY
    def _localityParser2D(self, errList):
        if len(errList) < 1:
            return [0, 0, 0, 0, 0]
        elif len(errList) == 1:
            return [0, 0, 0, 1, 0]
        else:
            allXPositions = [x[0] for x in errList]  # Get all positions of X
            allYPositions = [x[1] for x in errList]  # Get all positions of Y
            counterXPositions = collections.Counter(allXPositions)  # Count how many times each value is in the list
            counterYPositions = collections.Counter(allYPositions)  # Count how many times each value is in the list
            rowError = any(
                x > 1 for x in counterXPositions.values())  # Check if any value is in the list more than one time
            colError = any(
                x > 1 for x in counterYPositions.values())  # Check if any value is in the list more than one time
            if rowError and colError:  # square error
                return [0, 1, 0, 0, 0]
            elif rowError or colError:  # row/col error
                return [0, 0, 1, 0, 0]
            else:  # random error
                return [0, 0, 0, 0, 1]

    # return [cubic, square, line, single, random]
    # assumes errList[0] is posX, errList[1] is posY, and errList[2] is posZ
    def _localityParser3D(self, errList):
        if len(errList) < 1:
            return [0, 0, 0, 0, 0]
        elif len(errList) == 1:
            return [0, 0, 0, 1, 0]
        else:
            allXPositions = [x[0] for x in errList]  # Get all positions of X
            allYPositions = [x[1] for x in errList]  # Get all positions of Y
            allZPositions = [x[2] for x in errList]  # Get all positions of Y
            counterXPositions = collections.Counter(allXPositions)  # Count how many times each value is in the list
            counterYPositions = collections.Counter(allYPositions)  # Count how many times each value is in the list
            counterZPositions = collections.Counter(allZPositions)  # Count how many times each value is in the list
            rowError = any(
                x > 1 for x in counterXPositions.values())  # Check if any value is in the list more than one time
            colError = any(
                x > 1 for x in counterYPositions.values())  # Check if any value is in the list more than one time
            heightError = any(
                x > 1 for x in counterZPositions.values())  # Check if any value is in the list more than one time
            if rowError and colError and heightError:  # cubic error
                return [1, 0, 0, 0, 0]
            if (rowError and colError) or (rowError and heightError) or (heightError and colError):  # square error
                return [0, 1, 0, 0, 0]
            elif rowError or colError or heightError:  # line error
                return [0, 0, 1, 0, 0]
            else:  # random error
                return [0, 0, 0, 0, 1]



    """CSV file operations"""
    """public method to write csv"""
    def writeToCSV(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            output = self._dirName + "/logs_parsed_" + self._machine + ".csv"
            self._writeToCSV(output)


    """
    write a list as a row to CSV
    if you want other type of write to csv,
    the method __writeToCSV and atribute __csvHeader must be changed
    """
    def _writeToCSV(self, csvFileName):
        if os.path.isfile(csvFileName) == False:
            self._writeCSVHeader(csvFileName)
            self._headerWriten = True

        try:

            csvWFP = open(csvFileName, "a")
            writer = csv.writer(csvWFP, delimiter=';')
            outputList = [self._logFileName,
                          self._machine,
                          self._benchmark,
                          self._header,
                          self._sdcIteration,
                          self._accIteErrors,
                          self._iteErrors,
                          self._relErrLowerLimit,
                          self._relErrLowerLimit2]

            # self.__jaccard,
            # self.__jaccardF,
            # self.__jaccardF2,
            for i in self.__keys:
                outputList.append(self._jaccardCoefficientDict[i])

            # self.__cubic,
            # self.__square,
            # self.__colRow,
            # self.__single,
            # self.__random,
            # self.__cubicF,
            # self.__squareF,
            # self.__colRowF,
            # self.__singleF,
            # self.__randomF,
            # self.__cubicF2,
            # self.__squareF2,
            # self.__colRowF2,
            # self.__singleF2,
            # self.__randomF2,
            for i in self.__keys:
                outputList.extend(self._locality[i])

            outputList.extend([self._maxRelErr,
                               self._minRelErr,

                               self._avgRelErr,
                               self._zeroOut,
                               self._zeroGold])

            writer.writerow(outputList)
            csvWFP.close()

        except:
            #ValueError.message += ValueError.message + "Error on writing row to " + str(csvFileName)
            print "Error on writing row to " + str(csvFileName)
            raise


    """writes a csv header, and create the log_parsed directory"""
    def _writeCSVHeader(self, csvFileName):
        if not os.path.isfile(csvFileName):
            if not os.path.exists(os.path.dirname(csvFileName)):
                try:
                    os.makedirs(os.path.dirname(csvFileName))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            csvWFP = open(csvFileName, "a")
            writer = csv.writer(csvWFP, delimiter=';')
            writer.writerow(self._csvHeader)
            csvWFP.close()


    def _makeDirName(self):
        self._dirName = os.getcwd() + "/" + self._machine + "/" + self._benchmark + "/" + str(self._size) + "/"
        if not os.path.exists(os.path.dirname(self._dirName)):
            try:
                os.makedirs(os.path.dirname(self._dirName))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def getImageIndex(self):
        return self._imageIndex

    def setImageIndex(self, imageIndex):
        self._imageIndex = imageIndex