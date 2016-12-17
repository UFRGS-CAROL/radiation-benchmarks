from abc import ABCMeta, abstractmethod
from PIL import Image
import struct
from sklearn.metrics import jaccard_similarity_score
import os
import errno
import collections

"""Base class for parser, need be implemented by each benchmark"""


class Parser(object):

    __metaclass__ = ABCMeta
    toleratedRelErr = 2  # minimum relative error to be considered, in percentage
    toleratedRelErr2 = 5  # minimum relative error to be considered, in percentage
    buildImages = False  # build locality images

    @abstractmethod
    def parseErr(self, errString):
        raise NotImplementedError()

    @abstractmethod
    def relativeErrorParser(self, errList):
        raise NotImplementedError()

    @abstractmethod
    def header(self):
        raise NotImplementedError()

    @abstractmethod
    def getLogHeader(self, header):
        raise NotImplementedError()

    # return [highest relative error, lowest relative error, average relative error, # zeros in the output, #zero in the GOLD, #errors with relative errors lower than limit(toleratedRelErr), list of errors limited by toleratedRelErr, #errors with relative errors lower than limit(toleratedRelErr2), list of errors limited by toleratedRelErr2]
    # assumes errList[2] is read valued and errList[3] is expected value
    def relativeErrorParser(self, errList):
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
                if relError < self.toleratedRelErr:
                    relErrLowerLimit += 1
                else:
                    errListFiltered.append(err)
                if relError < self.toleratedRelErr2:
                    relErrLowerLimit2 += 1
                else:
                    errListFiltered2.append(err)
        if len(relErr) > 0:
            maxRelErr = max(relErr)
            minRelErr = min(relErr)
            avgRelErr = sum(relErr) / float(len(relErr))
            return [maxRelErr, minRelErr, avgRelErr, zeroOut, zeroGold, relErrLowerLimit, errListFiltered,
                    relErrLowerLimit2, errListFiltered2]
        else:
            return [None, None, None, zeroOut, zeroGold, relErrLowerLimit, errListFiltered, relErrLowerLimit2,
                    errListFiltered2]

            # fileNameSuffix = "errorFilterTo-"+str(toleratedRelErr) # add a suffix to csv filename

    def build_image(self, errors, size, filename):
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

    def jaccardCoefficient(self, errListJaccard):
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

    # return [square, col/row, single, random]
    # assumes errList[0] is posX and errList[1] is posY
    def localityParser2D(self, errList):
        if len(errList) < 1:
            return [0, 0, 0, 0]
        elif len(errList) == 1:
            return [0, 0, 1, 0]
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
                return [1, 0, 0, 0]
            elif rowError or colError:  # row/col error
                return [0, 1, 0, 0]
            else:  # random error
                return [0, 0, 0, 1]

    # return [cubic, square, line, single, random]
    # assumes errList[0] is posX, errList[1] is posY, and errList[2] is posZ
    def localityParser3D(errList):
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
