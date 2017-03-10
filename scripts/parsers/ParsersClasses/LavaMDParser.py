import re
import struct
import sys

from Parser import Parser
from sklearn.metrics import jaccard_similarity_score


class LavaMDParser(Parser):

    def __init__(self, **kwargs):
        super(Parser, self).__init__(kwargs)


    _box = None
    _blockSize = None
    _streams = None
    _hasThirdDimention = True

    def getBenchmark(self):
        return self._benchmark

    def _jaccardCoefficient(self, errListJaccard):
        #print "\n\nPassou no jaccard lava \n\n"
        expected = []
        read = []
        for err in errListJaccard:
            try:
                readGStr = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[2]))
                expectedGStr = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[3]))
                readGStr2 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[4]))
                expectedGStr2 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[5]))
                readGStr3 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[6]))
                expectedGStr3 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[7]))
                readGStr4 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[8]))
                expectedGStr4 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[9]))
            except OverflowError:
                readGStr = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[2]))
                expectedGStr = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[3]))
                readGStr2 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[4]))
                expectedGStr2 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[5]))
                readGStr3 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[6]))
                expectedGStr3 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[7]))
                readGStr4 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[8]))
                expectedGStr4 = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[9]))

            read.extend([n for n in readGStr])
            read.extend([n for n in readGStr2])
            read.extend([n for n in readGStr3])
            read.extend([n for n in readGStr4])
            expected.extend([n for n in expectedGStr])
            expected.extend([n for n in expectedGStr2])
            expected.extend([n for n in expectedGStr3])
            expected.extend([n for n in expectedGStr4])

        try:
            jac = jaccard_similarity_score(expected, read)
            dissimilarity = float(1.0 - jac)
            return dissimilarity
        except:
            return None

    # return [highest relative error, lowest relative error, average relative error, # zeros in the output, #zero in the GOLD, #errors with relative errors lower than limit(toleratedRelErr), list of errors limited by toleratedRelErr, #errors with relative errors lower than limit(toleratedRelErr2), list of errors limited by toleratedRelErr2]
    def _relativeErrorParser(self, errList):
        #print "\n\n\nPassou no relavite lava\n\n\n"
        relErr = []
        zeroGold = 0
        zeroOut = 0
        relErrLowerLimit = 0
        relErrLowerLimit2 = 0
        errListFiltered = []
        errListFiltered2 = []
        for err in errList:
            vr = err[2]
            ve = err[3]
            xr = err[4]
            xe = err[5]
            yr = err[6]
            ye = err[7]
            zr = err[8]
            ze = err[9]
            absoluteErrV = abs(ve - vr)
            absoluteErrX = abs(xe - xr)
            absoluteErrY = abs(ye - yr)
            absoluteErrZ = abs(ze - zr)
            relErrorV = 0
            relErrorX = 0
            relErrorY = 0
            relErrorZ = 0
            if abs(vr) < 1e-6:
                zeroOut += 1
            if abs(xr) < 1e-6:
                zeroOut += 1
            if abs(yr) < 1e-6:
                zeroOut += 1
            if abs(zr) < 1e-6:
                zeroOut += 1
            if abs(ve) < 1e-6:
                zeroGold += 1
            else:
                relErrorV = abs(absoluteErrV / ve) * 100
            if abs(xe) < 1e-6:
                zeroGold += 1
            else:
                relErrorX = abs(absoluteErrX / xe) * 100
            if abs(ye) < 1e-6:
                zeroGold += 1
            else:
                relErrorY = abs(absoluteErrY / ye) * 100
            if abs(ze) < 1e-6:
                zeroGold += 1
            else:
                relErrorZ = abs(absoluteErrZ / ze) * 100

            relError = relErrorV + relErrorX + relErrorY + relErrorZ
            if relError > 0:
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
            return [maxRelErr, minRelErr, avgRelErr, zeroOut, zeroGold, relErrLowerLimit, errListFiltered,
                    relErrLowerLimit2, errListFiltered2]
        else:
            return [None, None, None, zeroOut, zeroGold, relErrLowerLimit, errListFiltered, relErrLowerLimit2,
                    errListFiltered2]

    # Return [posX, posY, posZ, vr, ve, xr, xe, yr, ye, zr, ze] -> [int, int, int, float, float, float, float, float, float, float, float]
    # Returns None if it is not possible to parse
    def parseErrMethod(self, errString):
        if self._box is None:
            print ("box is None!!!\nerrString: ", errString)
            print("header: ", self._header)
            sys.exit(1)
        try:
            ##ERR p: [357361], ea: 4, v_r: 1.5453305664062500e+03, v_e: 1.5455440673828125e+03, x_r: 9.4729260253906250e+02, x_e: 9.4630560302734375e+02, y_r: -8.0158099365234375e+02, y_e: -8.0218914794921875e+02, z_r: 9.8227819824218750e+02, z_e: 9.8161871337890625e+02
            m = re.match(
                ".*ERR.*\[(\d+)\].*v_r\: ([0-9e\+\-\.]+).*v_e\: ([0-9e\+\-\.]+).*x_r\: ([0-9e\+\-\.]+).*x_e\: ([0-9e\+\-\.]+).*y_r\: ([0-9e\+\-\.]+).*y_e\: ([0-9e\+\-\.]+).*z_r\: ([0-9e\+\-\.]+).*z_e\: ([0-9e\+\-\.]+)",
                errString)
            if m:
                pos = int(m.group(1))
                boxSquare = self._box * self._box
                posZ = int(pos / boxSquare)
                posY = int((pos - (posZ * boxSquare)) / self._box)
                posX = pos - (posZ * boxSquare) - (posY * self._box)

                vr = float(m.group(2))
                ve = float(m.group(3))
                xr = float(m.group(4))
                xe = float(m.group(5))
                yr = float(m.group(6))
                ye = float(m.group(7))
                zr = float(m.group(8))
                ze = float(m.group(9))
                return [posX, posY, posZ, vr, ve, xr, xe, yr, ye, zr, ze]
            else:
                return None
        except ValueError:
            return None


    def setSize(self, header):
        #old versions of lava
        m = re.match(".*size\:(\d+).*", header)
        if m:
            try:
                self._size = int(m.group(1))
            except:
                self._size = None

        self._box = None
        m = re.match(".*boxes[\:-](\d+).*", header)
        if m:
            try:
                self._box = int(m.group(1))
            except:
                self._box = None

        m = re.match(".*box[\:-](\d+).*", header)
        if m:
            try:
                self._box = int(m.group(1))
            except:
                self._box = None

        #new versions of lava
        #HEADER streams: 1 boxes:6 block_size:192
        m = re.match(".*streams\: (\d+).*boxes\:(\d+).*block_size\:(\d+).*", header)
        if m:
            self._streams = int(m.group(1))
            self._box = int(m.group(2))
            self._blockSize = int(m.group(3))

        if self._blockSize and self._streams:
            self._size = "streams_"+ str(self._streams) + "_boxes_" + str(self._box) + "_block_size_" + str(self._blockSize)
        else:
            self._size = "old_lava_boxes_" + str(self._box)

    def buildImageMethod(self): return False