import re
import sys

from Parser import Parser


class LuleshParser(Parser):


    def __init__(self, **kwargs):
        super(Parser, self).__init__(kwargs)

    _box = 50
    _hasThirdDimention = True
    _iterations = None

    def _jaccardCoefficient(self, errListJaccard):
        return 0

    def _relativeErrorParser(self, errList):
        relErr = []
        zeroGold = 0
        zeroOut = 0
        relErrLowerLimit = 0
        relErrLowerLimit2 = 0
        errListFiltered = []
        errListFiltered2 = []

        for err in errList:
            # [posX, posY, posZ, None, None, xr, xe, yr, ye, zr, ze]

            xr = err[5]
            xe = err[6]
            yr = err[7]
            ye = err[8]
            zr = err[9]
            ze = err[10]
            # print xr,xe,yr,ye,zr,ze
            # print err
            # absoluteErrV = abs(ve - vr)
            absoluteErrX = abs(xe - xr)
            absoluteErrY = abs(ye - yr)
            absoluteErrZ = abs(ze - zr)
            relErrorV = 0
            relErrorX = 0
            relErrorY = 0
            relErrorZ = 0
            # if abs(vr) < 1e-6:
            #    zeroOut += 1
            if abs(xr) < 1e-6:
                zeroOut += 1
            if abs(yr) < 1e-6:
                zeroOut += 1
            if abs(zr) < 1e-6:
                zeroOut += 1
                #  if abs(ve) < 1e-6:
                #     zeroGold += 1
                # else:
                #     relErrorV = abs( absoluteErrV / ve ) * 100
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

            relError = relErrorX + relErrorY + relErrorZ  # relErrorV +
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

    # parse lulesh is the same as lava
    # Return [posX, posY, posZ, vr, ve, xr, xe, yr, ye, zr, ze] -> [int, int, int, float, float, float, float, float, float, float, float]
    # Returns None if it is not possible to parse
    def parseErrMethod(self, errString):
        if self._box is None:
            print ("box is None!!!\nerrString: ", errString)
            print("header: ", self._header)
            sys.exit(1)
        try:
            ##ERR p: [58978] x_gold:4.950000000000000e-01 x_output:4.949996815262007e-01 y_gold:7.650000000000000e-01 y_output:7.649996815262007e-01 z_gold:4.950000000000000e-01 z_output:4.949996815262007e-01
            m = re.match(
                '.*ERR.*\[(\d+)\].*x_gold\:([0-9e\+\-\.]+).*x_output\:([0-9e\+\-\.]+).*y_gold\:([0-9e\+\-\.]+).*y_output\:([0-9e\+\-\.]+).*z_gold\:([0-9e\+\-\.]+).*z_output\:([0-9e\+\-\.]+).*',
                errString)
            if m:
                pos = int(m.group(1))
                boxSquare = self._box * self._box
                posZ = int(pos / boxSquare)
                posY = pos if int((pos - (posZ * boxSquare)) / self._box) == 0 else int((pos - (posZ * boxSquare)) / self._box)

                posX = pos  # if (pos-(posZ*boxSquare)-(posY*box)) == 0 else ((pos-(posZ*boxSquare)) / box)

                xe = float(m.group(2))
                xr = float(m.group(3))
                ye = float(m.group(4))
                yr = float(m.group(5))
                ze = float(m.group(6))
                zr = float(m.group(7))
                # [posX, posY, posZ, vr, ve, xr, xe, yr, ye, zr, ze]
                # print [posX, posY, posZ, xr, xe, yr, ye, zr, ze]
                return [posX, posY, posZ, None, None, xr, xe, yr, ye, zr, ze]
            else:
                return None
        except ValueError:
            return None


    def setSize(self, header):
        #for lulesh
        #structured:YES size:50 iterations:50
        print "\n", header
        m = re.match(".*structured:YES.*size\:(\d+).*iterations:(\d+).*box:(\d+).*",header)
        if m:
            self._size = None
            self._iterations = None
            try:
                self._size = int (m.group(1))
                self._iterations = int(m.group(2))
                self._box = int(m.group(3))
            except:
                self._size = None
                self._iterations = None
        else:
            m = re.match(".*structured:YES.*size\:(\d+).*iterations:(\d+).*",header)
            self._size = None
            self._iterations = None
            if m:
                try:
                    self._size = int (m.group(1))
                    self._iterations = int(m.group(2))
                except:
                    self._size = None
                    self._iterations = None
        self._size = str(self._size) + str(self._iterations)
        # return self._size

    def getBenchmark(self):
        return self._benchmark

    def buildImageMethod(self):
        return False

