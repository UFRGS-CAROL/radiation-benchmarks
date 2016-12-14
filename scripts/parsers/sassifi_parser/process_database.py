#!/usr/bin/env python

import sys
import os
import csv
import re
import collections
from PIL import Image
import struct
from sklearn.metrics import jaccard_similarity_score

import shelve
#benchmarks dict => (bechmarkname_machinename : list of SDC item)
#SDC item => [logfile name, header, sdc iteration, iteration total amount error, iteration accumulated error, list of errors ]
#list of errors => list of strings with all the error detail print in lines using #ERR

toleratedRelErr = 2 # minimum relative error to be considered, in percentage
toleratedRelErr2 = 5 # minimum relative error to be considered, in percentage
buildImages = False # build locality images
#fileNameSuffix = "errorFilterTo-"+str(toleratedRelErr) # add a suffix to csv filename


def build_image(errors, size, filename):
    # identifica em qual posicao da matriz ocorreram os erros
    # definindo as bordas [esquerda, cabeca, direita, pe]
    err_limits = [int(size), int(size), 0, 0]
    for error in errors:
        if int(error[0])<err_limits[0]:
            err_limits[0]=int(error[0])
        if int(error[0])>err_limits[2]:
            err_limits[2]=int(error[0])
        if int(error[1])<err_limits[1]:
            err_limits[1]=int(error[1])
        if int(error[1])>err_limits[3]:
            err_limits[3]=int(error[1])

    # adiciona 5 pontos em cada lado para visualizacao facilitada
    # verifica que isso nao ultrapassa os limites da matriz
    err_limits[0] -= 5
    err_limits[1] -= 5
    err_limits[2] += 5
    err_limits[3] += 5
    if err_limits[0]<0:
        err_limits[0]=0
    if err_limits[1]<0:
        err_limits[1]=0
    if err_limits[2]>size:
        err_limits[2]=size
    if err_limits[3]>size:
        err_limits[3]=size

    # define uma imagem com o dobro do tamanho, para poder adicionar as guias
    # (o quadriculado)
    size_x = (err_limits[2]-err_limits[0])*2+1
    size_y = (err_limits[3]-err_limits[1])*2+1
    img = Image.new("RGB", (size_x, size_y), "white")

    n = 0

    # adiciona os erros a imagem
    for error in errors:
        n+=1
        try:
            if (n<499):
                img.putpixel(((int(error[0])-err_limits[0])*2, (int(error[1])-err_limits[1])*2), (255, 0, 0))
            else:
                img.putpixel(((int(error[0])-err_limits[0])*2, (int(error[1])-err_limits[1])*2), (0, 0, 255))
        except IndexError:
            print ("Index error: ",error[0],";",err_limits[0],";",error[1],";",err_limits[1])

    # adiciona as guias (quadriculado)
    if (size_x<512) and (size_y<512):
        for y in range(size_y):
            for x in range(size_x):
                if (x%2)==1 or (y%2)==1:
                    img.putpixel((x, y), (240, 240, 240))


    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    img.save(filename+'.png')
################# => build_image()

def jaccardCoefficient(errListJaccard):
    expected = []
    read = []
    for err in errListJaccard:
        try:
            readGStr = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[2]))
            expectedGStr = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', err[3]))
        except OverflowError:
            readGStr = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[2]))
            expectedGStr = ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!d', err[3]))

        read.extend([n for n in readGStr])
        expected.extend([n for n in expectedGStr])

    try:
        jac = jaccard_similarity_score(expected,read)
        dissimilarity = float(1.0-jac)
        return dissimilarity
    except:
        return None

def jaccardCoefficientLavaMD(errListJaccard):
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
        jac = jaccard_similarity_score(expected,read)
        dissimilarity = float(1.0-jac)
        return dissimilarity
    except:
        return None

# return [highest relative error, lowest relative error, average relative error, # zeros in the output, #zero in the GOLD, #errors with relative errors lower than limit(toleratedRelErr), list of errors limited by toleratedRelErr, #errors with relative errors lower than limit(toleratedRelErr2), list of errors limited by toleratedRelErr2]
def relativeErrorParserLavaMD(errList):
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
            relErrorV = abs( absoluteErrV / ve ) * 100
        if abs(xe) < 1e-6:
            zeroGold += 1
        else:
            relErrorX = abs( absoluteErrX / xe ) * 100
        if abs(ye) < 1e-6:
            zeroGold += 1
        else:
            relErrorY = abs( absoluteErrY / ye ) * 100
        if abs(ze) < 1e-6:
            zeroGold += 1
        else:
            relErrorZ = abs( absoluteErrZ / ze ) * 100

        relError = relErrorV + relErrorX + relErrorY + relErrorZ
        if relError > 0:
            relErr.append( relError )
            if relError < toleratedRelErr:
                relErrLowerLimit += 1
            else:
                errListFiltered.append(err)
            if relError < toleratedRelErr2:
                relErrLowerLimit2 += 1
            else:
                errListFiltered2.append(err)
    if len(relErr) > 0:
        maxRelErr = max(relErr)
        minRelErr = min(relErr)
        avgRelErr = sum(relErr)/float(len(relErr))
        return[maxRelErr,minRelErr,avgRelErr,zeroOut,zeroGold,relErrLowerLimit,errListFiltered,relErrLowerLimit2,errListFiltered2]
    else:
        return[None,None,None,zeroOut,zeroGold,relErrLowerLimit,errListFiltered,relErrLowerLimit2,errListFiltered2]

def relativeErrorParserLulesh(errList):
    relErr = []
    zeroGold = 0
    zeroOut = 0
    relErrLowerLimit = 0
    relErrLowerLimit2 = 0
    errListFiltered = []
    errListFiltered2 = []

    for err in errList:
        #[posX, posY, posZ, None, None, xr, xe, yr, ye, zr, ze]

        xr = err[5]
        xe = err[6]
        yr = err[7]
        ye = err[8]
        zr = err[9]
        ze = err[10]
       # print xr,xe,yr,ye,zr,ze
       # print err
        #absoluteErrV = abs(ve - vr)
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
            relErrorX = abs( absoluteErrX / xe ) * 100
        if abs(ye) < 1e-6:
            zeroGold += 1
        else:
            relErrorY = abs( absoluteErrY / ye ) * 100
        if abs(ze) < 1e-6:
            zeroGold += 1
        else:
            relErrorZ = abs( absoluteErrZ / ze ) * 100

        relError = relErrorX + relErrorY + relErrorZ # relErrorV +
        if relError > 0:
            relErr.append( relError )
            if relError < toleratedRelErr:
                relErrLowerLimit += 1
            else:
                errListFiltered.append(err)
            if relError < toleratedRelErr2:
                relErrLowerLimit2 += 1
            else:
                errListFiltered2.append(err)
    if len(relErr) > 0:
        maxRelErr = max(relErr)
        minRelErr = min(relErr)
        avgRelErr = sum(relErr)/float(len(relErr))
        return[maxRelErr,minRelErr,avgRelErr,zeroOut,zeroGold,relErrLowerLimit,errListFiltered,relErrLowerLimit2,errListFiltered2]
    else:
        return[None,None,None,zeroOut,zeroGold,relErrLowerLimit,errListFiltered,relErrLowerLimit2,errListFiltered2]
# return [highest relative error, lowest relative error, average relative error, # zeros in the output, #zero in the GOLD, #errors with relative errors lower than limit(toleratedRelErr), list of errors limited by toleratedRelErr, #errors with relative errors lower than limit(toleratedRelErr2), list of errors limited by toleratedRelErr2]
# assumes errList[2] is read valued and errList[3] is expected value
def relativeErrorParser(errList):
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
            relError = abs( absoluteErr / expected ) * 100
            relErr.append( relError )
            if relError < toleratedRelErr:
                relErrLowerLimit += 1
            else:
                errListFiltered.append(err)
            if relError < toleratedRelErr2:
                relErrLowerLimit2 += 1
            else:
                errListFiltered2.append(err)
    if len(relErr) > 0:
        maxRelErr = max(relErr)
        minRelErr = min(relErr)
        avgRelErr = sum(relErr)/float(len(relErr))
        return[maxRelErr,minRelErr,avgRelErr,zeroOut,zeroGold,relErrLowerLimit,errListFiltered,relErrLowerLimit2,errListFiltered2]
    else:
        return[None,None,None,zeroOut,zeroGold,relErrLowerLimit,errListFiltered,relErrLowerLimit2,errListFiltered2]

# return [square, col/row, single, random]
# assumes errList[0] is posX and errList[1] is posY
def localityParser2D(errList):
    if len(errList) < 1:
        return [0,0,0,0]
    elif len(errList) == 1:
        return [0,0,1,0]
    else:
        allXPositions = [x[0] for x in errList] # Get all positions of X
        allYPositions = [x[1] for x in errList] # Get all positions of Y
        counterXPositions=collections.Counter(allXPositions) # Count how many times each value is in the list
        counterYPositions=collections.Counter(allYPositions) # Count how many times each value is in the list
        rowError = any(x>1 for x in counterXPositions.values()) # Check if any value is in the list more than one time
        colError = any(x>1 for x in counterYPositions.values()) # Check if any value is in the list more than one time
        if rowError and colError: # square error
            return [1,0,0,0]
        elif rowError or colError: # row/col error
            return [0,1,0,0]
        else: # random error
            return [0,0,0,1]

# return [cubic, square, line, single, random]
# assumes errList[0] is posX, errList[1] is posY, and errList[2] is posZ
def localityParser3D(errList):
    if len(errList) < 1:
        return [0,0,0,0,0]
    elif len(errList) == 1:
        return [0,0,0,1,0]
    else:
        allXPositions = [x[0] for x in errList] # Get all positions of X
        allYPositions = [x[1] for x in errList] # Get all positions of Y
        allZPositions = [x[2] for x in errList] # Get all positions of Y
        counterXPositions=collections.Counter(allXPositions) # Count how many times each value is in the list
        counterYPositions=collections.Counter(allYPositions) # Count how many times each value is in the list
        counterZPositions=collections.Counter(allZPositions) # Count how many times each value is in the list
        rowError = any(x>1 for x in counterXPositions.values()) # Check if any value is in the list more than one time
        colError = any(x>1 for x in counterYPositions.values()) # Check if any value is in the list more than one time
        heightError = any(x>1 for x in counterZPositions.values()) # Check if any value is in the list more than one time
        if rowError and colError and heightError: #cubic error
            return [1,0,0,0,0]
        if (rowError and colError) or (rowError and heightError) or (heightError and colError): # square error
            return [0,1,0,0,0]
        elif rowError or colError or heightError: # line error
            return [0,0,1,0,0]
        else: # random error
            return [0,0,0,0,1]

# Return [posX, posY, posZ, vr, ve, xr, xe, yr, ye, zr, ze] -> [int, int, int, float, float, float, float, float, float, float, float]
# Returns None if it is not possible to parse
def parseErrLavaMD(errString, box, header):
    if box is None:
        print ("box is None!!!\nerrString: ",errString)
        print("header: ",header)
        sys.exit(1)
    try:
        ##ERR p: [357361], ea: 4, v_r: 1.5453305664062500e+03, v_e: 1.5455440673828125e+03, x_r: 9.4729260253906250e+02, x_e: 9.4630560302734375e+02, y_r: -8.0158099365234375e+02, y_e: -8.0218914794921875e+02, z_r: 9.8227819824218750e+02, z_e: 9.8161871337890625e+02
        m = re.match(".*ERR.*\[(\d+)\].*v_r\: ([0-9e\+\-\.]+).*v_e\: ([0-9e\+\-\.]+).*x_r\: ([0-9e\+\-\.]+).*x_e\: ([0-9e\+\-\.]+).*y_r\: ([0-9e\+\-\.]+).*y_e\: ([0-9e\+\-\.]+).*z_r\: ([0-9e\+\-\.]+).*z_e\: ([0-9e\+\-\.]+)", errString)
        if m:
            pos = int(m.group(1))
            boxSquare = box * box
            posZ = int( pos / boxSquare )
            posY = int( (pos-(posZ*boxSquare)) / box )
            posX = pos-(posZ*boxSquare)-(posY*box)

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

# Return [posX, posY, read, expected] -> [int, int, float, float]
# Returns None if it is not possible to parse
def parseErrGEMM(errString):
    try:
        #ERR stream: 0, p: [0, 0], r: 3.0815771484375000e+02, e: 0.0000000000000000e+00
        if 'nan' in errString:
            m = re.match(".*ERR.*\[(\d+)..(\d+)\].*r\:.*nan.*e\: ([0-9e\+\-\.]+)", errString)
            if m:
                posX = int(m.group(1))
                posY = int(m.group(2))
                read = float('nan')
                expected = float(m.group(3))
                return [posX, posY, read, expected]
            else:
                return None
        else:
            m = re.match(".*ERR.*\[(\d+)..(\d+)\].*r\: ([0-9e\+\-\.]+).*e\: ([0-9e\+\-\.]+)", errString)
            if m:
                posX = int(m.group(1))
                posY = int(m.group(2))
                read = float(m.group(3))
                expected = float(m.group(4))
                return [posX, posY, read, expected]
            else:
                return None


    except ValueError:
        return None

#Return [posX, posY, read, expected, comp_or_spans] -> [int, int, float, float, string]
#Returns None if it is not possible to parse
def parseErrACCL(errString):
    try:
        #ERR stream: 0, p: [0, 0], r: 3.0815771484375000e+02, e: 0.0000000000000000e+00
        #ERR t: [components], p: [256][83], r: 90370, e: 80131
        ##ERR t: [components], p: [770][148], r: 80130, e: -1
        m = re.match(".*ERR.*p\: \[(\d+)\]\[(\d+)\].*r\: ([(\d+)]+).*e\: ([(\d+\-)]+)", errString)
        if m:
            posX = int(m.group(1))
            posY = int(m.group(2))
            read = int(m.group(3))
            expected = int(m.group(4))
            #print [posX, posY, read, expected]
            return [posX, posY, read, expected]
        else:
            m = re.match(".*ERR.*p\: \[(\d+)\].*r\: ([(\d+\-)]+).*e\: ([(\d+\-)]+)", errString)
            if m:
                pos = int(m.group(1))
                #posY = int(m.group(2))
                read = int(m.group(2))
                expected = int(m.group(3))
                i = 0
                j = 0
                done = False

                if  'spans' in errString:
                    while (i < 512):
                        j = 0
                        while (j < (4096 * 2)):
                            if ( (i * j) >= pos ):
                                #print (i * j) , pos , errString
                                done = True
                                break
                            j += 1
                        if done:
                            break
                        i += 1


                if 'components' in errString:
                    while (i < 512):
                        j = 0
                        while (j < 4096):
                            if ( (i * j) >= pos ):
                                #print (i * j) , pos , errString
                                done = True
                                break
                            j += 1

                        if done:
                            break
                        i += 1



                posX = i
                posY = j

                return [posX, posY, read, expected]
            else:
                return None
    except ValueError:
        return None


# Return [posX, posY, read, expected] -> [int, int, float, float]
# Returns None if it is not possible to parse
def parseErrNW(errString):
    try:
        #ERR  p: [1, 467], r: -4654, e: 21, error: 467
        m = re.match(".*ERR.*\[(\d+)..(\d+)\].*r\: ([0-9\+\-]+).*e\: ([0-9\+\-]+).*", errString)
        #print errString
        #print m.group(1) , m.group(2) , m.group(3) , m.group(4)
        if m:
            posX = int(m.group(1))
            posY = int(m.group(2))
            read = int(m.group(3))
            expected = int(m.group(4))
            #print m.group(1) , m.group(2) , m.group(3) , m.group(4)
            return [posX, posY, read, expected]
        else:
            return None
    except ValueError:
        return None

# Return [posX, posY, read, expected] -> [int, int, float, float]
# Returns None if it is not possible to parse
def parseErrMergesort(errString):
    try:
        #ERR stream: 0, p: [0, 0], r: 3.0815771484375000e+02, e: 0.0000000000000000e+00
        m = re.match(".*ERR.*\[(\d+)..(\d+)\].*r\: ([0-9e\+\-\.]+).*e\: ([0-9e\+\-\.]+)", errString)
        if m:
            posX = int(m.group(1))
            posY = int(m.group(2))
            read = float(m.group(3))
            expected = float(m.group(4))
            return [posX, posY, read, expected]
        else:
            return None
    except ValueError:
        return None

# Return [posX, posY, read, expected] -> [int, int, float, float]
# Returns None if it is not possible to parse
def parseErrQuicksort(errString):
    try:
        #ERR stream: 0, p: [0, 0], r: 3.0815771484375000e+02, e: 0.0000000000000000e+00
        m = re.match(".*ERR.*\[(\d+)..(\d+)\].*r\: ([0-9e\+\-\.]+).*e\: ([0-9e\+\-\.]+)", errString)
        if m:
            posX = int(m.group(1))
            posY = int(m.group(2))
            read = float(m.group(3))
            expected = float(m.group(4))
            return [posX, posY, read, expected]
        else:
            return None
    except ValueError:
        return None

# Return [posX, posY, read, expected] -> [int, int, float, float]
# Return [posX, posY, expected*2, read] if read is NaN
# Returns None if it is not possible to parse
def parseErrHotspot(errString):
   # print "Passou"
    try:
        m = re.match(".*ERR.*.*r\:([0-9e\+\-\.nan]+).*e\:([0-9e\+\-\.]+).*\[(\d+),(\d+)\]", errString)
        #OCL -> ERR r:293.943054 e:293.943024 [165,154]
        if m:
            posX=int(m.group(3))
            posY=int(m.group(4))
            read=float(m.group(1))
            expected=float(m.group(2))
            if re.match(".*nan.*",read):
                return [posX, posY, expected*2, expected]
            else:
                read = float(read)
                return [posX, posY, read, expected]

        m = re.match(".*ERR.*\[(\d+)..(\d+)\].*r\: ([0-9e\+\-\.nan]+).*e\: ([0-9e\+\-\.]+)", errString)
        #CUDA -> ERR stream: 0, p: [0, 0], r: 3.0815771484375000e+02, e: 0.0000000000000000e+00
        if m:
            print m.group(1) + " " + m.group(2) + " " +m.group(3) + " " +m.group(4) + " "
            posX=int(m.group(1))
            posY=int(m.group(2))
            read=m.group(3)
            expected=float(m.group(4))
            if re.match(".*nan.*",read):
                return [posX, posY, expected*2, expected]
            else:
                read = float(read)
                return [posX, posY, read, expected]
        return None
    except Exception as e:
        return None
#parse lulesh is the same as lava
# Return [posX, posY, posZ, vr, ve, xr, xe, yr, ye, zr, ze] -> [int, int, int, float, float, float, float, float, float, float, float]
# Returns None if it is not possible to parse
def parseErrLulesh(errString, box, header):
    if box is None:
        print ("box is None!!!\nerrString: ",errString)
        print("header: ",header)
        sys.exit(1)
    try:
        ##ERR p: [58978] x_gold:4.950000000000000e-01 x_output:4.949996815262007e-01 y_gold:7.650000000000000e-01 y_output:7.649996815262007e-01 z_gold:4.950000000000000e-01 z_output:4.949996815262007e-01
        m = re.match(".*ERR.*\[(\d+)\].*x_gold\:([0-9e\+\-\.]+).*x_output\:([0-9e\+\-\.]+).*y_gold\:([0-9e\+\-\.]+).*y_output\:([0-9e\+\-\.]+).*z_gold\:([0-9e\+\-\.]+).*z_output\:([0-9e\+\-\.]+).*", errString)
        if m:
            pos = int(m.group(1))
            boxSquare = box * box
            posZ = int( pos / boxSquare )
            posY = pos if int( (pos-(posZ*boxSquare)) / box ) == 0 else int( (pos-(posZ*boxSquare)) / box )

            posX = pos #if (pos-(posZ*boxSquare)-(posY*box)) == 0 else ((pos-(posZ*boxSquare)) / box)

            xe = float(m.group(2))
            xr = float(m.group(3))
            ye = float(m.group(4))
            yr = float(m.group(5))
            ze = float(m.group(6))
            zr = float(m.group(7))
            # [posX, posY, posZ, vr, ve, xr, xe, yr, ye, zr, ze]
            #print [posX, posY, posZ, xr, xe, yr, ye, zr, ze]
            return [posX, posY, posZ, None, None, xr, xe, yr, ye, zr, ze]
        else:
            return None
    except ValueError:
        return None
def parseErrors(benchmarkname_machinename, sdcItemList):
    benchmark = benchmarkname_machinename
    machine = benchmarkname_machinename
    m = re.match("(.*)_(.*)", benchmarkname_machinename)
    if m:
        benchmark = m.group(1)
        machine = m.group(2)

    dirName = "./"+machine+"/"+benchmark
    if not os.path.exists(os.path.dirname(dirName)):
        try:
            os.makedirs(os.path.dirname(dirName))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    sdci = 1
    total_sdcs = len(sdcItemList)
    imageIndex = 0
    for sdcItem in sdcItemList:
        progress = "{0:.2f}".format(sdci/total_sdcs * 100)
        sys.stdout.write("\rProcessing SDC "+str(sdci)+" of "+str(total_sdcs)+" - "+progress+"%")
        sys.stdout.flush()

        logFileName = sdcItem[0]
        header = sdcItem[1]
        header = re.sub(r"[^\w\s]", '-', header) #keep only numbers and digits
        sdcIteration = sdcItem[2]
        iteErrors = sdcItem[3]
        accIteErrors = sdcItem[4]
        errList = sdcItem[5]

        logFileNameNoExt = logFileName
        m = re.match("(.*).log", logFileName)

        #for each header of each benchmark
        if m:
            logFileNameNoExt = m.group(1)

        size = None
        m = re.match(".*size\:(\d+).*", header)
        if m:
            try:
                size = int(m.group(1))
            except:
                size = None
        box = None
        m = re.match(".*boxes[\:-](\d+).*", header)
        if m:
            try:
                box = int(m.group(1))
            except:
                box = None
        m = re.match(".*box[\:-](\d+).*", header)
        if m:
            try:
                box = int(m.group(1))
            except:
                box = None
        #for accl
        m = re.match(".*frames[\:-](\d+).*",header)
        frames = None
        framesPerStream = None
        if m:
            try:
                frames = int(m.group(1))
                m = re.match(".*framesPerStream[\:-](\d+).*",header)
                framesPerStream = int(m.group(1))
            except:
                frames = None
                framesPerStrem = None
        max_rows = None
        max_cols = None
        penalty  = None
        #for nw
        m = re.match(".*max_rows\: (\d+).*max_cols\: (\d+).*penalty\: (\d+).*", header)
        if m:
            try:
                max_rows = int(m.group(1))
                max_cols = int(m.group(2))
                penalty  = int(m.group(3))
            except:
                max_rows = None
                max_cols = None
                penalty  = None
        else: #for old logs
            m = re.match(".*size\:(\d+).*(\d+).*", header)
            if m:
                max_rows = int(m.group(1))
                max_cols = int(m.group(2))
                penalty  = None

        #for lulesh
        #structured:YES size:50 iterations:50
        m = re.match("structured:YES.*size\:(\d+).*iterations:(\d+)",header)
        size = None
        iterations = None
        if m:
            try:
                size = int (m.group(1))
                iterations = int(m.group(2))
            except:
                size = None
                iterations = None

        #for lud
        m = re.match(".*matrix_size\:(\d+).*reps\:(\d+)", header)
        if m:
            try:
                m_size = int(m.group(1))
                size = int(m.group(2))
            except:
                m_size = None
                size = None

        isHotspot = re.search("hotspot",benchmark,flags=re.IGNORECASE)
        isGEMM = re.search("GEMM",benchmark,flags=re.IGNORECASE)
        isLavaMD = re.search("lavamd",benchmark,flags=re.IGNORECASE)
        isCLAMR = re.search("clamr",benchmark,flags=re.IGNORECASE)
        #algoritmos ACCL, NW, Lulesh, Mergesort e Quicksort
        isACCL = re.search("accl", k, flags=re.IGNORECASE)
        isNW   = re.search("nw", k, flags=re.IGNORECASE)
        isLulesh = re.search("lulesh",k , flags=re.IGNORECASE)
        isLud = re.search("lud", k, flags=re.IGNORECASE)
        #isMergesort = re.search("mergesort", k, flags=re.IGNORECASE)
        #isQuicksort = re.search("quicksort", k, flags=re.IGNORECASE)

        if isLavaMD and box is None:
            continue
        errorsParsed = []
        # Get error details from log string
        for errString in errList:
            err = None
            if isGEMM or isLud:
                err = parseErrGEMM(errString)
            elif isHotspot:

                err = parseErrHotspot(errString)
            elif isLavaMD:
                err = parseErrLavaMD(errString, box, header)
            elif isACCL:
                err = parseErrACCL(errString)
            elif isNW:
                err = parseErrNW(errString)
            elif isLulesh:
                err = parseErrLulesh(errString,50, header)
            #elif isMergesort:
            #   err = parseErrMergesort(errString)
            #elif isQuicksort:
            #   err = parseErrQuicksort(errString)
            if err is not None:
                errorsParsed.append(err)

        # Parse relative error
        if isGEMM or isHotspot or isACCL or isNW or isLud:
            (maxRelErr, minRelErr, avgRelErr, zeroOut, zeroGold, relErrLowerLimit, errListFiltered, relErrLowerLimit2, errListFiltered2) = relativeErrorParser(errorsParsed)
        elif isLavaMD:
            (maxRelErr, minRelErr, avgRelErr, zeroOut, zeroGold, relErrLowerLimit, errListFiltered, relErrLowerLimit2, errListFiltered2) = relativeErrorParserLavaMD(errorsParsed)
        elif isLulesh:
            (maxRelErr, minRelErr, avgRelErr, zeroOut, zeroGold, relErrLowerLimit, errListFiltered, relErrLowerLimit2, errListFiltered2) = relativeErrorParserLulesh(errorsParsed)

        # Parse locality metric
        if isGEMM or isHotspot or isACCL or isNW or isLud:
            #if isHotspot:
            #    print errListFiltered
            #    print errorsParsed
            (square,colRow,single,random) = localityParser2D(errorsParsed)
            (squareF,colRowF,singleF,randomF) = localityParser2D(errListFiltered)
            (squareF2,colRowF2,singleF2,randomF2) = localityParser2D(errListFiltered2)
            jaccard = jaccardCoefficient(errorsParsed)
            jaccardF = jaccardCoefficient(errListFiltered)
            jaccardF2 = jaccardCoefficient(errListFiltered2)
            errListFiltered = []
            errListFiltered2 = []
            cubic = 0
            cubicF = 0
            cubicF2 = 0

            if single == 0 and buildImages:
                if size is not None:
                    build_image(errorsParsed, size, dirName+'/'+header+'/'+logFileNameNoExt+'_'+str(imageIndex))
                else:
                    build_image(errorsParsed, 8192, dirName+'/'+header+'/'+logFileNameNoExt+'_'+str(imageIndex))
                imageIndex += 1
        elif isLavaMD or isLulesh:
            (cubic,square,colRow,single,random) = localityParser3D(errorsParsed)
            (cubicF,squareF,colRowF,singleF,randomF) = localityParser3D(errListFiltered)
            (cubicF2,squareF2,colRowF2,singleF2,randomF2) = localityParser3D(errListFiltered2)
            if isLavaMD:
                jaccard = jaccardCoefficientLavaMD(errorsParsed)
                jaccardF = jaccardCoefficientLavaMD(errListFiltered)
                jaccardF2 = jaccardCoefficientLavaMD(errListFiltered2)
            else:
                jaccard = jaccardF = jaccardF2 = 0
            errListFiltered = []
            errListFiltered2 = []

        else: # Need to add locality parser for other benchmarks, if possible!
            (cubic,square,colRow,single,random) = [0,0,0,0,0]
            (cubicF,squareF,colRowF,singleF,randomF) = [0,0,0,0,0]
            (cubicF2,squareF2,colRowF2,singleF2,randomF2) = [0,0,0,0,0]
            jaccard = None
            jaccardF = None
            jaccardF2 = None


        # Write info to csv file
        #if fileNameSuffix is not None and fileNameSuffix != "":
        #   csvFileName = dirName+'/'+header+'/logs_parsed_'+machine+'_'+fileNameSuffix+'.csv'
        #else:
        csvFileName = dirName+'/'+header+'/logs_parsed_'+machine+'.csv'
        if not os.path.exists(os.path.dirname(csvFileName)):
            try:
                os.makedirs(os.path.dirname(csvFileName))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        flag=0
        if not os.path.exists(csvFileName):
            flag=1
        csvWFP = open(csvFileName, "a")
        writer = csv.writer(csvWFP, delimiter=';')
        if flag==1: # csv header
            writer.writerow(["logFileName","Machine","Benchmark","Header","SDC Iteration","#Accumulated Errors","#Iteration Errors","Relative Errors <= "+str(toleratedRelErr)+"%","Relative Errors <= "+str(toleratedRelErr2)+"%", "Jaccard", "Jaccard > "+str(toleratedRelErr)+"%","Jaccard > "+str(toleratedRelErr2)+"%","Cubic","Square","Line","Single","Random","Cubic Err > "+str(toleratedRelErr),"Square Err > "+str(toleratedRelErr),"Line Err > "+str(toleratedRelErr),"Single Err > "+str(toleratedRelErr),"Random Err > "+str(toleratedRelErr),"Cubic Err > "+str(toleratedRelErr2),"Square Err > "+str(toleratedRelErr2),"Line Err > "+str(toleratedRelErr2),"Single Err > "+str(toleratedRelErr2),"Random Err > "+str(toleratedRelErr2),"Max Relative Error", "Min Rel Error","Average Rel Err","# zero output", "# zeor Gold"])
        writer.writerow([logFileName,machine,benchmark,header,sdcIteration,accIteErrors,iteErrors,relErrLowerLimit,relErrLowerLimit2,jaccard,jaccardF,jaccardF2,cubic,square,colRow,single,random,cubicF,squareF,colRowF,singleF,randomF,cubicF2,squareF2,colRowF2,singleF2,randomF2,maxRelErr,minRelErr,avgRelErr,zeroOut,zeroGold])
        csvWFP.close()
        sdci += 1

    sys.stdout.write("\rProcessing SDC "+str(sdci-1)+" of "+str(total_sdcs)+" - 100%                     "+"\n")
    sys.stdout.flush()
################ => parseErrors()


###########################################
# MAIN
###########################################
#db = shelve.open("errors_log_database") #python3
db = shelve.open("errors_log_database") #python2

#for k, v in db.items(): #python3
for k, v in db.iteritems(): #python2
    isHotspot = re.search("Hotspot",k,flags=re.IGNORECASE)
    isGEMM = re.search("GEMM",k,flags=re.IGNORECASE)
    isLavaMD = re.search("lavamd",k,flags=re.IGNORECASE)
    isCLAMR = re.search("clamr",k,flags=re.IGNORECASE)
    #algoritmos ACCL, NW, Lulesh, Mergesort e Quicksort
    isACCL = re.search("accl", k, flags=re.IGNORECASE)
    isNW = re.search("nw", k, flags=re.IGNORECASE)
    isLulesh = re.search("lulesh", k, flags=re.IGNORECASE)
    isLud = re.search("lud", k, flags=re.IGNORECASE)

    if isHotspot or isGEMM or isLavaMD or isACCL or isNW or isLulesh or isLud:
        print("Processing ",k)
        parseErrors(k,v)
    else:
        print("Ignoring ",k)

db.close()
