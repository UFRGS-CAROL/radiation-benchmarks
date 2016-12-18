#!/usr/bin/env python
import sys
GOLD_DIR = "/home/fernando/Dropbox/UFRGS/Pesquisa/LANSCE_2016_PARSED/Gold_CNNs/"
import os
import csv
import re
# import pr_parsers.parse_neural_networks as pn
import shelve
import errno
import argparse
import SupportClasses as sp


#temporary set
buildImages = False
size = 0

# benchmarks dict => (bechmarkname_machinename : list of SDC item)
# SDC item => [logfile name, header, sdc iteration, iteration total amount error, iteration accumulated error, list of errors ]
# list of errors => list of strings with all the error detail print in lines using #ERR
def parseErrors(benchmarkname_machinename, sdcItemList, gold_dir):
    # matchBench = MatchBenchmark(sdcItemList, benchmarkname_machinename)
    # benchmark = benchmarkname_machinename
    # machine = benchmarkname_machinename

    sdci = 1
    total_sdcs = len(sdcItemList)
    imageIndex = 0
    # goldDarknet = None
    # goldPyFaster = None
    # readGoldDarknet = True
    # readGoldPyFaster = True
    # img_list_file = ""
    matchBench = sp.MatchBenchmark()
    for sdcItem in sdcItemList:

        progress = "{0:.2f}".format(float(sdci) / float(total_sdcs) * 100)
        sys.stdout.write("\rProcessing SDC " + str(sdci) + " of " + str(total_sdcs) + " - " + progress + "%")

        sys.stdout.flush()

        matchBench.processHeader(sdcItem, benchmarkname_machinename)
        currObj =  matchBench.currObj
        #
        # if isLavaMD and box is None:
        #     continue
        errorsParsed = []
        # Get error details from log string

        for errString in matchBench.errList:
            err = None
            # if isGEMM or isLud:
            #     err = parseErrGEMM(errString)
            # elif isHotspot:
            #
            #     err = parseErrHotspot(errString)
            # elif isLavaMD:
            #     err = parseErrLavaMD(errString, box, header)
            # elif isACCL:
            #     err = parseErrACCL(errString)
            # elif isNW:
            #     err = parseErrNW(errString)
            # elif isLulesh:
            #     err = parseErrLulesh(errString, 50, header)
            # elif isDarknet:
            #     err = pn.parseErrDarknet(errString)
            # elif isPyFaster:
            #     err = pn.parseErrPyFaster(errString, sdcIteration)
            err = currObj.parseErr(errString)

            if err is not None:
                errorsParsed.append(err)

        # (goldLines, detectedLines, xMass, yMass, precision, recall, falseNegative, falsePositive, truePositive,
        #  imgFile) = (
        #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # # Parse relative error
        # if isGEMM or isHotspot or isACCL or isNW or isLud:
        #     (maxRelErr, minRelErr, avgRelErr, zeroOut, zeroGold, relErrLowerLimit, errListFiltered, relErrLowerLimit2,
        #      errListFiltered2) = relativeErrorParser(errorsParsed)
        # elif isLavaMD:
        #     (maxRelErr, minRelErr, avgRelErr, zeroOut, zeroGold, relErrLowerLimit, errListFiltered, relErrLowerLimit2,
        #      errListFiltered2) = relativeErrorParserLavaMD(errorsParsed)
        # elif isLulesh:
        #     (maxRelErr, minRelErr, avgRelErr, zeroOut, zeroGold, relErrLowerLimit, errListFiltered, relErrLowerLimit2,
        #      errListFiltered2) = relativeErrorParserLulesh(errorsParsed)
        (maxRelErr, minRelErr, avgRelErr, zeroOut, zeroGold, relErrLowerLimit, errListFiltered, relErrLowerLimit2,
              errListFiltered2) = currObj.relativeErrorParser(errorsParsed)
        # object detection algorithms need other look
        # elif isPyFaster:
        #      # only for CNNs
        #     if readGoldPyFaster:
        #         goldPyFaster = pn.GoldContent(filepath=gold_dir, nn="pyfaster")
        #         readGoldPyFaster = False
        #
        #     (maxRelErr, minRelErr, avgRelErr, precision, recall, relErrLowerLimit, errListFiltered, relErrLowerLimit2,
        #      errListFiltered2) = pn.relativeErrorParserPyFaster(img_list_file, errorsParsed, goldPyFaster, sdci)
        #
        #
        # elif isDarknet:
        #     # only for CNNs
        #     if readGoldDarknet:
        #         goldDarknet = pn.GoldContent(filepath=gold_dir, nn="darknet")
        #         readGoldDarknet = False
        #
        #     (goldLines, detectedLines, xMass, yMass, precision, recall, falseNegative, falsePositive, truePositive,
        #      imgFile) = pn.relativeErrorParserDarknet(img_list_file, errorsParsed, goldDarknet)

        # Parse locality metric
        # if isGEMM or isHotspot or isACCL or isNW or isLud:
            # if isHotspot:
            #    print errListFiltered
            #    print errorsParsed
        (square, colRow, single, random) =  currObj.localityParser2D(errorsParsed)
        (squareF, colRowF, singleF, randomF) =  currObj.localityParser2D(errListFiltered)
        (squareF2, colRowF2, singleF2, randomF2) =  currObj.localityParser2D(errListFiltered2)
        jaccard =  currObj.jaccardCoefficient(errorsParsed)
        jaccardF =  currObj.jaccardCoefficient(errListFiltered)
        jaccardF2 =  currObj.jaccardCoefficient(errListFiltered2)
        errListFiltered = []
        errListFiltered2 = []
        cubic = 0
        cubicF = 0
        cubicF2 = 0

        if single == 0 and buildImages:
            if size is not None:
                currObj.buildImage(errorsParsed, size,
                                   currObj.dirName + '/' + currObj.header + '/' + currObj.logFileNameNoExt + '_' + str(imageIndex))
            else:
                currObj.buildImage(errorsParsed, 8192,
                                   currObj.dirName + '/' + currObj.header + '/' + currObj.logFileNameNoExt + '_' + str(imageIndex))
            imageIndex += 1

        elif isLavaMD or isLulesh:
            (cubic, square, colRow, single, random) = currObj.localityParser3D(errorsParsed)
            (cubicF, squareF, colRowF, singleF, randomF) = currObj.localityParser3D(errListFiltered)
            (cubicF2, squareF2, colRowF2, singleF2, randomF2) = currObj.localityParser3D(errListFiltered2)
            if isLavaMD:
                jaccard = currObj.jaccardCoefficientLavaMD(errorsParsed)
                jaccardF = currObj.jaccardCoefficientLavaMD(errListFiltered)
                jaccardF2 = currObj.jaccardCoefficientLavaMD(errListFiltered2)
            else:
                jaccard = jaccardF = jaccardF2 = 0
            errListFiltered = []
            errListFiltered2 = []

        else:  # Need to add locality parser for other benchmarks, if possible!
            (cubic, square, colRow, single, random) = [0, 0, 0, 0, 0]
            (cubicF, squareF, colRowF, singleF, randomF) = [0, 0, 0, 0, 0]
            (cubicF2, squareF2, colRowF2, singleF2, randomF2) = [0, 0, 0, 0, 0]
            jaccard = None
            jaccardF = None
            jaccardF2 = None

        # Write info to csv file
        # if fileNameSuffix is not None and fileNameSuffix != "":
        #   csvFileName = dirName+'/'+header+'/logs_parsed_'+machine+'_'+fileNameSuffix+'.csv'
        # else:
        if 'caltech' in gold_dir:
            dataset = 'caltech'
        else:
            dataset = 'voc2012'

        if isPyFaster:
            dir = "py_faster_csv_" + dataset
        elif isDarknet:
            dir = "darknet_csv_" + dataset
        else:
            dir = currObj.header
        csvFileName = currObj.dirName + '/' + dir + '/logs_parsed_' + currObj.machine + '.csv'
        if not os.path.exists(os.path.dirname(csvFileName)):
            try:
                os.makedirs(os.path.dirname(csvFileName))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        flag = 0
        if not os.path.exists(csvFileName):
            flag = 1

        csvWFP = open(csvFileName, "a")
        writer = csv.writer(csvWFP, delimiter=';')

        if flag == 1 and (isPyFaster or isDarknet):  # csv header
            # (len(gold), len(tempBoxes), x, y, pR.getPrecision(), pR.getRecall(), pR.getFalseNegative(),
            #  pR.getFalsePositive(), pR.getTruePositive())
            writer.writerow(["logFileName", "Machine", "Benchmark", "imgFile", "SDC_Iteration", "#Accumulated_Errors",
                             "#Iteration_Errors", "gold_lines", "detected_lines", "x_center_of_mass",
                             "y_center_of_mass", "precision", "recall", "false_negative", "false_positive",
                             "true_positive"])

        elif flag == 1:
            writer.writerow()

        writer.writerow(
            [currObj.logFileName, currObj.machine, currObj.benchmark, imgFile, currObj.sdcIteration, currObj.accIteErrors, currObj.iteErrors, goldLines, detectedLines,
             xMass, yMass, precision, recall, falseNegative, falsePositive, truePositive])

        csvWFP.close()
        sdci += 1

    sys.stdout.write(
        "\rProcessing SDC " + str(sdci - 1) + " of " + str(total_sdcs) + " - 100%                     " + "\n")
    sys.stdout.flush()


def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='Parse logs for Neural Networks')
    parser.add_argument('--gold', dest='gold_dir', help='Directory where gold is located',
                        default=GOLD_DIR, type=str)
    parser.add_argument('--database', dest='error_database',
                        help='Where database is located', default="errors_log_database")

    args = parser.parse_args()

    return args

###########################################
# MAIN
###########################################'

if __name__ == '__main__':
    args = parse_args()
    # db = shelve.open("errors_log_database") #python3
    db = shelve.open(args.error_database)  # python2
    # jump = True
    # for k, v in db.items(): #python3
    for k, v in db.iteritems():  # python2
        isHotspot = re.search("Hotspot", k, flags=re.IGNORECASE)
        isGEMM = re.search("GEMM", k, flags=re.IGNORECASE)
        isLavaMD = re.search("lavamd", k, flags=re.IGNORECASE)
        isCLAMR = re.search("clamr", k, flags=re.IGNORECASE)
        # algoritmos ACCL, NW, Lulesh, Mergesort e Quicksort
        isACCL = re.search("accl", k, flags=re.IGNORECASE)
        isNW = re.search("nw", k, flags=re.IGNORECASE)
        isLulesh = re.search("lulesh", k, flags=re.IGNORECASE)
        isLud = re.search("lud", k, flags=re.IGNORECASE)
        isDarknet = re.search("darknet", k, flags=re.IGNORECASE)
        isPyFaster = re.search("pyfasterrcnn", k, flags=re.IGNORECASE)

        if isHotspot or isGEMM or isLavaMD or isACCL or isNW or isLulesh or isLud or isDarknet or isPyFaster:
            print("Processing ", k)
            parseErrors(k, v, args.gold)
        else:
            print("Ignoring ", k)

    db.close()
