#!/usr/bin/env python

import sys
import os
import csv
import re
import collections
from PIL import Image
import struct
from sklearn.metrics import jaccard_similarity_score
import parse_neural_networks as pn
import shelve
import errno

# benchmarks dict => (bechmarkname_machinename : list of SDC item)
# SDC item => [logfile name, header, sdc iteration, iteration total amount error, iteration accumulated error, list of errors ]
# list of errors => list of strings with all the error detail print in lines using #ERR

toleratedRelErr = 2  # minimum relative error to be considered, in percentage
toleratedRelErr2 = 5  # minimum relative error to be considered, in percentage
buildImages = False  # build locality images

datasets = ['caltech.critical','caltech', 'voc.2012']


def getGoldName(txtList):
    txtName = str(os.path.basename(os.path.normpath(txtList)))
    for i in datasets:
        if i in txtName:
            return txtName.replace("txt", "test").replace('.pedestrians','')


def getDataset(header):
    for i in datasets:
        if i in header:
            return i

def parseErrors(benchmarkname_machinename, sdcItemList, gold_dir, brokenHeader='not_broken'):
    benchmark = benchmarkname_machinename
    machine = benchmarkname_machinename
    m = re.match("(.*)_(.*)", benchmarkname_machinename)
    if m:
        benchmark = m.group(1)
        machine = m.group(2)

    dirName = "./" + machine + "/" + benchmark
    if not os.path.exists(os.path.dirname(dirName)):
        try:
            os.makedirs(os.path.dirname(dirName))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    sdci = 1
    total_sdcs = len(sdcItemList)
    imageIndex = 0
    goldDarknet = {i:[True,0] for i in datasets}
    goldPyFaster = {i:[True,0] for i in datasets}
    #print goldDarknet
    dataset = ''
    # readGoldDarknet = [True for i in datasets]
    # readGoldPyFaster = [True for i in datasets]
    img_list_file = ""
    for sdcItem in sdcItemList:

        progress = "{0:.2f}".format(float(sdci) / float(total_sdcs) * 100)
        sys.stdout.write("\rProcessing SDC " + str(sdci) + " of " + str(total_sdcs) + " - " + progress + "%")

        sys.stdout.flush()

        logFileName = sdcItem[0]
        header = sdcItem[1]
        pure_header = sdcItem[1]
        header = re.sub(r"[^\w\s]", '-', header)  # keep only numbers and digits
        sdcIteration = sdcItem[2]
        iteErrors = sdcItem[3]
        accIteErrors = sdcItem[4]
        errList = sdcItem[5]

        dataset = getDataset(pure_header)
        print "\n" , dataset #, "\n", pure_header
        logFileNameNoExt = logFileName
        m = re.match("(.*).log", logFileName)

        # for each header of each benchmark
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




        darknet_m = re.match(
            ".*execution_type\:(\S+).*execution_model\:(\S+).*img_list_path\:(\S+).*weights\:(\S+).*config_file\:(\S+).*iterations\:(\d+).*",
            pure_header)

        if darknet_m:
            try:
                execution_type = darknet_m.group(1)
                execution_model = darknet_m.group(2)
                img_list_path = darknet_m.group(3)
                img_list_file = (gold_dir.replace('darknet/','')) + os.path.basename(os.path.normpath(img_list_path))
                weights = darknet_m.group(4)
                config_file = darknet_m.group(5)
                iterations = darknet_m.group(6)

            except:
                #header is not ok
                pass
            # img_list_file = currentImgFile

        # for CNNs
        # once I need to know the paths of inputs I cannot use the changed header
        # darknet
        if "txt" not in pure_header and brokenHeader != 'not_broken':
        # sdci += 1
            img_list_file = brokenHeader


        # pyfaster
        py_faster_m = re.match(".*iterations\: (\d+).*img_list\: (\S+).*board\: (\S+).*", pure_header)
        if py_faster_m:
            iterations = py_faster_m.group(1)
            img_list_path = py_faster_m.group(2)
            img_list_file = (gold_dir.replace('pyfaster/','')) + os.path.basename(os.path.normpath(img_list_path))
            board = py_faster_m.group(3)
            # img_list_file = currentImgFile

        isDarknet = re.search("darknet", k, flags=re.IGNORECASE)
        isPyFaster = re.search("pyfasterrcnn", k, flags=re.IGNORECASE)

        goldFile = getGoldName(img_list_path)
        goldFile = gold_dir +  "gold." + goldFile
        errorsParsed = []
        # Get error details from log string

        for errString in errList:
            err = None
            if isDarknet:
                err = pn.parseErrDarknet(errString)
            elif isPyFaster:
                err = pn.parseErrPyFaster(errString, sdcIteration)

            if err is not None:
                errorsParsed.append(err)

        (goldLines, detectedLines, xMass, yMass, precision, recall, falseNegative, falsePositive, truePositive,
         imgFile) = (
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Parse relative error
        # object detection algorithms need other look
        if isPyFaster:
             # only for CNNs
            if goldPyFaster[dataset][0]:
                print goldFile
                goldPyFaster[dataset][1] = pn.GoldContent(filepath=goldFile, nn="pyfaster")
                goldPyFaster[dataset][0] = False
            (maxRelErr, minRelErr, avgRelErr, precision, recall, relErrLowerLimit, errListFiltered, relErrLowerLimit2,
             errListFiltered2) = pn.relativeErrorParserPyFaster(img_list_file, errorsParsed,  goldPyFaster[dataset][1], str(sdcIteration))


        elif isDarknet:
            # only for CNNs
            if goldPyFaster[dataset][0]:
                goldPyFaster[dataset][1] = pn.GoldContent(filepath=goldFile, nn="darknet")
                goldPyFaster[dataset][0] = False
            (goldLines, detectedLines, xMass, yMass, precision, recall, falseNegative, falsePositive, truePositive,
             imgFile) = pn.relativeErrorParserDarknet(img_list_file, errorsParsed, goldPyFaster[dataset][1], str(sdcIteration))


        # Write info to csv file
        # if fileNameSuffix is not None and fileNameSuffix != "":
        #   csvFileName = dirName+'/'+header+'/logs_parsed_'+machine+'_'+fileNameSuffix+'.csv'
        # else:

        if isPyFaster:
            dir = "py_faster_csv_" + dataset
        elif isDarknet:
            dir = "darknet_csv_" + dataset
        else:
            dir = header
        csvFileName = dirName + '/' + dir + '/logs_parsed_' + machine + '.csv'
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

        writer.writerow(
            [logFileName, machine, benchmark, imgFile, sdcIteration, accIteErrors, iteErrors, goldLines, detectedLines,
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
                        default=pn.GOLD_DIR, type=str,required=True)
    parser.add_argument('--database', dest='error_database',
                        help='Where database is located', default="errors_log_database", required=True)

    parser.add_argument('--broken', dest='brokenHeader', help='If darknet has a broken header', default='not_broken')
    args = parser.parse_args()

    return args

###########################################
# MAIN
###########################################'

if __name__ == '__main__':
    args = parse_args()
    # db = shelve.open("errors_log_database") #python3
    db = shelve.open(str(args.error_database))  # python2
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
            parseErrors(k, v, str(args.gold_dir), str(args.brokenHeader))
        else:
            print("Ignoring ", k)

    db.close()
