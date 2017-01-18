#!/usr/bin/env python
"""
this parser is select which SDC error select based on the calculed crossections
for example I have 900 SDCs on a radiation test
but not all of them are good for the criticality calcultion
so we select from this 900 errors which one that are good based on csvs generated by crossections scripts
"""
import argparse
import csv
import re
from datetime import datetime
import  os

def getFilename(csvFilename):
    csvFilename = os.path.basename(csvFilename)
    splited = csvFilename.split('.')[0]
    return splited + "_final.csv"

"""
csv1        - normal Parser.py output file
del1        - delimiter for csv1
csv2        - calcCrossSection.py output file (it will only be used for check timestamp)
del2        - delimiter for csv2
outputFile  - obvious outputFile
"""
def joinFiles(csv1, del1, csv2, del2, outputFile):
    csvFileOne = open(csv1)
    csvFileTwo = open(csv2)
    readerOne = csv.DictReader(csvFileOne, delimiter=del1)
    readerTwo = csv.DictReader(csvFileTwo, delimiter=del2)

    outputCsv = open(outputFile, 'w')
    writer = csv.DictWriter(outputCsv, fieldnames=readerOne.fieldnames)
    writer.writeheader()

    #it is not possible iterate a reader twice, so
    #copy it to a list first
    listCsv1 = [i for i in readerOne]
    listCsv2 = [i for i in readerTwo]

    for i in listCsv1:
        logFileName = i["logFileName"]
        #process data
        #2016_12_13_19_00_34_cudaDarknet_carol-k402.log
        m = re.match("(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(.*)_(.*).log", logFileName)
        if m:
            year = m.group(1)
            month = m.group(2)
            day = m.group(3)
            hour = m.group(4)
            minutes = m.group(5)
            second = m.group(6)
            benchmark = m.group(7)

            # assuming microsecond = 0
            currDate = datetime(int(year), int(month), int(day), int(hour), int(minutes), int(second))
            for j in listCsv2:
                startDate = j["start timestamp"]
                endDate = j["end timestamp"]
                startDate = datetime.strptime(startDate, "%c")
                endDate = datetime.strptime(endDate, "%c")
                if startDate <= currDate <= endDate:
                    writer.writerow(i)
    #finishing
    csvFileOne.close()
    csvFileTwo.close()
    outputCsv.close()


def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='Parser to select good runs with good crossections')
    # parser.add_argument('--gold', dest='gold_dir', help='Directory where gold is located',
    #                     default=GOLD_DIR, type=str)
    # parser.add_argument('--csv1', dest='csv1',
    #                     help='Where first csv file is located')
    # parser.add_argument('--del1', dest='del1',
    #                     help='delimiter for csv 1', default=',')
    #
    # parser.add_argument('--csv2', dest='csv2',
    #                     help='Where second csv file is located, it must be the file which have the start and end timestamp')
    #
    # parser.add_argument('--del2', dest='del2',
    #                     help='delimiter for csv 2', default=',')
    #
    # parser.add_argument('--out', dest='output_file', help='Output file',default='no_out_def')
    parser.add_argument('--dir', dest='dir', help='Output file', default='.')

    args = parser.parse_args()

    return args

###########################################
# MAIN
###########################################'
CROSS_SECTION_FILES_K40 = ['CROSSECTION_RESULTS/darknet_k40_abft.csv',
                        'CROSSECTION_RESULTS/darknet_k40_ecc_off.csv',
                        'CROSSECTION_RESULTS/darknet_k40_ecc_on.csv']

CROSS_SECTION_FILES_X1 = ['CROSSECTION_RESULTS/darknet_x1.csv']

CROSS_SECTION_FILES_TX = ['CROSSECTION_RESULTS/darknet_titan_abft.csv',
                        'CROSSECTION_RESULTS/darknet_titan_abft.csv',
                        'CROSSECTION_RESULTS/darknet_titan.csv'
                        ]

if __name__ == '__main__':
    args = parse_args()
    # csv1 = str(args.csv1)
    # csv2 = str(args.csv2)
    # del1 = str(args.del1)
    # del2 = str(args.del2)
    # outF = str(args.output_file)
    # if outF == 'no_out_def':
    #     outF = getFilename(csv1)

    # joinFiles(csv1, del1, csv2, del2, outF)
    for i in [args.dir + '/ecc_off', args.dir + '/ecc_on']:
        for root, dirs, files in os.walk(i):
            for file in files:
                smallPath = os.path.join(root, file)
                fullPath  = os.path.abspath(smallPath)

                if '_ecc' in smallPath:
                    continue
                if '.csv' in smallPath and ('ecc_on' in root or 'ecc_off' in root and '_ecc' not in root):

                    outPutFile = smallPath.split('logs_parsed')[0].replace('/', '_').replace('.', '_') + \
                                 smallPath.split('logs_parsed')[1]

                    outPutFile = os.path.abspath(root) + '/' + outPutFile

                    if 'ecc_on' in smallPath and '_ecc' not in smallPath:
                        if 'k40' in smallPath:
                            joinFiles(fullPath, ';', 'CROSSECTION_RESULTS/darknet_k40_ecc_on.csv', ',', outPutFile)

                    elif 'ecc_off' in smallPath and '_ecc' not in smallPath:
                        if 'k40' in smallPath:
                            if 'abft' in smallPath:
                                joinFiles(fullPath, ';', 'CROSSECTION_RESULTS/darknet_k40_abft.csv', ',', outPutFile)
                            else:
                                joinFiles(fullPath, ';', 'CROSSECTION_RESULTS/darknet_k40_ecc_off.csv', ',', outPutFile)

                        elif 'x1' in smallPath:
                            joinFiles(fullPath, ';', 'CROSSECTION_RESULTS/darknet_x1.csv', ',', outPutFile)

                        elif 'tx' in smallPath:
                            if 'abft' in smallPath:
                                joinFiles(fullPath, ';', 'CROSSECTION_RESULTS/darknet_titan_abft.csv', ',', outPutFile)
                            else:
                                joinFiles(fullPath, ';', 'CROSSECTION_RESULTS/darknet_titan.csv', ',', outPutFile)

            print
