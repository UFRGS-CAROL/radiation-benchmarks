#!/usr/bin/env python

import csv


# from ParsersClasses import DarknetParser

DARKNET_DATASETS = {'caltech.pedestrians.critical.1K.txt': {'dumb_abft': 'gold.caltech.critical.abft.1K.test',
                                                         'no_abft': 'gold.caltech.critical.1K.test'},
                 'caltech.pedestrians.1K.txt': {'dumb_abft': 'gold.caltech.abft.1K.test',
                                                'no_abft': 'gold.caltech.1K.test'},
                 'voc.2012.1K.txt': {'dumb_abft': 'gold.voc.2012.abft.1K.test', 'no_abft': 'gold.voc.2012.1K.test'}}



def parserLayersSassify(**kwargs):
    # open sassify csv
    sassifiCsvFilename = kwargs.pop("sassifi_csv")
    layersGoldPath = kwargs.pop("layers_gold_path")
    layersPath = kwargs.pop("layers_path")
    localRadiationBench = kwargs.pop("local_rad")
    datasets = kwargs.pop("datasets")
    machine = kwargs.pop("machine")
    benchmark = kwargs.pop("benchmark")
    header = kwargs.pop("header")
    sdcIteration = kwargs.pop("sdc_it")
    iteErrors = kwargs.pop("it_errors")
    accIteErrors = kwargs.pop("acc_errors")

    csvfile = open(sassifiCsvFilename)
    reader = csv.DictReader(csvfile)

    # create a fake object to parse
    darknetParserObj = DarknetParser.DarknetParser(parseLayers=True,
                                                   prThreshold=0.5,
                                                   layersGoldPath=layersGoldPath,
                                                   layersPath=layersPath,
                                                   imgOutputDir='',
                                                   localRadiationBench=localRadiationBench,
                                                   check_csv=None,
                                                   ecc=False,
                                                   is_fi=False,
                                                   goldBaseDir='',
                                                   datasets=datasets
                                                   )

    for row in reader:
        logFileName = row['log_file']

        darknetParserObj.setDefaultValues(logFileName, machine, benchmark, header, sdcIteration, accIteErrors,
                                          iteErrors, None, logFileName, header)

        darknetParserObj.parseLayers()

        darknetParserObj.writeToCSV()

def isInside(logFileName, rows):
    for i in rows:
        if logFileName in i['logFileName']:
            return True,i

    return False, None


def getLayerHeaderName(layerNum, infoName, filterName):
    # layerHeaderName :: layer<layerNum><infoName>_<filterName>
    # <infoName> :: 'smallestError', 'biggestError', 'numErrors', 'errorsAverage', 'errorsVariance'
    # <filterName> :: 'allErrors', 'newErrors', 'propagatedErrors'
    layerHeaderName = 'layer' + str(layerNum) + infoName + '_' + filterName
    return layerHeaderName

def getLayerHeaderNameErrorType(layerNum):
        # layer3<layerNum>ErrorType
        layerHeaderName = 'layer' + str(layerNum) + 'ErrorType'
        return layerHeaderName

def joinCsvLayersParsers(csvFileLayersPath, csvFilePRPath, csvOutParsedPath, infoNames, filterNames):
    csvFileLayers = open(csvFileLayersPath,'r')
    dictLayers = csv.DictReader(csvFileLayers, delimiter=';')

    csvFilePR = open(csvFilePRPath, 'r')
    dictPR = csv.DictReader(csvFilePR, delimiter=';')

    csvFileParsed = open(csvOutParsedPath, 'w')
    joinedDictRet = csv.DictWriter(csvFileParsed,fieldnames=dictLayers.fieldnames, delimiter=';')
    joinedDictRet.writeheader()


    rowLayers = [i for i in dictLayers]
    rowPR = [i for i in dictPR]
    [i.close() for i in [csvFileLayers, csvFilePR]]


    header = [getLayerHeaderName(layerNum, infoName, filterName)
              for filterName in filterNames
              for infoName in infoNames
             for layerNum in xrange(32)]
    header.extend(getLayerHeaderNameErrorType(layerNum) for layerNum in xrange(32))


    for i in rowLayers:
        isIn, row = isInside(i['logFileName'], rowPR)
        if isIn:
            for errType in header:
                row[errType] = i[errType]
            row['failed_layer'] = i['failed_layer']
            joinedDictRet.writerow(row)
        else:
            joinedDictRet.writerow(i)

    csvFileParsed.close()

if __name__ == "__main__":
# parse a simple parser for layers on SASSIFY
#     parserLayersSassify(sassifi_csv="/home/fernando/temp/parser_layers_test/temp_fernando/logs_sdcs_darknet-inst.csv",
#                     layers_gold_path="/home/fernando/temp/parser_layers_test/temp_fernando/golds/",
#                     layers_path="/home/fernando/temp/parser_layers_test/temp_fernando/",
#                     local_rad='/home/fernando/git_pesquisa',
#                     datasets=DARKNET_DATASETS,
#                     machine="carolk402",
#                     benchmark="darknet",
#                     header="execution_type:yolo execution_model:valid img_list_path:/home/carol/radiation-benchmarks/data/networks_img_list/caltech.pedestrians.critical.1K.txt "
#                            "weights:/home/carol/radiation-benchmarks/data/darknet/yolo.weights config_file:/home/carol/radiation-benchmarks/data/darknet/yolo.cfg iterations:1 "
#                            "abft: no_abft",
#                     sdc_it='0',
#                     it_errors='1',
#                     acc_errors='0'
#                     )


    infoNames = ['smallestError', 'biggestError', 'numErrors', 'errorsAverage', 'errorsStdDeviation']
    filterNames = ['allErrors', 'newErrors', 'propagatedErrors']
    # inst
    joinCsvLayersParsers(
        #csvLayers
        "/home/fernando/Dropbox/UFRGS/Pesquisa/fault_injections/sassifi_darknet_paper_micro/inst/parsed_layers/logs_parsed_layers_inst.csv",
        #csvFilePRPath
        "/home/fernando/Dropbox/UFRGS/Pesquisa/fault_injections/sassifi_darknet_paper_micro/inst/parsed_layers/logs_parsed_inst_pr.csv",
        # csvOutParsedPath
        "/home/fernando/Dropbox/UFRGS/Pesquisa/fault_injections/sassifi_darknet_paper_micro/inst/parsed_layers/parsed_inst_pr_plus_layers.csv",

        infoNames,
        filterNames
    )
    # # rf
    # joinCsvLayersParsers(
    #     # csvLayers
    #     "",
    #     # csvFilePRPath
    #     "",
    #     # csvOutParsedPath
    #     ""
    # )