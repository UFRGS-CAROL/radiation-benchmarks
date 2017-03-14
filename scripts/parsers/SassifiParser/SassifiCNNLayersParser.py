import argparse
import csv

from ParsersClasses import DarknetParser
import Parameters as par

def parserLayersSassify(**kwargs):
    #open sassify csv
    sassifiCsvFilename = kwargs.pop("sassifi_csv")
    layersGoldPath = kwargs.pop("layers_gold_path")
    layersPath = kwargs.pop("layers_path")
    localRadiationBench = kwargs.pop("local_rad")
    datasets = kwargs.pop("datasets")


    csvfile = open(sassifiCsvFilename)
    reader = csv.DictReader(csvfile)

    #create a fake object to parse
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
        machine = row['Machine']
        benchmark = row['Benchmark']
        header = row['header']
        sdcIteration = row['SDC_Iteration']
        accIteErrors = row['#Accumulated_Errors']
        iteErrors = row['#Iteration_Errors']



        darknetParserObj.setDefaultValues(logFileName, machine, benchmark, header, sdcIteration, accIteErrors,
                         iteErrors, None, logFileName, header)

        darknetParserObj.parseLayers()

        darknetParserObj.writeToCSV()



if __name__ == "__main__":


    #parse a simple parser for layers on SASSIFY
    parserLayersSassify(sassifi_csv="",
                        layers_gold_path="",
                        layers_path="",
                        local_rad="",
                        datasets=""
                        )


