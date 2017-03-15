#!/usr/bin/env python

import csv


from ParsersClasses import DarknetParser

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


# if __name__ == "__main__":
# parse a simple parser for layers on SASSIFY
parserLayersSassify(sassifi_csv="/home/fernando/temp/parser_layers_test/temp_fernando/logs_sdcs_darknet-inst.csv",
                    layers_gold_path="/home/fernando/temp/parser_layers_test/temp_fernando/golds/",
                    layers_path="/home/fernando/temp/parser_layers_test/temp_fernando/",
                    local_rad='/home/fernando/git_pesquisa',
                    datasets=DARKNET_DATASETS,
                    machine="carolk402",
                    benchmark="darknet",
                    header="execution_type:yolo execution_model:valid img_list_path:/home/carol/radiation-benchmarks/data/networks_img_list/caltech.pedestrians.critical.1K.txt "
                           "weights:/home/carol/radiation-benchmarks/data/darknet/yolo.weights config_file:/home/carol/radiation-benchmarks/data/darknet/yolo.cfg iterations:1 "
                           "abft: no_abft",
                    sdc_it='0',
                    it_errors='1',
                    acc_errors='0'
                    )
