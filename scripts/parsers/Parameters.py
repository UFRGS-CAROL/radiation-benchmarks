
from ParsersClasses import GemmParser
from ParsersClasses import ACCLParser
from ParsersClasses import DarknetParser
from ParsersClasses import FasterRcnnParser
from ParsersClasses import HogParser
from ParsersClasses import HotspotParser
from ParsersClasses import LavaMDParser
from ParsersClasses import NWParser
from ParsersClasses import LudParser
from ParsersClasses import LuleshParser
from ParsersClasses import MergesortParser
from ParsersClasses import QuicksortParser

#set all benchmarks to be parsed here
radiationBenchmarks = dict(
        darknet=DarknetParser.DarknetParser(),
        # hotspot=HotspotParser.HotspotParser(),
        #hog=HogParser.HogParser(),
        # lavamd=LavaMDParser.LavaMDParser(),
        # mergesort=MergesortParser.MergesortParser(),
        # nw=NWParser.NWParser(),
        # quicksort=QuicksortParser.QuicksortParser(),
        # accl=ACCLParser.ACCLParser(),
        pyfasterrcnn=FasterRcnnParser.FasterRcnnParser(),
        # lulesh=LuleshParser.LuleshParser(),
        # lud=LudParser.LudParser(),
        # gemm=GemmParser.GemmParser()
    )

# CNNs parameters ######################################################################################################

#these strings in GOLD_BASE_DIR must be the directory paths of the gold logs for each machine
cnnGoldBaseDir = {
    'carol-k402': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_K40',
    'carol-tx': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_TITAN',
    # carolx1a
    'carolx1a': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_X1/tx1b',
    # carolx1b
    'carolx1b': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_X1/tx1b',
    # carolx1c
    'carolx1c': '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_X1/tx1c',
    #fault injection
    'carolk402': '/home/fernando/Dropbox/UFRGS/Pesquisa/fault_injections/sassifi_darknet'
}


# darknet parameters ###################################################################################################

#IMG_OUTPUT_DIR is the directory to where the images with error comparisons will be saved
darknetImgOutputDir = '/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/img_corrupted_output'

#LOCAL_RADIATION_BENCH must be the parent directory of the radiation-benchmarks folder
localRadiationBenchDir = '/home/fernando/git_pesquisa'  # '/mnt/4E0AEF320AEF15AD/PESQUISA/git_pesquisa'

darknetDatasets = {
    # normal
    'caltech.pedestrians.critical.1K.txt': {'dumb_abft': 'gold.caltech.critical.abft.1K.test',
                                            'no_abft': 'gold.caltech.critical.1K.test'},
    'caltech.pedestrians.1K.txt': {'dumb_abft': 'gold.caltech.abft.1K.test', 'no_abft': 'gold.caltech.1K.test'},
    'voc.2012.1K.txt': {'dumb_abft': 'gold.voc.2012.abft.1K.test', 'no_abft': 'gold.voc.2012.1K.test'}
}


# Pyfaster parameters ###################################################################################################


pyFasterDatasets = {
    # normal
    'caltech.pedestrians.critical.1K.txt': 'gold.caltech.critical.1K.test',
    'caltech.pedestrians.1K.txt': 'gold.caltech.1K.test',
    'voc.2012.1K.txt': 'gold.voc.2012.1K.test'
}

# HOG parameters ###################################################################################################

hogLocalGoldFolder = "/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_K40/histogram_ori_gradients/" #"/home/aluno/parser_hog/gold/"
hogLocalTxtFolder = "/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_K40/networks_img_list/"#"/home/aluno/radiation-benchmarks/data/networks_img_list/"
hogParameters = "0,1.05,1,1,48,0.9,100"
