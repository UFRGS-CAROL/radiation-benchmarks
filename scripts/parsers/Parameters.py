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

############################################################################################
#################################DARKNET PARSER PARAMETERS##################################
############################################################################################
"""This section MUST, I WRITE MUST, BE SET ACCORDING THE GOLD PATHS"""


LAYERS_GOLD_PATH = '/home/fernando/temp/camadas/data/' #'/home/pfpimenta/darknetLayers/golds/'
LAYERS_PATH = '/home/fernando/temp/camadas/data/' #'/home/pfpimenta/darknetLayers/layers/'

# these strings in GOLD_BASE_DIR must be the directory paths of the gold logs for each machine
GOLD_BASE_DIR = {
    # 'carol-k402': '/home/pfpimenta/Dropbox/ufrgs/bolsaPaolo/GOLD_K40',
    # 'carol-tx': '/home/pfpimenta/Dropbox/ufrgs/bolsaPaolo/GOLD_TITAN',
    # # carolx1a
    # 'carolx1a': '/home/pfpimenta/Dropbox/ufrgs/bolsaPaolo/GOLD_X1/tx1b',
    # # carolx1b
    # 'carolx1b': '/home/pfpimenta/Dropbox/ufrgs/bolsaPaolo/GOLD_X1/tx1b',
    # # carolx1c
    # 'carolx1c': '/home/pfpimenta/Dropbox/ufrgs/bolsaPaolo/GOLD_X1/tx1c',
    # fault injection
    'carolk402': '/home/fernando/Dropbox/UFRGS/Pesquisa/Fault_Injections/sassifi_darknet_paper_micro'
}

# IMG_OUTPUT_DIR is the directory to where the images with error comparisons will be saved
IMG_OUTPUT_DIR = '' #''/home/pfpimenta/Dropbox/ufrgs/bolsaPaolo/img_corrupted_output/'

# LOCAL_RADIATION_BENCH must be the parent directory of the radiation-benchmarks folder
LOCAL_RADIATION_BENCH = '/mnt/4E0AEF320AEF15AD/PESQUISA/git_pesquisa'  # '/home/pfpimenta'

DATASETS = {
    # normal
    'caltech.pedestrians.critical.1K.txt': {'dumb_abft': 'gold.caltech.critical.abft.1K.test',
                                            'no_abft': 'gold.caltech.critical.1K.test'},
    'caltech.pedestrians.1K.txt': {'dumb_abft': 'gold.caltech.abft.1K.test', 'no_abft': 'gold.caltech.1K.test'},
    'voc.2012.1K.txt': {'dumb_abft': 'gold.voc.2012.abft.1K.test', 'no_abft': 'gold.voc.2012.1K.test'}
}





############################################################################################
#################################OVERALL PARAMETERS ########################################
############################################################################################

#set all benchmarks to be parsed here
radiationBenchmarks = {}

def setBenchmarks(**kwargs):
    benchmarks = kwargs.pop("benchmarks")
    pr_threshold = float(kwargs.pop("pr_threshold"))
    parse_layers = bool(kwargs.pop("parse_layers"))
    print "Parsing for: "
    for i in benchmarks:
        benchObj = None
        print i , " ",
        if i == 'darknet':
            benchObj = DarknetParser.DarknetParser(parseLayers=parse_layers,
                                                   pr_threshold=pr_threshold,
                                                   goldBaseDir=GOLD_BASE_DIR,
                                                   layersGoldPath=LAYERS_GOLD_PATH,
                                                   layersPath=LAYERS_PATH,
                                                   imgOutputDir=IMG_OUTPUT_DIR,
                                                   localRadiationBench=LOCAL_RADIATION_BENCH,
                                                   datasets=DATASETS,
                                                   )
        elif i == 'hotspot':
            benchObj = HotspotParser.HotspotParser()
        elif i == 'hog':
            benchObj = HogParser.HogParser()
        elif i == 'lavamd':
            benchObj = LavaMDParser.LavaMDParser()
        elif i == 'mergesort':
            benchObj = MergesortParser.MergesortParser()
        elif i == 'nw':
            benchObj = NWParser.NWParser()
        elif i == 'quicksort':
            benchObj = QuicksortParser.QuicksortParser()
        elif i == 'accl':
            benchObj = ACCLParser.ACCLParser()
        elif i == 'pyfasterrcnn':
            benchObj = FasterRcnnParser.FasterRcnnParser()
        elif i == 'lulesh':
            benchObj = LuleshParser.LuleshParser()
        elif i == 'lud':
            benchObj = LudParser.LudParser()
        elif i == 'gemm':
            benchObj = GemmParser.GemmParser()

        radiationBenchmarks[i] = benchObj









