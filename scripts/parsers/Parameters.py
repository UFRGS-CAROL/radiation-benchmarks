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

# IMG_OUTPUT_DIR is the directory to where the images with error comparisons will be saved
IMG_OUTPUT_DIR = '' #''/home/pfpimenta/Dropbox/ufrgs/bolsaPaolo/img_corrupted_output/'

# LOCAL_RADIATION_BENCH must be the parent directory of the radiation-benchmarks folder
LOCAL_RADIATION_BENCH = '/mnt/4E0AEF320AEF15AD/PESQUISA/git_pesquisa'  # '/home/pfpimenta'

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
                                                   layersGoldPath=LAYERS_GOLD_PATH,
                                                   layersPath=LAYERS_PATH,
                                                   imgOutputDir=IMG_OUTPUT_DIR,
                                                   localRadiationBench=LOCAL_RADIATION_BENCH,
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









