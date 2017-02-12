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
        # darknet=DarknetParser.DarknetParser(),
         hotspot=HotspotParser.HotspotParser(),
        # hog=HogParser.HogParser(),
         lavamd=LavaMDParser.LavaMDParser(),
        # mergesort=MergesortParser.MergesortParser(),
         nw=NWParser.NWParser(),
        # quicksort=QuicksortParser.QuicksortParser(),
         accl=ACCLParser.ACCLParser(),
        # pyfasterrcnn=FasterRcnnParser.FasterRcnnParser(),
        # lulesh=LuleshParser.LuleshParser(),
        # lud=LudParser.LudParser(),
        gemm=GemmParser.GemmParser()
    )
