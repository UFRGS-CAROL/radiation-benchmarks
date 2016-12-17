import re
import os
import errno
import DarknetParser, HotspotParser, HogParser, LavaMDParser
import MergesortParser, NWParser, QuicksortParser, ACCLParser, FasterRcnnParser
import LuleshParser, LudParser

"""All benchmarks must be an atribute of MatchBenchmark, it will turn allmost all parser process invisible"""
class MatchBenchmark(object):

    #all fucking benchmarks here
    radiationBenchmarks = {"darknet":DarknetParser()
                              , "hotspot":HotspotParser()
                              , "hog":HogParser()
                              , "lavamd":LavaMDParser()
                              , "mergesort":MergesortParser()
                              , "nw":NWParser()
                              , "quicksort":QuicksortParser()
                              , "accl":ACCLParser()
                              , "pyfasterrcnn":FasterRcnnParser()
                              , "lulesh":LuleshParser()
                              , "lud":LudParser()}
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~


    """sdcItem is => [logfile name, header, sdc iteration, iteration total amount error, iteration accumulated error, list of errors ]"""
    def __init__(self):
        for key, value in self.radiationBenchmarks.iteritems():
            self.benchSet = False
            self.currBench = None


    #

    def processHeader(self, sdcItem, benchmarkMachineName):
        self.logFileName = sdcItem[0]
        self.header = sdcItem[1]
        self.pure_header = sdcItem[1]
        self.header = re.sub(r"[^\w\s]", '-', self.header)  # keep only numbers and digits
        self.sdcIteration = sdcItem[2]
        self.iteErrors = sdcItem[3]
        self.accIteErrors = sdcItem[4]
        self.errList = sdcItem[5]
        m = re.match("(.*)_(.*)", benchmarkMachineName)
        if m:
            self.benchmark = m.group(1)
            self.machine = m.group(2)

        self.dirName = "./" + self.machine + "/" + self.benchmark
        if not os.path.exists(os.path.dirname(self.dirName)):
            try:
                os.makedirs(os.path.dirname(self.dirName))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise


        for key, values in self.radiationBenchmarks.iteritems():
            isBench = re.search(str(key), self.benchmark, flags=re.IGNORECASE)
            if isBench:
                self.currBench = self.radiationBenchmarks[str(key)]
                self.benchSet = True
                break

        if not self.benchSet:
            ValueError.message += ValueError.message + "MatchBenchmark: There is no benchmark as " + str(self.benchmark)
            raise

            # isHotspot = re.search("hotspot", self.benchmark, flags=re.IGNORECASE)
    # isGEMM = re.search("GEMM", self.benchmark, flags=re.IGNORECASE)
    # isLavaMD = re.search("lavamd", self.benchmark, flags=re.IGNORECASE)
    # isCLAMR = re.search("clamr", self.benchmark, flags=re.IGNORECASE)
    # # algoritmos ACCL, NW, Lulesh, Mergesort e Quicksort
    # isACCL = re.search("accl", self.benchmark, flags=re.IGNORECASE)
    # isNW = re.search("nw", self.benchmark, flags=re.IGNORECASE)
    # isLulesh = re.search("lulesh", self.benchmark, flags=re.IGNORECASE)
    # isLud = re.search("lud", self.benchmark, flags=re.IGNORECASE)
    # isMergesort = re.search("mergesort", self.benchmark, flags=re.IGNORECASE)
    # isQuicksort = re.search("quicksort", self.benchmark, flags=re.IGNORECASE)
    # isDarknet = re.search("darknet", self.benchmark, flags=re.IGNORECASE)
    # isPyFaster = re.search("pyfasterrcnn", self.benchmark, flags=re.IGNORECASE)
    def turnBenchmarkOff(self):
        if self.benchSet:
            self.benchParser[self.benchmark][0] = False


    def matchBench(self, logFileName, ):
        self.logFileNameNoExt = logFileName
        m = re.match("(.*).log", logFileName)

        # for each header of each benchmark
        if m:
            self.logFileNameNoExt = m.group(1)






