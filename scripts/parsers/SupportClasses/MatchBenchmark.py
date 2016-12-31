import re
import os
import errno

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

"""All benchmarks must be an atribute of MatchBenchmark, it will turn allmost all parser process invisible"""


class MatchBenchmark():
    # all fucking benchmarks here
    __radiationBenchmarks = dict(
        darknet=DarknetParser.DarknetParser(),
        # hotspot=HotspotParser.HotspotParser(),
        # hog=HogParser.HogParser(),
        # lavamd=LavaMDParser.LavaMDParser(),
        # mergesort=MergesortParser.MergesortParser(),
        # nw=NWParser.NWParser(),
        # quicksort=QuicksortParser.QuicksortParser(),
        # accl=ACCLParser.ACCLParser(),
        # pyfasterrcnn=FasterRcnnParser.FasterRcnnParser(),
        # lulesh=LuleshParser.LuleshParser(),
        # lud=LudParser.LudParser(),
        # gemm=GemmParser.GemmParser()
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    __notMatchedBenchs = []
    """sdcItem is => [logfile name, header, sdc iteration, iteration total amount error, iteration accumulated error, list of errors ]"""

    def __init__(self):
        self.__currBench = None

    #

    def processHeader(self, sdcItem, benchmarkMachineName):
        logFileName = sdcItem[0]
        try:
            logFileNameNoExt = (logFileName.split("."))[0]
        except:
            logFileNameNoExt = ""

        pureHeader = sdcItem[1]
        header = re.sub(r"[^\w\s]", '-', pureHeader)  # keep only numbers and digits
        sdcIteration = sdcItem[2]
        iteErrors = sdcItem[3]
        accIteErrors = sdcItem[4]
        errList = sdcItem[5]
        # print "\n" , len(errList)
        m = re.match("(.*)_(.*)", benchmarkMachineName)
        benchmark = "default"
        machine = "carol"
        if m:
            benchmark = m.group(1)
            machine = m.group(2)


        isBench = False
        key = None
        for key, values in self.__radiationBenchmarks.iteritems():
            isBench = re.search(str(key), benchmark, flags=re.IGNORECASE)
            if isBench:
                # print "\n\ncurrent bench ", key
                # print "\n\n len " ,logFileName, machine, benchmark, header, sdcIteration, accIteErrors, iteErrors
                self.__currBench = self.__radiationBenchmarks[str(key)]
                # doind it I will have duplicate data, but it is the cost of generalization
                self.__currBench.setDefaultValues(logFileName, machine, benchmark, header, sdcIteration, accIteErrors,
                                                  iteErrors, errList, logFileNameNoExt, pureHeader)
                #print self.__currBench.debugAttPrint()

                break

        if not isBench:
            if key not in self.__notMatchedBenchs:
                self.__notMatchedBenchs.append(benchmark)
        return isBench

            #raise BaseException

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
    # def turnBenchmarkOff(self):
    #     if self.__benchSet:
    #         self.__benchParser[self.__benchmark][0] = False

    # #it is only for build image
    # def matchBench(self, logFileName):
    #     self.__logFileNameNoExt = logFileName
    #     m = re.match("(.*).log", logFileName)
    #
    #     # for each header of each benchmark
    #     if m:
    #         self.__logFileNameNoExt = m.group(1)
    def checkNotDoneBenchs(self):
        if len(self.__notMatchedBenchs) == 0:
            return ""

        return "These benchmarks were not found on radiation_list " + str(set(self.__notMatchedBenchs))


    def getCurrentObj(self):
        return self.__currBench

    def parseErrCall(self):
        self.__currBench.parseErr()

    def relativeErrorParserCall(self):
        self.__currBench.relativeErrorParser()


    def localityParserCall(self):
        self.__currBench.localityParser()

    def jaccardCoefficientCall(self):
        self.__currBench.jaccardCoefficient()

    def buildImageCall(self):
        self.__currBench.buildImageMethod()

    def writeToCSVCall(self):
        self.__currBench.writeToCSV()