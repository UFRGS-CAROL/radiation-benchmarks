import re
import os
import errno
import ParsersClasses

"""All benchmarks must be an atribute of MatchBenchmark, it will turn allmost all parser process invisible"""


class MatchBenchmark(object):
    """        "darknet": ParsersClasses.DarknetParser
        , "hotspot": ParsersClasses.HotspotParser()
        , "hog": ParsersClasses.HogParser()
        , "lavamd": ParsersClasses.LavaMDParser()
        , "mergesort": ParsersClasses.MergesortParser()
        , "nw": ParsersClasses.NWParser()
        , "quicksort": ParsersClasses.QuicksortParser()
        , "accl": ParsersClasses.ACCLParser()
        , "pyfasterrcnn": ParsersClasses.FasterRcnnParser()
        , "lulesh": ParsersClasses.LuleshParser()
        , "lud": ParsersClasses.LudParser()}
    """
    # all fucking benchmarks here
    __radiationBenchmarks = {
        # "darknet": ParsersClasses.DarknetParser
        # , "hotspot": ParsersClasses.HotspotParser()
        # , "hog": ParsersClasses.HogParser()
        # , "lavamd": ParsersClasses.LavaMDParser()
        # , "mergesort": ParsersClasses.MergesortParser()
        # , "nw": ParsersClasses.NWParser()
        # , "quicksort": ParsersClasses.QuicksortParser()
        # , "accl": ParsersClasses.ACCLParser()
        # , "pyfasterrcnn": ParsersClasses.FasterRcnnParser()
        # , "lulesh": ParsersClasses.LuleshParser()
        # , "lud": ParsersClasses.LudParser()
        "gemm" : ParsersClasses.GemmParser()
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~


    """sdcItem is => [logfile name, header, sdc iteration, iteration total amount error, iteration accumulated error, list of errors ]"""

    def __init__(self, localDir):
        self.__benchSet = False
        self.__currBench = None
        self.__localDir = str(localDir)

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
        m = re.match("(.*)_(.*)", benchmarkMachineName)
        benchmark = "default"
        machine = "carol"
        if m:
            benchmark = m.group(1)
            machine = m.group(2)

        self.__dirName = self.__localDir + "/" + machine + "/" + benchmark
        if not os.path.exists(os.path.dirname(self.__dirName)):
            try:
                os.makedirs(os.path.dirname(self.__dirName))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        for key, values in self.__radiationBenchmarks.iteritems():
            isBench = re.search(str(key), benchmark, flags=re.IGNORECASE)
            if isBench:
                self.__currBench = self.__radiationBenchmarks[str(key)]
                # doind it I will have duplicate data, but it is the cost of generalization
                self.__currBench.setDefaultValues(logFileName, machine, benchmark, header, sdcIteration, accIteErrors,
                                                  iteErrors, errList, logFileNameNoExt, pureHeader)
                break

        if not isBench:
            ValueError.message += ValueError.message + "MatchBenchmark: There is no benchmark as " + str(benchmark)
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

    def getCurrentObj(self):
        return self.__currBench
