from Parser import Parser
import csv

class SortParser(Parser):
    #tem que setar essas variaveis no _relatievErrorParser
    _timestamp = None
    _errOutOfOrder = None
    _errCorrupted = None
    _errLink = None
    _errSync = None
    _itOOO = None
    _itCorrupted = None
    _itSync = None
    _itOOOCorr = None
    _itSyncCorr = None
    _itSyncOOO = None
    _itLink  = None
    _itLinkOOO = None
    _itLinkSync = None
    _itLinkCorr = None
    _itMultiple = None
    _balanceMismatches = None

    _csvHeader = ['Timestamp','Machine','Benchmark','Header','SDC','LOGGED_ERRORS','ACC_ERR',
                  'ACC_TIME','ERR_OUTOFORDER','ERR_CORRUPTED','ERR_LINK','ERR_SYNC',
                  'IT_OOO','IT_CORRUPTED','IT_LINK','IT_SYNC','IT_OOO_CORR','IT_SYNC_CORR',
                  'IT_SYNC_OOO','IT_LINK_OOO','IT_LINK_CORR','IT_LINK_SYNC','IT_MULTIPLE',
                  'BALANCE_MISMATCHES','logFileName']


    def getBenchmark(self):
        return self._benchmark

    def localityParser(self):
        pass

    def jaccardCoefficient(self):
        pass


    # esse metodo tambem vai ser chamado na classe pai, entao so tem que estar ajustado para escrever
    # o conteudo certo,
    # ele vai ser chamado em cada processamento de sdc
    def _writeToCSV(self, csvFileName):
        self._writeCSVHeader(csvFileName)

        try:

            csvWFP = open(csvFileName, "a")
            writer = csv.writer(csvWFP, delimiter=';')
            # ['Timestamp', 'Machine', 'Benchmark', 'Header', 'SDC', 'LOGGED_ERRORS', 'ACC_ERR',
            #  'ACC_TIME', 'ERR_OUTOFORDER', 'ERR_CORRUPTED', 'ERR_LINK', 'ERR_SYNC',
            #  'IT_OOO', 'IT_CORRUPTED', 'IT_LINK', 'IT_SYNC', 'IT_OOO_CORR', 'IT_SYNC_CORR',
            #  'IT_SYNC_OOO', 'IT_LINK_OOO', 'IT_LINK_CORR', 'IT_LINK_SYNC', 'IT_MULTIPLE',
            #  'BALANCE_MISMATCHES', 'logFileName']
            #arrumar para ficar as variaveis igual a ordem do do header
            outputList = [self._timestamp,
                          self._machine,
                          self._benchmark,
                          self._header,
                          self._sdcIteration,
                          self._iteErrors,
                          self._iteErrors,
                          self._accIteErrors,
                          self._errOutOfOrder,
                          self._errCorrupted,
                          self._errLink,
                          self._errSync,
                          self._itOOO,
                          self._itCorrupted,
                          self._itLink,
                          self._itOOOCorr,
                          self._itSyncCorr,
                          self._itSyncOOO,
                          self._itLinkOOO,
                          self._itLinkCorr,
                          self._itLinkSync,
                          self._itMultiple,
                          self._balanceMismatches,
                          self._logFileName,
                    ]

            # if self._abftType != 'no_abft' and self._abftType != None:
            #     outputList.extend([])

            writer.writerow(outputList)
            csvWFP.close()

        except:
            #ValueError.message += ValueError.message + "Error on writing row to " + str(csvFileName)
            print "Error on writing row to " + str(csvFileName)
            raise

    def relativeErrorParser(self):
        self._relativeErrorParser(self._errors["errorsParsed"])