import re
from SortParser import SortParser


class QuicksortParser(SortParser):
    # esse metodo vai ser chamado por um outro na classe parser
    # e depois a lista que tem os resultados produzidos por todas as chamadas desse
    # metodo vai ser processada pelo _relativeErrorParser
    # ou seja, aqui voce processa as strings do .log e no _relativeErrorParser
    # voce calcula o que voce quer caulcular
    def parseErrMethod(self, errString):
        try:
            # ERR stream: 0, p: [0, 0], r: 3.0815771484375000e+02, e: 0.0000000000000000e+00
            m = re.match(".*ERR.*\[(\d+)..(\d+)\].*r\: ([0-9e\+\-\.]+).*e\: ([0-9e\+\-\.]+)", errString)
            if m:
                posX = int(m.group(1))
                posY = int(m.group(2))
                read = float(m.group(3))
                expected = float(m.group(4))
                return [posX, posY, read, expected]
            else:
                return None
        except ValueError:
            return None


    def _relativeErrorParser(self, errList):
        #implementar aqui o parser do erro
        for i in errList:
            #processa cada erro gerado no parseErrMethod
            #e escreve os resultados nos atributos
            pass

        self._timestamp = 0
        self._errOutOfOrder = 0
        self._errCorrupted = 0
        self._errLink = 0
        self._errSync = 0
        self._itOOO = 0
        self._itCorrupted = 0
        self._itSync = 0
        self._itOOOCorr = 0
        self._itSyncCorr = 0
        self._itSyncOOO = 0
        self._itLink = 0
        self._itLinkOOO = 0
        self._itLinkSync = 0
        self._itLinkCorr = 0
        self._itMultiple = 0
        self._balanceMismatches = 0
        #nao tem retorno


    #tem que dar um set para o size baseado no header
    #por exemplo se o size for um numero e o input
    #setar o _size para mergesort_input_1MB
    #porque as pastas dos csvs vao ter o nome do _size
    def setSize(self, header):
        self._size = None
        m = re.match(".*size\:(\d+).*", header)
        if m:
            try:
                self._size = int(m.group(1))
            except:
                self._size = None
        self._size = 'quicksort_input_' + str(self._size)



    def buildImageMethod(self):
        return False


