import Parser
import re

class LudParser(Parser):


    def getLogHeader(self, header):
        self.size = None
        m = re.match(".*size\:(\d+).*", header)
        if m:
            try:
                self.size = int(m.group(1))
            except:
                self.size = None



        # for lud
        m = re.match(".*matrix_size\:(\d+).*reps\:(\d+)", header)
        if m:
            try:
                self.m_size = int(m.group(1))
                self.size = int(m.group(2))
            except:
                self.m_size = None
                self.size = None