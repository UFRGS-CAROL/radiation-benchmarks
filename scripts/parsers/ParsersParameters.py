"""check which parser is going to execute
"""
import argparse


import sys
import ConfigParser
import sys
import subprocess


INSTALLDIRS = ["ParsersClasses", "SassifiParser", "SortsParser", "CrossSectionScripts"]

"""Execute a parser object, in a generic way"""
class Execute(object):
    readyToProcess = False
    execute = {}
    def __init__(self, **kwargs):
        self.confFile = '/etc/radiation-benchmarks.conf'
        try:
            self.config = ConfigParser.RawConfigParser()
            self.config.read(self.confFile)

            self.installDir = self.config.get('DEFAULT', 'installdir') + "/"
            self.varDir = self.config.get('DEFAULT', 'vardir') + "/"
            self.logDir = self.config.get('DEFAULT', 'logdir') + "/"
            self.tmpDir = self.config.get('DEFAULT', 'tmpdir') + "/"

        except IOError as e:
            print >> sys.stderr, "Configuration setup error: " + str(e)
            sys.exit(1)

        #after configuration get
        benchmark = kwargs.pop("bench")
        if benchmark == "hog":
            self.hogList = kwargs.pop("hog_list")
            self.readyToProcess = True
            self.execute["command_line"] = self.installDir + "/scripts/parsers/nn_parsers/"

    def process(self):
        strExe = str(self.execute["command_line"]) + " " + str(" ".join(self.hogList))
        proc = subprocess.Popen(strExe, stdout=subprocess.PIPE, shell=True)
        try:
            (out, err) = proc.communicate()
        except:
            print "Error on executing command"
            exit(-1)

    def addDirs(self):
        for i in INSTALLDIRS:
            stringToAdd = str(self.installDir) + "/scripts/parsers/" +  str(i)
            sys.path.insert(0, stringToAdd)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Parse logs for Neural Networks')
    parser.add_argument('--get_info', dest='toGetInfo', help='If you want to retrieve all #Info',
                        default='no_info', type=str,required=True)

    args = parser.parse_args()

    return args




