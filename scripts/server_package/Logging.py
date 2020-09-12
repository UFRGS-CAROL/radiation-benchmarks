import time
import datetime

"""
Logging class
"""


class Logging:
    log_file = None
    debug_var = None
    unique_id = None

    def __init__(self, log_file, unique_id=''):
        self.log_file = log_file
        self.unique_id = unique_id

    def info(self, msg):
        with open(self.log_file, "a") as fp:
            d = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            fp.write("[INFO -- " + d + "]\n" + msg + "\n")
            # fp.close()

    def exception(self, msg):
        with open(self.log_file, "a") as fp:
            d = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            fp.write("[EXCEPTION -- " + d + "]\n" + msg + "\n")
            # fp.close()

    def error(self, msg):
        with open(self.log_file, "a") as fp:
            d = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            fp.write("[ERROR -- " + d + "]\n" + msg + "\n")
            # fp.close()

    def debug(self, msg):
        with open(self.log_file, "a") as fp:
            d = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            fp.write("[DEBUG -- " + d + "]\n" + msg + "\n")
            # fp.close()

    def summary(self, msg):
        with open(self.log_file, "a") as fp:
            d = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            fp.write("[SUMMARY -- " + d + "]\nFI-uniqueID=" + str(self.unique_id) + "\n" + msg + "\n")
            # fp.close()

    def search(self, find):
        with open(self.log_file, "r") as fp:
            lines = fp.readlines()
            # fp.close()
        for l in lines:
            if find in l:
                return l
        return None
