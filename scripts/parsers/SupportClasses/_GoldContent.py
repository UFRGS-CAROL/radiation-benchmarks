import copy
import math
import pickle
from ctypes import *
import numpy as np

#from SupportClasses
import Rectangle

"""Read a darknet Gold content to memory"""


class Float(Structure):
    _fields_ = [('f', c_float)]

    def __repr__(self):
        return str(self.f)


class Box(Structure):
    _fields_ = [('x', c_float), ('y', c_float), ('w', c_float), ('h', c_float)]

    def __repr__(self):
        return str(self.x) + " " + str(self.y) + " " + str(self.w) + " " + str(self.h)


class Long(Structure):
    _fields_ = [('l', c_long)]

    def __repr__(self):
        return str(self.l)


class Point(object):
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y


class _GoldContent():
    __plistSize = 0
    __classes = 0
    __totalSize = 0
    __prob_array = {}
    __pyFasterGold = []
    pyFasterImgList = ""

    # return a dict that look like this
    # //to store all gold filenames
    # typedef struct gold_pointers {
    # //	box *boxes_gold;
    # //	ProbArray pb;
    # 	ProbArray *pb_gold;
    # 	long plist_size;
    # 	FILE* gold;
    # 	int has_file;
    # } GoldPointers;
    #
    def __init__(self, **kwargs):
        #use keyargs
        try:
            nn = kwargs.pop("nn")
            filePath = kwargs.pop("filepath")

            if "darknet" in nn:
                self.darknetConstructor(filePath)
            elif "pyfaster" in nn:
                self.pyFasterConstructor(filePath)
        except:
            pass


    def getTotalSize(self): return self.__totalSize
    def getClasses(self): return self.__classes
    def getPlistSize(self): return self.__plistSize
    def getPyFasterGold(self): return self.__pyFasterGold

    def getRectArray(self):
        return self.__prob_array['boxes']

    def getProbArray(self):
        return self.__prob_array['probs']

    def darknetConstructor(self, filePath):
        cc_file = open(filePath, 'rb')
        result = []
        # darknet write file in this order, so we need recover data in this order
        # long plist_size;
        # long classes;
        # long total_size;
        # for ( < plist_size times >){
        # -----pb_gold.boxes
        # -----pb_gold.probs
        # }
        plist_size = Long()
        classes = Long()
        total_size = Long()
        cc_file.readinto(plist_size)
        cc_file.readinto(classes)
        cc_file.readinto(total_size)
        plist_size = long(plist_size.l)
        classes = long(classes.l)
        total_size = long(total_size.l)
        i = 0
        self.__plistSize = plist_size
        self.__classes = classes
        self.__totalSize = total_size
        self.__prob_array["boxes"] = []
        self.__prob_array["probs"] = []

        while i < long(plist_size):
            # boxes has total_size size
            boxes = self.readBoxes(cc_file, total_size)
            probs = self.readProbs(cc_file, total_size, classes)
            self.__prob_array["probs"].append(probs)
            self.__prob_array["boxes"].append(boxes)
            i += 1

        cc_file.close()


    def pyFasterConstructor(self, filepath):
        try:
            f = open(filepath, "rb")
            tempGold = pickle.load(f)
            f.close()

        except:
            raise
        self.__pyFasterGold = tempGold



    def __copy__(self):
        return copy.deepcopy(self)

    def readBoxes(self,cc_file, n):
        i = 0
        boxes = np.empty(n, dtype=object)
        while i < n:
            box = Box()
            cc_file.readinto(box)
            # instead boxes I put the Rectangles to make it easy
            # (self, x, y, height, width, br_x, br_y):
            # x_min_gold = x_max_gold - gold[i].width;
            # y_min_gold = y_max_gold - gold[i].height;
            # left, top, right, bottom):
            left = int(box.x)
            bottom = int(box.y)
            h = int(box.h)
            w = int(box.w)
            boxes[i] = Rectangle.Rectangle(left, bottom, w, h)
            i += 1
        return boxes

    def readProbs(self, cc_file, total_size, classes):
        i = 0
        prob = np.empty((total_size, classes), dtype=object)

        while i < total_size:
            j = 0
            while j < classes:
                pb_ij = Float()
                cc_file.readinto(pb_ij)
                prob[i][j] = float(pb_ij.f)
                j += 1
            i += 1

        return prob



# gold = _GoldContent(nn='darknet', filepath="/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_K40/darknet/gold.voc.2012.1K.test")
#
# print gold.getRectArray()[77][31]