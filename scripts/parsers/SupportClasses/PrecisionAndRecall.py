import multiprocessing

"""Calculates precision and recall between two sets of rectangles"""

class PrecisionAndRecall(object):
    precision = 0
    recall = 0
    specifity = 0
    falsePositive = 0
    falseNegative = 0
    truePositive = 0
    __threshold = 1


    def __init__(self, threshold):
        manager = multiprocessing.Manager()
        self.__threshold = threshold
        self.precision = manager.Value('f', 0.0)
        self.recall = manager.Value('f', 0.0)
        self.falseNegative = manager.Value('i', 0)
        self.truePositive = manager.Value('i', 0)
        self.falsePositive =manager.Value('i', 0)

    def __repr__(self):
        return "precision " + str(self.precision) + " recall:" + str(self.recall) + " false positive:" + str(self.falsePositive) \
                + " false negative:" + str(self.falseNegative) + " true positive:" + str(self.truePositive) + " threshold:" + str(self.__threshold)

    def getPrecision(self): return self.precision.value
    def getRecall(self): return self.recall.value

    def getFalsePositive(self): return self.falsePositive.value
    def getFalseNegative(self): return self.falseNegative.value
    def getTruePositive(self): return self.truePositive.value


    def cleanValues(self):
        self.precision.value = 0
        self.recall.value = 0
        self.falseNegative.value = 0
        self.truePositive.value =  0
        self.falsePositive.value =  0

    """
    Calculates the precision an recall value, based on Lucas C++ function for HOG
    gold = List of Rectangles
    found = List o Rectangles
    """

    # def precisionAndRecallSerial(self, gold, found):
        # self.recallMethod(gold, found)
        #
        # self.precisionMethod(gold, found)
        # self.precision.value = float(self.truePositive.value) / float(self.truePositive.value + self.falsePositive.value)
        # self.precisionAndRecallSerial(gold, found, self.

    def precisionAndRecallParallel(self, gold, found):
        # print "running in parallel"
        rp = multiprocessing.Process(target=self.recallMethod, args=(gold, found))
        pp = multiprocessing.Process(target=self.precisionMethod, args=(gold, found))
        rp.start()
        pp.start()
        rp.join()
        pp.join()
        # print "precision " + str(self.precision.value) + " recall:" + str(self.recall.value) + " false positive:" + str(self.falsePositive) \
        #         + " false negative:" + str(self.falseNegative) + " true positive:" + str(self.truePositive) + " threshold:" + str(self.threshold)
        if self.truePositive.value + self.falsePositive.value > 0:
            self.precision.value = float(self.truePositive.value)\
                               / float(self.truePositive.value + self.falsePositive.value)
        else:
            self.precision.value = 0
    """
        split precision for parallelism
    """
    
    def precisionMethod(self, gold, found):
        # print "passou precision"
        out_positive = 0
        for i in found:
            for g in gold:
                if (g.jaccard_similarity(i)) >= self.__threshold:
                    out_positive += 1
                    break
        self.falsePositive.value = len(found) - out_positive

    """
        split recall for parallelism
    """


    def recallMethod(self, gold, found):
        # print "passou recall"
        for i in gold:
            for z in found:
                if (i.jaccard_similarity(z)) >= self.__threshold:
                    self.truePositive.value += 1
                    break

        self.falseNegative.value = len(gold) - self.truePositive.value
        if self.truePositive.value + self.falseNegative.value > 0:
            self.recall.value = float(self.truePositive.value) / float(self.truePositive.value + self.falseNegative.value)
        else:
            self.recall.value = 0

    def precisionAndRecallSerial(self, gold, found):
        #precision
        outPositive = 0
        for i in found:
            for g in gold:
                if (g.jaccard_similarity(i)) >= self.__threshold:
                    outPositive += 1
                    break

        falsePositive = len(found) - outPositive

        #recall
        truePositive = 0
        for i in gold:
            for z in found:
                if (i.jaccard_similarity(z)) >= self.__threshold:
                    truePositive += 1
                    break

        falseNegative = len(gold) - truePositive
        # print "\n", "falsePositive", falsePositive, "falseNegative", falseNegative, "truePositive", truePositive
        recall = float(truePositive) / float(truePositive + falseNegative)

        precision = float(truePositive) / float(truePositive + falsePositive)
        self.falseNegative.value = falseNegative
        self.falsePositive.value = falsePositive
        self.truePositive.value  = truePositive
        # print "precision" , precision, "recall" ,recall
        self.precision.value = precision
        self.recall.value = recall

    """

    cPixel center_of_mass(vector<Rect> rectangles) {
    cPixel pixel;
    long x_total = 0, y_total = 0, pixels = 0;
    long x_max, x_min, y_max, y_min;
    for (long i = 0; i < rectangles.size(); i++) {
        x_max = rectangles[i].br_x;
        y_max = rectangles[i].br_y;
        x_min = x_max - rectangles[i].width;
        y_min = y_max - rectangles[i].height;

        for(long x = x_min; x <= x_max; x++) {
            for (long y = y_min; y <= y_max; y++) {
                x_total = x_total + x;
                y_total = y_total + y;
                pixels++;
            }
        }
    }
    pixel.x = x_total/pixels;
    pixel.y = y_total/pixels;

    return pixel;

    }

    return x and y pixel cordinates
    """

    def _centerOfMass(self, rectangles):
        xTotal = 0
        yTotal = 0
        pixels = 0
        for r in rectangles:
            # for x in xrange(r.left, r.right + 1):
            #  for y in xrange(r.bottom, r.top + 1):
            xTotal += ((r.bottom - r.top - 1) * (r.left - r.right - 1) * (r.left + r.right)) / 2
            yTotal += ((r.bottom - r.top - 1) * (r.bottom + r.top) * (r.left - r.right - 1)) / 2
            pixels += (r.top - r.bottom + 1) * (r.right - r.left + 1)

        if pixels > 0:
            return ((xTotal) / pixels, (yTotal) / pixels)
        else:
            return 0,0
    def centerOfMassGoldVsFound(self, gold, found, xSize, ySize):
        xGold, yGold = self._centerOfMass(gold)
        xFound, yFound = self._centerOfMass(found)

        # (test_center_of_mass.x - gold_center_of_mass.x)/x_size << "," << (double)(test_center_of_mass.y - gold_center_of_mass.y)/y_size
        return float(xFound - xGold) / xSize, float(yFound - yGold) / ySize

    # def getImageSize(self, imgPath):
    #     # print imgPath
    #     with Image.open(imgPath) as im:
    #         width, height = im.size
    #     return width, height
# #
# import random
# import Rectangle
# found = [Rectangle.Rectangle(random.randint(1,10),random.randint(1,10),random.randint(1,10),random.randint(1,10)) for n in range(1,10)]
# gold = [Rectangle.Rectangle(random.randint(5,15),random.randint(5,15),random.randint(5,15),random.randint(5,15)) for n in range(1,5)]
#
# print found , "\n"
# print gold
#
# print "fazendo PR"
# pr = PrecisionAndRecall(0.3)
# pr.precisionAndRecallSerial(gold, found)
# print pr.getPrecision()  , pr.getRecall()

