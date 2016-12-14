import multiprocessing
import subprocess

"""Calculates precision and recall between two sets of rectangles"""

class PrecisionAndRecall(object):
    precision = 0
    recall = 0
    falsePositive = 0
    falseNegative = 0
    truePositive = 0
    threshold = 1

    def __init__(self, threshold):
        manager = multiprocessing.Manager()
        self.threshold = threshold
        self.precision = manager.Value('f', 0.0)
        self.recall = manager.Value('f', 0.0)
        self.falseNegative = manager.Value('i', 0)
        self.truePositive = manager.Value('i', 0)
        self.falsePositive =manager.Value('i',0)

    def __repr__(self):
        return "precision " + str(self.precision) + " recall:" + str(self.recall) + " false positive:" + str(self.falsePositive) \
                + " false negative:" + str(self.falseNegative) + " true positive:" + str(self.truePositive) + " threshold:" + str(self.threshold)

    def getPrecision(self): return self.precision.value
    def getFalsePositive(self): return self.falsePositive.value
    def getFalseNegative(self): return self.falseNegative.value
    def getTruePositive(self): return self.truePositive.value
    def getRecall(self): return self.precision.value

    """
    Calculates the precision an recall value, based on Lucas C++ function for HOG
    gold = List of Rectangles
    found = List o Rectangles
    """

    def precisionAndRecallSerial(self, gold, found):
        self.recallMethod(gold, found)

        self.precisionMethod(gold, found)
        self.precision.value = float(self.truePositive.value) / float(self.truePositive.value + self.falsePositive.value)

    def precisionAndRecallParallel(self, gold, found):
        # print "running in parallel"
        rp = multiprocessing.Process(target=self.recallMethod, args=(gold, found))
        pp = multiprocessing.Process(target=self.precisionMethod, args=(gold, found))
        rp.start()
        pp.start()
        rp.join(timeout=1)
        pp.join(timeout=1)
        # print "precision " + str(self.precision.value) + " recall:" + str(self.recall.value) + " false positive:" + str(self.falsePositive) \
        #         + " false negative:" + str(self.falseNegative) + " true positive:" + str(self.truePositive) + " threshold:" + str(self.threshold)
        self.precision.value = float(self.truePositive.value) / float(self.truePositive.value + self.falsePositive.value)

    """
        split precision for parallelism
    """

    def precisionMethod(self, gold, found):
        # print "passou precision"
        out_positive = 0
        for i in found:
            for g in gold:
                # calc the intersection
                intRect = g.intersection(i)
                intersectionArea = intRect.area()
                unionRect = g.union(i)
                unionArea = unionRect.area()

                if (float(intersectionArea) / float(unionArea)) >= self.threshold:
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
                intRect = z.intersection(i)
                unionRect = z.union(i)
                intersectionArea = intRect.area()
                unionArea = unionRect.area()
                if (float(intersectionArea) / float(unionArea)) >= self.threshold:
                    self.truePositive.value += 1
                    break

        self.falseNegative.value = len(gold) - self.truePositive.value
        self.recall.value = float(self.truePositive.value) / float(self.truePositive.value + self.falseNegative.value)


    def precisionRecall(self, gold, found, threshold):
        #precision
        outPositive = 0
        for i in found:
            for g in gold:
                # calc the intersection
                intRect = g.intersection(i)
                intersectionArea = intRect.area()
                unionRect = g.union(i)
                unionArea = unionRect.area()

                if (float(intersectionArea) / float(unionArea)) >= threshold:
                    outPositive += 1
                    break

        falsePositive = len(found) - outPositive

        #recall
        truePositive = 0
        for i in gold:
            for z in found:
                intRect = z.intersection(i)
                unionRect = z.union(i)
                intersectionArea = intRect.area()
                unionArea = unionRect.area()
                if (float(intersectionArea) / float(unionArea)) >= threshold:
                    truePositive += 1
                    break

        falseNegative = len(gold) - truePositive
        recall = float(truePositive) / float(truePositive + falseNegative)

        precision = float(truePositive) / float(truePositive + falsePositive)
        self.falseNegative.value = falseNegative
        self.falsePositive.value = falsePositive
        self.truePositive.value  = truePositive

        return recall, precision