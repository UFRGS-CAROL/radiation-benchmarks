from Parser import Parser
import csv
import copy
import os
#build image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

class ObjectDetectionParser(Parser):

    # precisionRecallObj = None
    _prThreshold = 0.5
    _detectionThreshold = 0.24

    # def __init__(self):
    #     Parser.__init__(self)
        # self.precisionRecallObj = PrecisionAndRecall.PrecisionAndRecall(self._prThreshold)

    _classes = ['__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    #overiding csvheader
    _csvHeader = ["logFileName", "Machine", "Benchmark", "SDC_Iteration", "#Accumulated_Errors",
                             "#Iteration_Errors", "gold_lines", "detected_lines", "wrong_elements", "x_center_of_mass",
                             "y_center_of_mass", "precision", "recall", "false_negative", "false_positive",
                             "true_positive", "header"]

    # ["gold_lines",
    #     "detected_lines", "x_center_of_mass", "y_center_of_mass", "precision",
    #     "recall", "false_negative", "false_positive", "true_positive"]
    _goldLines = None
    _detectedLines = None
    _wrongElements = None
    _xCenterOfMass = None
    _yCenterOfMass = None
    _precision = None
    _recall = None
    _falseNegative = None
    _falsePositive = None
    _truePositive = None

    # only for darknet
    _abftType = None
    _rowDetErrors = None
    _colDetErrors = None

    def getBenchmark(self):
        return self._benchmark


    def _writeToCSV(self, csvFileName):
        if not os.path.isfile(csvFileName) and self._abftType != 'no_abft':
            self._csvHeader.extend(
                ["abft_type", "row_detected_errors", "col_detected_errors",
                    "header"])

        self._writeCSVHeader(csvFileName)

        try:

            csvWFP = open(csvFileName, "a")
            writer = csv.writer(csvWFP, delimiter=';')
            # ["logFileName", "Machine", "Benchmark", "imgFile", "SDC_Iteration",
            #     "#Accumulated_Errors", "#Iteration_Errors", "gold_lines",
            #     "detected_lines", "x_center_of_mass", "y_center_of_mass",
            #     "precision", "recall", "false_negative", "false_positive",
            #     "true_positive"]
            outputList = [self._logFileName,
                          self._machine,
                          self._benchmark,
                          self._sdcIteration,
                          self._accIteErrors,
                          self._iteErrors, self._goldLines,
            self._detectedLines,
            self._wrongElements,
            self._xCenterOfMass,
            self._yCenterOfMass,
            self._precision,
            self._recall,
            self._falseNegative,
            self._falsePositive,
            self._truePositive,
            self._header
            ]

            if self._abftType != 'no_abft':
                outputList.extend([self._abftType, self._rowDetErrors, self._colDetErrors])

            writer.writerow(outputList)
            csvWFP.close()

        except:
            #ValueError.message += ValueError.message + "Error on writing row to " + str(csvFileName)
            print "Error on writing row to " + str(csvFileName)
            raise

    def relativeErrorParser(self):
        self._relativeErrorParser(self._errors["errorsParsed"])

    def localityParser(self):
        pass

    def jaccardCoefficient(self):
        pass

    def copyList(self, objList):
        temp = []
        if 'Darknet' in self._benchmark:
            for i in objList:
                temp.append(i.deepcopy())
        elif self._benchmark == 'pyfasterrcnn':
            for i in objList:
                temp.append(copy.deepcopy(i))

        return temp

    def buildImageMethod(self, imageFile, rectangles, classes, probs):
        im = np.array(Image.open(imageFile), dtype=np.uint8)

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(im)

        # Create a Rectangle patch
        for r, c, p in zip(rectangles, classes, probs):
            rect = patches.Rectangle((r.left, r.bottom), r.width,
                                     r.height, linewidth=1, edgecolor='r',
                                     facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.text(r.left, r.bottom - 2,
                    'class ' + str(c) + ' prob ' + str(p),
                    bbox=dict(facecolor='blue', alpha=0.5), fontsize=14,
                    color='white')

        plt.show()