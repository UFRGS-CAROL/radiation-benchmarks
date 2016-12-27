from Parser import Parser
import csv
import os

class ObjectDetectionParser(Parser):
    __prThreshold = 0.5

    _classes = ['__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    #overiding csvheader
    _csvHeader = ["logFileName", "Machine", "Benchmark", "imgFile", "SDC_Iteration", "#Accumulated_Errors",
                             "#Iteration_Errors", "gold_lines", "detected_lines", "x_center_of_mass",
                             "y_center_of_mass", "precision", "recall", "false_negative", "false_positive",
                             "true_positive"]

    # ["gold_lines",
    #     "detected_lines", "x_center_of_mass", "y_center_of_mass", "precision",
    #     "recall", "false_negative", "false_positive", "true_positive"]
    _goldLines = None
    _detectedLines = None
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
        if os.path.isfile(csvFileName) == False:
            if self._abftType:
                self._csvHeader.extend(["abft_type", "row_detected_errors", "col_detected_errors"])

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
                          self._header,
                          self._sdcIteration,
                          self._accIteErrors,
                          self._iteErrors, self._goldLines,
            self._detectedLines,
            self._xCenterOfMass,
            self._yCenterOfMass,
            self._precision,
            self._recall,
            self._falseNegative,
            self._falsePositive,
            self._truePositive]

            if self._abftType:
                self._csvHeader.extend([self._abftType, self._rowDetErrors, self._colDetErrors])

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