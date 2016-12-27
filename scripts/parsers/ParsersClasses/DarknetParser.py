#!/usr/bin/env python
from SupportClasses import Rectangle
import copy
import math
import os
import re
import numpy as np

from ObjectDetectionParser import ObjectDetectionParser

CONFTHRESH = 0.8
NMSTHRESH = 0.3


class DarknetParser(ObjectDetectionParser):
    __executionType = None
    __executionModel = None
    __imgListPath = None
    __imgListFile = None
    __weights = None
    __configFile = None
    __iterations = None

    def setSize(self, header):
        if "abft" in header:
            darknetM = re.match(
                ".*execution_type\:(\S+).*execution_model\:(\S+).*img_list_path\:"
                "(\S+).*weights\:(\S+).*config_file\:(\S+).*iterations\:(\d+).*abft:(\S+).*",
                header)
        else:
            darknetM = re.match(
                ".*execution_type\:(\S+).*execution_model\:(\S+).*img_list_path\:"
                "(\S+).*weights\:(\S+).*config_file\:(\S+).*iterations\:(\d+).*",
                header)

        if darknetM:
            try:
                self.__executionType = darknetM.group(1)
                self.__executionModel = darknetM.group(2)
                self.__imgListPath = darknetM.group(3)
                self.__weights = darknetM.group(4)
                self.__configFile = darknetM.group(5)
                self.__iterations = darknetM.group(6)
                if "abft" in header:
                    self._abftType = darknetM.group(7)
            except:
                self.__imgListFile = None

        # return self.__imgListFile
        self._size = str(self.__imgListFile)

    """
       Compare two sets of boxes



              ret["type"] = "boxes"
               ret["image_list_position"] = image_err.group(1)
               ret["boxes"] = image_err.group(2)
               # x
               ret["x_r"] = image_err.group(3)
               ret["x_e"] = image_err.group(4)
               ret["x_diff"] = image_err.group(5)
               # y
               ret["y_r"] = image_err.group(6)
               ret["y_e"] = image_err.group(7)
               ret["y_diff"] = image_err.group(8)
               # w
               ret["w_r"] = image_err.group(9)
               ret["w_e"] = image_err.group(10)
               ret["w_diff"] = image_err.group(11)
               # h
               ret["h_r"] = image_err.group(12)
               ret["h_e"] = image_err.group(13)
               ret["h_diff"] = image_err.group(14)

       """

    def _relativeErrorParser(self, errList):
        if len(errList) <= 0:
            return

        #set all
        self._goldLines = None
        self._detectedLines = None
        self._xCenterOfMass = None
        self._yCenterOfMass = None
        self._precision = None
        self._recall = None
        self._falseNegative = None
        self._falsePositive = None
        self._truePositive = None

        # only for darknet
        self._abftType = None
        self._rowDetErrors = None
        self._colDetErrors = None

    # parse Darknet
    # returns a dictionary
    def parseErrMethod(self, errString):
        # parse errString for darknet
        ret = {}
        if 'boxes' in errString:
            ret["boxes"] = self.__processBoxes(errString)
        elif 'probs' in errString:
            ret["probs"] = self.__processProbs(errString)
        elif 'INF' in errString:
            ret["abft_det"] = self.__processAbft(errString)

        return ret if len(ret) > 0 else None

    def __processBoxes(self, errString):
        ret = {}
        # ERR image_list_position: [823] boxes: [0]  x_r:
        # 6.7088904380798340e+00 x_e: 6.7152066230773926e+00 x_diff:
        # 6.3161849975585938e-03 y_r: 4.9068140983581543e+00 y_e:
        # 4.9818339347839355e+00 y_diff: 7.5019836425781250e-02 w_r:
        # 2.1113674640655518e+00 w_e: 2.1666510105133057e+00 w_diff:
        # 5.5283546447753906e-02 h_r: 3.4393796920776367e+00 h_e:
        # 3.4377186298370361e+00 h_diff: 1.6610622406005859e-03
        image_err = re.match(
            ".*image_list_position\: \[(\d+)\].*boxes\: \[(\d+)\].*x_r\: (\S+).*x_e\: (\S+).*x_diff\:"
            " (\S+).*y_r\: (\S+).*y_e\: (\S+).*y_diff\: (\S+).*w_r\: (\S+).*w_e\: (\S+).*w_diff\:"
            " (\S+).*h_r\: (\S+).*h_e\: (\S+).*h_diff\: (\S+).*",
            errString)

        if image_err:
            try:

                ret["image_list_position"] = image_err.group(1)
                ret["boxes"] = image_err.group(2)
                # x
                ret["x_r"] = image_err.group(3)
                ret["x_e"] = image_err.group(4)
                ret["x_diff"] = image_err.group(5)
                try:
                    long(float(ret["x_r"]))
                except:
                    ret["x_r"] = 1e30

                try:
                    long(float(ret["x_e"]))
                except:
                    ret["x_e"] = 1e30

                try:
                    long(float(ret["x_diff"]))
                except:
                    ret["x_diff"] = 1e30

                # y
                ret["y_r"] = image_err.group(6)
                ret["y_e"] = image_err.group(7)
                ret["y_diff"] = image_err.group(8)
                try:
                    long(float(ret["y_r"]))
                except:
                    ret["y_r"] = 1e30

                try:
                    long(float(ret["y_e"]))
                except:
                    ret["y_e"] = 1e30

                try:
                    long(float(ret["y_diff"]))
                except:
                    ret["y_diff"] = 1e30


                # w
                ret["w_r"] = image_err.group(9)
                ret["w_e"] = image_err.group(10)
                ret["w_diff"] = image_err.group(11)
                try:
                    long(float(ret["w_r"]))
                except:
                    ret["w_r"] = 1e30

                try:
                    long(float(ret["w_e"]))
                except:
                    ret["w_e"] = 1e30

                try:
                    long(float(ret["w_diff"]))
                except:
                    ret["w_diff"] = 1e30

                # h
                ret["h_r"] = image_err.group(12)
                ret["h_e"] = image_err.group(13)
                ret["h_diff"] = image_err.group(14)
                try:
                    long(float(ret["h_r"]))
                except:
                    ret["h_r"] = 1e30

                try:
                    long(float(ret["h_e"]))
                except:
                    ret["h_e"] = 1e30

                try:
                    long(float(ret["h_diff"]))
                except:
                    ret["h_diff"] = 1e30
            except:
                print "Error on parsing boxes"
                raise

        return ret if len(ret) > 0 else None

    def __processProbs(self, errString):
        ret = {}
        image_err = re.match(
            ".*image_list_position\: \[(\d+)\].*probs\: \[(\d+),"
            "(\d+)\].*prob_r\: ([0-9e\+\-\.]+).*prob_e\: ([0-9e\+\-\.]+).*", errString)
        if image_err:
            try:
                ret["image_list_position"] = image_err.group(1)
                ret["probs_x"] = image_err.group(2)
                ret["probs_y"] = image_err.group(3)
                ret["prob_r"] = image_err.group(4)
                ret["prob_e"] = image_err.group(5)
            except:
                print "Error on parsing probs"
                raise

        return ret if len(ret) > 0 else None

    def __processAbft(self, errString):
        # INF abft_type: dumb image_list_position: [151] row_detected_errors: 1 col_detected_errors: 1
        m = re.match(".*abft_type\: (\S+).*image_list_position\: \[(\d+)\].*row_detected_errors\:"
            "(\d+).*col_detected_errors\: (\d+).*", errString)

        ret = {}
        if m:
            try:
                ret["abft_type"] = str(m.group(1))
                ret["img_list_position"] = str(m.group(2))
                ret["row_detected_errors"] = int(m.group(3))
                ret["col_detected_errors"] = int(m.group(4))
            except:
                print "Error on parsing abft info"
                raise
        return ret if len(ret) > 0 else None

    def buildImageMethod(self):
        return False