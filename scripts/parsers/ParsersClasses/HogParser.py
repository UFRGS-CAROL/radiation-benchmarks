import os
from SupportClasses import Rectangle
from SupportClasses import PrecisionAndRecall
from ObjectDetectionParser import ObjectDetectionParser
from ObjectDetectionParser import ImageRaw
import re

LOCAL_GOLD_FOLDER = "/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_K40/histogram_ori_gradients/" #"/home/aluno/parser_hog/gold/"
LOCAL_TXT_FOLDER = "/home/fernando/Dropbox/UFRGS/Pesquisa/Teste_12_2016/GOLD_K40/networks_img_list/"#"/home/aluno/radiation-benchmarks/data/networks_img_list/"
PARAMETERS = "0,1.05,1,1,48,0.9,100"

class HogParser(ObjectDetectionParser):
    _imgListPath = None

    def getBenchmark(self):
        return self._benchmark

    # ERR Rectangles found: 6 (gold has 9).
    # ERR Image: set08_V000_318.jpg
    # ERR 102,51,448,191,499,293
    # ERR 108,54,574,131,628,239
    # ERR 171,86,17,73,103,244
    # ERR 211,105,183,18,288,229
    # ERR 261,130,0,0,130,261
    # ERR 305,153,388,8,541,313
    # ERR MISSED
    # SDC Ite:6220 KerTime:0.036264 AccTime:226.066852 KerErr:2 AccErr:9
    # ABORT amount of errors equals of the last iteration
    # END
    def parseErrMethod(self, errString):
        ret = {}
        try:
            # primeiro o image set
            if 'Image' in errString:
                ret['image_set_name'] = self._parseImageSet(errString)

            # depois se detectou ou nao
            elif 'MISSED' in errString or 'DETECTED' in errString:
                error = re.search(r'\MISSED\b', errString) or re.search(r'\DETECTED\b', errString)
                if error:
                    ret['hardening_result'] = error.group(0)

            # parser da diferenca
            elif 'Rectangle' in errString:
                ret['err_info'] = self._parseErrRectInfo(errString)

            #coordenadas do retangulo
            else:
                ret['rect_cord'] = self._parseRectError(errString)
        except:
            print "Error on HogParser.parseErrHog"
            raise

        return (ret if len(ret) > 0 else None)

    def _parseErrRectInfo(self, errString):
        ret = {}
        error = re.match(".*Rectangles found\: (\d+).*\(gold has (\d+)\).*", errString)
        if error:
            ret['rectangles_found'] = error.group(1)
            ret['rectangles_gold'] = error.group(2)
        return ret

    def _parseRectError(self, errString):
        ret = {}
        error = re.match(".*(\S+),(\S+),(\S+),(\S+),(\S+),(\S+).*", errString)
        if error:
            # r.height, r.width, r.x,	r.y, r.br().x, r.br().y
            ret["r_height"] = error.group(1)
            ret["r_width"] = error.group(2)
            ret["r_x"] = error.group(3)
            ret["r_y"] = error.group(4)
            ret["r_br_x"] = error.group(5)
            ret["r_br_y"] = error.group(6)
        return ret if len(ret) > 0 else None

    # parse only the image set
    # return ret
    # processa o image set dai retorna o que precisar
    def _parseImageSet(self, errString):
        ret = {}
        # ERR Image: set08_V000_318.jpg
        error = re.match(".*Image\: (\S+).*", errString)
        if error:
            ret['set'] = error.group(1)

        return ret if len(ret) > 0 else None

    def _relativeErrorParser(self, errList):
        if len(errList) <= 0:
            return
        images_txt = LOCAL_TXT_FOLDER + os.path.basename(self._imgListPath) # "caltech.pedestrians.critical.1K.txt"

        # o imgFilename tem que ter onde esta a imagem que esta sendo processada
        # para saber qual imagem abrir na darknet eu pegava pelo sdcIteration, que ja e um atributo da classe ObjectDetection
        # imgPos = int(self._sdcIteration) % len(txt das imagens)
        # imgFilename =  listComTxtDasImagens[imgPos]

        imgPos = int(self._sdcIteration) % len(images_txt)
        listFile = open(images_txt).readlines()
        imgFilename = listFile[imgPos]


        # imgObj = ImageRaw(imgFilename)
        # tem que abrir o gold da imagem que voce esta fazendo e colocar no gValidRects ou outra variavel

        #imgObj = ImageRaw(imgFilename)

                      
        # o fValidRects voce tira do errList
        fValidRects = []
        print "\n"
        # primeiro default depois altera
        self._rowDetErrors = self._colDetErrors = 'MISSED'
        self._wrongElements = 0
        for i in errList:
            self._wrongElements += 1
            if 'image_set_name' in i:
                img_set = i['image_set_name']
                current_img = img_set['set']
                print i['image_set_name']
            elif 'hardening_result' in i:
                self._rowDetErrors = self._colDetErrors = i['hardening_result']
            elif 'err_info' in i:
                print i['err_info']

            elif 'rect_cord' in i:
                rect = i['rect_cord']
                # found
                #{'r_x': '91', 'r_y': '76', 'r_br_x': '181', 'r_br_y': '256', 'r_height': '0', 'r_width': '90'}
                lr = int(rect["r_x"])
                br = int(rect["r_y"])
                hr = int(rect["r_height"])
                wr = int(rect["r_width"])
                fValidRects.append(Rectangle.Rectangle(lr, br, wr, hr))

        goldFilename = LOCAL_GOLD_FOLDER + current_img + ".data"
        gold_rectangles = open(goldFilename).readlines()
        
        gValidRects = []
        for rectangle in gold_rectangles:
            rectangle_stripped = rectangle.rstrip()
            if rectangle_stripped != PARAMETERS:
                attributes = rectangle_stripped.split(',')
                width = attributes[1]
                height = attributes[0]
                left = attributes[2]
                bottom = attributes[3]
                rect = Rectangle.Rectangle(int(left), int(bottom), int(width), int(height))
                gValidRects.append(rect)
        print gValidRects

        self._abftType = 'hog_' + self._type
        print fValidRects
        precisionRecallObj = PrecisionAndRecall.PrecisionAndRecall(self._prThreshold)
        gValidSize = len(gValidRects)
        fValidSize = len(fValidRects)

        precisionRecallObj.precisionAndRecallParallel(gValidRects, fValidRects)
        self._precision = precisionRecallObj.getPrecision()
        self._recall = precisionRecallObj.getRecall()

        print self._precision, self._recall

        imgFilename = '/mnt/4E0AEF320AEF15AD/PESQUISA/git_pesquisa/radiation-benchmarks/' + str(imgFilename.split('radiation-benchmarks/')[1]).rstrip()
        self.buildImageMethod(imgFilename, gValidRects, fValidRects)
        self._falseNegative = precisionRecallObj.getFalseNegative()
        self._falsePositive = precisionRecallObj.getFalsePositive()
        self._truePositive = precisionRecallObj.getTruePositive()
        # set all
        self._goldLines = gValidSize
        self._detectedLines = fValidSize
        [self._xCenterOfMass, self._yCenterOfMass] = 0, 0


        # self._xCenterOfMass, self._yCenterOfMass = precisionRecallObj.centerOfMassGoldVsFound(gValidRects, fValidRects,
        #                                                                                       imgObj.w, imgObj.h)


    # HEADER type: ecc_off dataset: /home/carol/radiation-benchmarks/data/networks_img_list/caltech.pedestrians.critical.1K.txt
    def setSize(self, header):
        self._size = None
        m = re.match(".*type\: (\S+).*dataset\: (\S+).*", header)
        if m:
            try:
                self._type = m.group(1)
                self._imgListPath = m.group(2)
            except:
                self.size = None
        self._size = 'hog_' + os.path.basename(self._imgListPath) + '_' + str(self._type)
