


class FasterRcnnParser(object):
    # parse PyFaster
    def parseErrPyFaster(self,errString, imgIndex):
        # ERR boxes: [27,4] e: 132.775177002 r: 132.775024414
        ret = {}
        if 'boxes' in errString:
            image_err = re.match(
                ".*boxes\: \[(\d+),(\d+)\].*e\: (\S+).*r\: (\S+).*",
                errString)
            if image_err:
                ret["type"] = "boxes"
                ret["imgindex"] = imgIndex
                ###########
                ret["boxes_x"] = image_err.group(1)
                try:
                    long((ret["boxes_x"]))
                except:
                    ret["boxes_x"] = 1e30
                ###########
                ret["boxes_y"] = image_err.group(2)
                try:
                    long((ret["boxes_y"]))
                except:
                    ret["boxes_y"] = 1e30
                ###########
                ret["e"] = image_err.group(3)
                try:
                    long(float(ret["e"]))
                except:
                    ret["e"] = 1e30
                ############
                ret["r"] = image_err.group(4)
                try:
                    long(float(ret["r"]))
                except:
                    ret["r"] = 1e30


                    # try:
                    #     long(float(ret["x_r"]))
                    # except:
                    #     ret["x_r"] = 1e30
        # else:
        #     image_err = re.match(".*probs\: \[(\d+),(\d+)\].*e\: ([0-9e\+\-\.]+).*r\: ([0-9e\+\-\.]+).*",
        #                          errString)
        #     if image_err:
        #         ret["type"] = "probs"
        #         ret["probs_x"] = image_err.group(1)
        #         ret["probs_y"] = image_err.group(2)
        #         ret["prob_e"] = image_err[3]
        #         ret["prob_r"] = image_err[4]

        return (ret if len(ret) > 0 else None)

    def generatePyFasterRectangles(self, dets, thresh=0):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]

        if len(inds) == 0:
            return

        # im = im[:, :, (2, 1, 0)]
        # fig, ax = plt.subplots(figsize=(12, 12))
        # ax.imshow(im, aspect='equal')
        bboxList = []
        scoresList = []
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            scoresList.append(score)
            # left = int(math.floor(float(i["x_r"])))
            # bottom = int(math.floor(float(i["y_r"])))
            # h = int(math.ceil(float(i["h_r"])))
            # w = int(math.ceil(float(i["w_r"])))
            # tempBoxes[boxPos] = Rectangle(left, bottom, w, h)
            left = int(math.floor(float(bbox[0])))
            bottom = int(math.floor(float(bbox[1])))
            w = int(math.ceil(bbox[2] - bbox[0]))
            h = int(math.ceil(bbox[3] - bbox[1]))
            bboxList.append(Rectangle(left, bottom, w, h))

        # ax.add_patch(
        #                           X       Y
        #         plt.Rectangle((bbox[0], bbox[1]),
        #                         Xmax   -   Xmin
        #                       bbox[2] - bbox[0],
        #                         Ymax   -  Ymin
        #                       bbox[3] - bbox[1], fill=False,
        #                       edgecolor='red', linewidth=3.5)
        #         )
        #     ax.text(bbox[0], bbox[1] - 2,
        #             '{:s} {:.3f}'.format(class_name, score),
        #             bbox=dict(facecolor='blue', alpha=0.5),
        #             fontsize=14, color='white')
        #
        # ax.set_title(('{} detections with '
        #               'p({} | box) >= {:.1f}').format(class_name, class_name,
        #                                               thresh),
        #               fontsize=14)

        return scoresList, bboxList

    """
    ret["type"] = "boxes"
    ret["imgindex"] = imgIndex
    ###########
    ret["boxes_x"] = image_err.group(1)

    ###########
    ret["boxes_y"] = image_err.group(2)
    ###########
    ret["e"] = image_err.group(3)
    ############
    ret["r"] = image_err.group(4)
    """

    def relativeErrorParserPyFaster(self, img_list_path, errList, gold_obj, sdcIte):
        if len(errList) <= 0:
            return ("errlist fucked", None, None, None, None, None, None, None, None, None)

        goldPyfaster = gold_obj.pyFasterGold
        img_list = open(img_list_path, "r").readlines()
        imgLPos = getImgLPos(sdcit=sdcIte, maxsize=len(
            goldPyfaster.keys()))  # getImgLPos(errList=errList, cnn="pyfaster", sdcit=sdcIte, maxsize=len(img_list))
        imgFile = img_list[imgLPos].rstrip()
        gold = goldPyfaster[imgFile]

        goldArray = getCls(gold)

        tempArray = copyList(goldArray, 'pyfaster')

        print "Gold array size ", len(tempArray)
        # for i in tempArray:
        #     print len(i)

        for i in errList:
            x = long(i["boxes_x"])
            y = long(i["boxes_y"])
            print i["boxes_x"], i["boxes_y"]
            # print "vet size ", len(goldArray)
            # print "x size ", len (tempArray[x])
            # tempArray[x][y] = float(i["r"])

        goldRectangles = generatePyFasterRectangles(goldArray)
        tempRectangles = generatePyFasterRectangles(tempArray)

        pR = PrecisionAndRecall(0.5)
        pR.precisionAndRecallParallel(goldRectangles, tempRectangles)

        return (
            len(gold), len(tempRectangles), 0, 0, pR.getPrecision(), pR.getRecall(), pR.getFalseNegative(),
            pR.getFalsePositive(),
            pR.getTruePositive(), imgFile)

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

    def relativeErrorParserDarknet(self, img_list_path, errList, gold_obj, sdcIt):
        if len(errList) <= 0:
            return ("errlist fucked", None, None, None, None, None, None, None, None, None)

        imgList = open(img_list_path, "r").readlines()

        imgLPos = getImgLPos(sdcit=sdcIt, maxsize=gold_obj.plist_size)

        # print "\nTamanho do plist " , gold_obj.plist_size , " tamanho do imgLPos" , imgLPos
        gold = gold_obj.prob_array["boxes"][imgLPos]
        tempBoxes = copyList(gold)
        # {'x_e': '2.3084202575683594e+02', 'h_r': '4.6537536621093750e+01', 'x_diff': '0.0000000000000000e+00',
        #  'w_diff': '3.8146972656250000e-06', 'y_r': '2.5291372680664062e+02', 'y_diff': '0.0000000000000000e+00',
        #  'w_e': '3.0895418167114258e+01', 'boxes': '94', 'h_e': '4.6537536621093750e+01', 'x_r': '2.3084202575683594e+02',
        #  'h_diff': '0.0000000000000000e+00', 'y_e': '2.5291372680664062e+02', 'w_r': '3.0895414352416992e+01',
        #  'type': 'boxes', 'image_list_position': '314'}
        for i in errList:
            if i["type"] == "boxes":
                boxPos = int(i['boxes'])
                left = int(math.floor(float(i["x_r"])))
                bottom = int(math.floor(float(i["y_r"])))
                h = int(math.ceil(float(i["h_r"])))
                w = int(math.ceil(float(i["w_r"])))
                tempBoxes[boxPos] = Rectangle(left, bottom, w, h)

        # writeTempFileCXX("/tmp/temp_test.log", tempBoxes)
        pR = PrecisionAndRecall(0.5)
        imgFile = imgList[imgLPos]
        # pR.precisionAndRecallSerial(gold, tempBoxes)
        p, r = pR.precisionRecall(gold, tempBoxes, 0.5)
        # sizX, sizY = getImageSize((GOLD_DIR + imgFile).rstrip("\n"))
        # start = time.clock()
        x, y = 0, 0  # centerOfMassGoldVsFound(gold,tempBoxes,sizX, sizY)
        # print time.clock() - start
        return (
        len(gold), len(tempBoxes), x, y, p, r, pR.getFalseNegative(), pR.getFalsePositive(), pR.getTruePositive(),
        imgFile)

    def py_cpu_nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
