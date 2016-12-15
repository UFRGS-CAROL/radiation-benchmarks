
import PrecisionAndRecall as pr

class HogParser(object):
    def __init__(self, **kwargs):
        self.prThreshold = float(kwargs.pop("threshold"))
        self.precisionAndRecall = pr.PrecisionAndRecall(self.prThreshold)



    def parseErrHog(self, errString):

        # ERR boxes: [27,4] e: 132.775177002 r: 132.775024414
        ret = {}



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