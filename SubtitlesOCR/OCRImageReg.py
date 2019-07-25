# ----------------------------------------------
# Project: SengokushiAI
# Author: huang ying
# Date: 2018.11.2
# ----------------------------------------------

import cv2
import json
import time
import numpy as np
import ImageReg


class OCRImagePair:
    def __init__(self, image, ocr):
        self.image = image
        self.ocr   = ocr

class OCRTemplate:
    def __init__(self, name, image, searchRect, tmplRect):
        # rect: left, top, right, bottom
        top = tmplRect[1]
        left = tmplRect[0]
        right = tmplRect[2]
        bottom = tmplRect[3]

        self.tmplImg    = image[top:bottom, left:right]
        self.searchRect = searchRect
        self.tmplRect   = tmplRect
        self.origImg    = image
        self.name       = name

        # Save pic
        # fileName = './tmp/' + name + '.png'
        # cv2.imencode('.png', self.tmplImg)[1].tofile(fileName)

    def GetMatchTmpl(self, frame, fThreshold = 0.01):
        bestMinVal  = 1.0
        bestPos     = (-1, -1, -1, -1)

        tmplImg = self.tmplImg
        rectROI = self.searchRect
        x1      = rectROI[0]
        x2      = rectROI[2]
        y1      = rectROI[1]
        y2      = rectROI[3]

        if x2 >= frame.shape[1]:
            x2 = frame.shape[1] - 1
        if y2 >= frame.shape[0]:
            y2 = frame.shape[0] - 1

        src = frame[y1:y2, x1:x2]

        # cv2.imwrite('src.png', src)
        # cv2.imwrite('tmpl.png', tmplImg)

        topLeft, minVal = ImageReg.MatchTemplate(src, tmplImg, fThreshold)

        if minVal < fThreshold:
            if topLeft is not None:
                bestMinVal  = minVal
                bestPos = (topLeft[0] + x1, topLeft[1] + y1, tmplImg.shape[1], tmplImg.shape[0])

        return bestPos

class OCRImageTmpls:
    def __init__(self, logger, imgFile):
        self.__logger   = logger
        self.__tmplDict = {}
        self.__image    = None
        self.__ocr      = None

        self._LoadFiles(imgFile)

    def _LoadFiles(self, imgFile):
        if imgFile.find('png') < 0 and imgFile.find('bmp') < 0:
            self.__logger.error('Wrong image files: {0}'.format(imgFile))
            return

        jsonFile = '{0}.pos.json'.format(imgFile)
        with open(jsonFile, 'r', encoding='utf-8') as file:
            self.__ocr = json.load(file)

        self.__image = cv2.imdecode(np.fromfile(imgFile, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        if self.__image.shape[2] == 4:
            self.__image = cv2.cvtColor(self.__image, cv2.COLOR_RGBA2RGB)

        searchRect = (self.__ocr['x'],
                      self.__ocr['y'],
                      self.__ocr['x'] + self.__ocr['w'],
                      self.__ocr['y'] + self.__ocr['h'])
        imgNum = self.__ocr['words_result_num']

        for i in range(imgNum):
            item = self.__ocr['words_result'][i]
            tmplRect = (item['location']['left'],
                        item['location']['top'],
                        item['location']['left'] + item['location']['width'],
                        item['location']['top'] + item['location']['height'])
            name = item['words']
            tmpl = OCRTemplate(name, self.__image, searchRect, tmplRect)
            self.__tmplDict[name] = tmpl

    def FindName(self, name):
        for key in self.__tmplDict:
            if key == name:
                return True

        return False

    def GetOperationPos(self, name, frame, fThreshold, sleep):
        if name not in self.__tmplDict:
            return (-1, -1, -1, -1)

        pos = self.__tmplDict[name].GetMatchTmpl(frame, fThreshold)
        if sleep > 0:
            time.sleep(sleep)
        return pos 