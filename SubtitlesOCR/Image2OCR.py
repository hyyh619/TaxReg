# ----------------------------------------------
# Project: SengokushiAI
# Author: huang ying
# Date: 2018.11.1
# ----------------------------------------------

import os
import json
import cv2
import numpy as np
from aip import AipOcr


BAIDU_OCR_APP_ID      = '14641603'
BAIDU_OCR_API_KEY     = 'gyfGXWZ0fCswzhM2Nd2HyDDH'
BAIDU_OCR_SECRET_KEY  = 'R1kR4Nq1hBaoLFomcvHYnCv4pxmKFI8Z'


class OCRReg:
    def __init__(self, logger):
        self.__client = AipOcr(BAIDU_OCR_APP_ID, BAIDU_OCR_API_KEY, BAIDU_OCR_SECRET_KEY)
        self.__logger = logger

    def _GetFileContent(self, filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    def GetResult(self, imgFile):
        img = self._GetFileContent(imgFile)

        """ 如果有可选参数 """
        options = {}
        options["language_type"] = "CHN_ENG"
        # options["detect_direction"] = "true"  # 图像朝向
        options["detect_language"] = "true"
        options["probability"] = "true"

        """ 带参数调用通用文字识别, 图片参数为本地图片 """
        resultLow     = self.__client.basicGeneral(img, options)
        self.__logger.info('Process basic img: {0}'.format(imgFile))

        #resultHigh    = self.__client.basicAccurate(img)
        resultHigh = None

        resultHighPos = self.__client.accurate(img)
        self.__logger.info('Process accurate img: {0}'.format(imgFile))
        #resultHighPos = None

        return resultLow, resultHigh, resultHighPos

    def GetTextOnly(self, cv2Img, fillColor = 255):
        imgFile = './tmp/text.png'
        if cv2Img.shape[0] < 15 and cv2Img.shape[1] < 15:
            w = cv2Img.shape[1]
            h = cv2Img.shape[0]
            newImg = np.zeros((h + 10, w+10, 3), dtype=np.uint8)
            for i in range(h+10):
                for j in range(w+10):
                    newImg[i][j] = fillColor

            newImg[5:h+5, 5:w+5] = cv2Img[0:h, 0:w]
        elif cv2Img.shape[0] < 15:
            w = cv2Img.shape[1]
            h = cv2Img.shape[0]
            newImg = np.zeros((h + 10, w, 3), dtype=np.uint8)
            for i in range(h+10):
                for j in range(w):
                    newImg[i][j] = fillColor

            newImg[5:h+5, 0:w] = cv2Img[0:h, 0:w]
        elif cv2Img.shape[1] < 15:
            w = cv2Img.shape[1]
            h = cv2Img.shape[0]
            newImg = np.zeros((h, w + 10, 3), dtype=np.uint8)
            for i in range(h):
                for j in range(w+10):
                    newImg[i][j] = fillColor

            newImg[0:h, 5:w+5] = cv2Img[0:h, 0:w]
        else:
            newImg = cv2Img

        cv2.imwrite(imgFile, newImg)
        img = self._GetFileContent(imgFile)

        """ 如果有可选参数 """
        options = {}
        options["language_type"] = "CHN_ENG"
        options["detect_language"] = "true"
        options["probability"] = "true"
        # options["detect_direction"] = "true"  # 图像朝向

        """ 带参数调用通用文字识别, 图片参数为本地图片 """
        # resultLow     = self.__client.basicGeneral(img, options)
        # self.__logger.info('Process basic img: {0}'.format(imgFile))

        resultHigh = self.__client.basicAccurate(img)
        # resultHigh = None

        return resultHigh


def GetCharFromOCR(imgFile, ocr):
    result, resultHigh, resultHighPos = ocr.GetResult(imgFile)

    jsonFileName = '{0}.json'.format(imgFile)
    file = open(jsonFileName, 'w', encoding='utf-8')
    file.write(json.dumps(result, ensure_ascii=False, indent=2))
    file.flush()
    file.close()

    if resultHighPos is None:
        return

    jsonFileName = '{0}.pos.json'.format(imgFile)
    file = open(jsonFileName, 'w', encoding='utf-8')
    file.write(json.dumps(resultHighPos, ensure_ascii=False, indent=2))
    file.flush()
    file.close()
    return


def ProcessOCR(folder, logger):
    ocr = OCRReg(logger)

    for root, dirs, files in os.walk(folder):
        # print('root_dir:', root)  # 当前目录路径
        # print('sub_dirs:', dirs)  # 当前路径下所有子目录
        # print('files:', files)  # 当前路径下所有非目录子文件

        list = files  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(root, list[i])
            if os.path.isfile(path) and (path.find('png') >= 0 or path.find('bmp') >= 0):

                if path.find('swp') >= 0 or path.find('json') >= 0:
                    continue

                GetCharFromOCR(path, ocr)

def ProcessOneImgOCR(imgFile, logger):
    ocr = OCRReg(logger)
    cv2Img = cv2.imdecode(np.fromfile(imgFile, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    result = ocr.GetTextOnly(cv2Img)