# ----------------------------------------------
# Project: SengokushiAI
# Author: huang ying
# Date: 2018.10.31
# ----------------------------------------------

# coding=utf-8

import os
import threading
import logging
import logging.config
import time
import json
import FrameCap
import KeyboardMouse
import cv2
import numpy as np
import Image2OCR
import ImageReg


COMMON_CFG_FILE  = './cfg/common.json'
LOGGER_CFG_FILE  = './cfg/SubtitlesLog.ini'
INPUT_EVENT_FILE = './cfg/SE_1920_1080_InputEvent.json'


class CommonCfg:
    def __init__(self, logger, commonFile):
        self.__logger   = logger
        self.__bSuccess = self._LoadCommonParams(commonFile)
        return

    def _LoadCommonParams(self, commonFile):
        if os.path.exists(commonFile):
            with open(commonFile, 'rb') as file:
                jsonstr = file.read()
                commonCfg = json.loads(str(jsonstr, encoding='utf-8'))
                self.gameName        = commonCfg.get('GameName')
                self.screenX         = commonCfg.get('ScreenCaptureX')
                self.screenY         = commonCfg.get('ScreenCaptureY')
                self.screenWidth     = commonCfg.get('ScreenCaptureWidth')
                self.screenHeight    = commonCfg.get('ScreenCaptureHeight')
                self.showWidth       = commonCfg.get('ShowWidth')
                self.showHeight      = commonCfg.get('ShowHeight')
                self.processWidth    = commonCfg.get('ProcessWidth')
                self.processHeight   = commonCfg.get('ProcessHeight')
                self.timePerFrame    = commonCfg.get('TimePerFrame')
            return True
        else:
            self.__logger.error('No common param file.')
            return False

    def IsCommonCfgOK(self):
        return self.__bSuccess


def CreateLog(loggerFile):
    logPath = "./log"
    if not os.path.exists(logPath):
        os.mkdir(logPath)

    # logPath = "./log/AI"
    # if not os.path.exists(logPath):
    #     os.mkdir(logPath)

    logging.config.fileConfig(loggerFile)
    logger = logging.getLogger('Subtitles')
    logger.info('Create logger success')
    return logger


def Main():
    logger = CreateLog(LOGGER_CFG_FILE)

    ocrReg = Image2OCR.OCRReg(logger)
    img = cv2.imread("./tmp/Test1.png")

    charOrigImg = img[408:408+26, 322:322+164, 0:3]
    # cv2.imshow('image', charOrigImg)
    # k = cv2.waitKey(0)
    # if k == ord('s'):
    #     cv2.imwrite("./tmp/char.png", charImg)

    # black background
    # (158, 158, 156), (159, 159, 159), (174, 174, 174)
    # (166, 166, 166), (81, 86, 79), (97, 98, 107), (162, 152, 151)
    enhanceColor = (255, 255, 255)
    filledColor = (0, 0, 0)
    diffVal = 10
    avgColor = 100

    # Save subtitle only.
    charImg = ImageReg.FilledSubtitleImg(charOrigImg, avgColor, diffVal, filledColor)
    name = "./tmp/subtitle-{}.png".format(diffVal)
    cv2.imwrite(name, charImg)
    result = ocrReg.GetTextOnly("./tmp/charFilled-13.png")
    logger.info(result)


if __name__ == '__main__':
    # noise_t = np.random.normal(loc=0, scale=0.1, size=7)
    # index = np.argmax(noise_t)
    # Create logger.
    # logger = CreateLog(LOGGER_CFG_FILE)

    # main(logger)

    # 1: 把双屏的图像裁剪到1920x1080
    # 2：去掉城名的上下两根空白横线
    # 3：去掉字符串图片的左右两边的空白
    # 4：根据字符串图片生成一个个中文字符
    # ProcessImgFiles('./tmp', 1)

    # text = Image2OCR.ProcessOneImgOCR('./tmp/龟山城-大地图.png', logger)

    # Image2OCR.ProcessOCR('./data/Pic', logger)

    Main()
