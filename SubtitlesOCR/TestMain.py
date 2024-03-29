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


def TestImgProcess(ocrReg, logger):
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


def CheckDuplicatedImg(imgDir, logger, threshold=0.45, saveDir="./tmp/02-subtitles-NoDuplicated-pre-0.45-1030x60/"):
    counter = 0
    imgList = []
    for root, dirs, files in os.walk(imgDir):
        list = files  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(root, list[i])
            (shortname, extname) = os.path.splitext(list[i])
            if os.path.isfile(path) is not True:
                continue

            if path.find('png') < 0:
                continue

            img = cv2.imread(path)
            if len(imgList) == 0:
                imgList.append((img, list[i]))
                logger.info("img: {}".format(list[i]))
            else:
                logger.info("******************begin********************")
                for item in imgList:
                    topLeft, minVal = ImageReg.MatchTemplate(item[0], img, threshold)

                    if topLeft is not None:
                        logger.info("{} --- {}".format(list[i], item[1]))
                        logger.info("{} topLeft: {}, minVal: {}".format(list[i], topLeft, minVal))

                    if topLeft != None:
                        break

                if topLeft == None or minVal > threshold:
                    #imgList.append((img, list[i]))
                    logger.info("Add {}".format(list[i]))
                    imgList.insert(0, (img, list[i]))
                logger.info("******************end********************")

    for item in imgList:
        name = "{}/{}".format(saveDir, item[1])
        cv2.imwrite(name, item[0])
        logger.info("Save: {}".format(name))


def PreprocessImg(imgDir):
    counter = 0
    for root, dirs, files in os.walk(imgDir):
        list = files  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(root, list[i])
            (shortname, extname) = os.path.splitext(list[i])
            if os.path.isfile(path) is not True:
                continue

            if path.find('png') < 0:
                continue

            img = cv2.imread(path)
            enhanceColor = (255, 255, 255)
            filledColor = (0, 0, 0)
            diffVal = 10
            avgColor = 100

            # Save subtitle only.
            charImg = ImageReg.FilledSubtitleImg(img, avgColor, diffVal, filledColor)
            name = "./tmp/02-subtitles-pre/02-subtitle-pre-{}.png".format(counter)
            counter += 1
            cv2.imwrite(name, charImg)


def GetSubtitlesImg(frame, counter):
    # For 1080p, subtitles in (440, 970, 440+1030, 1030+60)
    # We will get one subtitle image per 25 frames.
    if (counter % 25) != 0:
        return

    startX = 440
    startY = 970
    charOrigImg = frame[startY:startY+60, startX:startX+1030, 0:3]
    name = "./tmp/02-subtitles/02-subtitle-{}.png".format(counter)
    cv2.imwrite(name, charOrigImg)
    return


def ProcessVideo(videoFile, ocrReg, logger):
    cap=cv2.VideoCapture(videoFile)

    counter = 0
    while (True):
        ret, frame = cap.read()
    
        if ret == True:
            counter += 1

            # Get subtitles
            GetSubtitlesImg(frame, counter)

            # resize displayFrame to 720p
            displayFrame = frame.copy()
            w = frame.shape[1]
            h = frame.shape[0]

            wShow = 800
            hShow = int((float(wShow) / float(w)) * float(h))
            displayFrame = cv2.resize(displayFrame, (wShow, hShow))

            cv2.imshow("video", displayFrame)
            key = cv2.waitKey(25)
            if key == ord('q'):
                # cv2.imwrite("frame-1080p.png", frame)
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def ProcessOCR(imgDir, ocrReg, logger):
    for root, dirs, files in os.walk(imgDir):
        list = files  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(root, list[i])
            (shortname, extname) = os.path.splitext(list[i])
            if os.path.isfile(path) is not True:
                continue

            if path.find('png') < 0:
                continue

            result = ocrReg.GetTextOnly(path)

            if len(result['words_result']) > 0:
                name = "{}-{}.png".format(shortname, result['words_result'][0]['words'])
            else:
                name = "{}-none.png".format(shortname)
            logger.info(name)

            fileName = "./tmp/02-subtitles-OCR/{}".format(name)
            img = cv2.imread(path)
            cv2.imencode('.png', img)[1].tofile(fileName)


def Main():
    logger = CreateLog(LOGGER_CFG_FILE)
    ocrReg = Image2OCR.OCRReg(logger)

    # Test
    # TestImgProcess(ocrReg, logger)

    # Treat video
    # ProcessVideo("E:/Development/Data/TaxVideos/02.mp4", ocrReg, logger)
    # PreprocessImg("./tmp/02-subtitles/")

    # Check duplicated images
    # CheckDuplicatedImg("./tmp/02-subtitles/", logger, 0.1, "./tmp/02-subtitles-NoDuplicated-0.1-1030x60")

    # OCR
    ProcessOCR("tmp/02-subtitles", ocrReg, logger)

    # 02-subtitle-4375-我们会把剩下的三项内容讲完.png
    # img = cv2.imread("./tmp/02-subtitles/02-subtitle-25.png")
    # name = u"./tmp/02-subtitle-4375-我们会把剩下的三项内容讲完.png"
    # cv2.imencode('.png', img)[1].tofile(name)

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
