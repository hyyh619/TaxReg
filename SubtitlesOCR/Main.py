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
import cv2
import sys
import FrameCap
import KeyboardMouse
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


def CheckDuplicatedImg(preImg, imgFile, imgList, counter, logger, threshold=0.06):
    if len(imgList) == 0:
        imgList.append((preImg, imgFile))
        logger.info("img: {}".format(imgFile))
        return False

    logger.info("******************begin********************")
    for item in imgList:
        topLeft, minVal = ImageReg.MatchTemplate(item[0], preImg, threshold)

        if topLeft is not None:
            logger.info("{} --- {}".format(imgFile, item[1]))
            logger.info("{} topLeft: {}, minVal: {}".format(imgFile, topLeft, minVal))

        if topLeft != None:
            break

    logger.info("******************end********************")

    # There is no similar image. We should call OCR.
    if topLeft == None or minVal > threshold:
        return False

    # There is no need to do OCR on this image.
    return True


def PreprocessImg(dataDir, imgFile, counter, bOutput=True):
    img = cv2.imread(imgFile)
    enhanceColor = (255, 255, 255)
    filledColor = (0, 0, 0)
    diffVal = 10
    avgColor = 100

    # Save subtitle only.
    charImg = ImageReg.FilledSubtitleImg(img, avgColor, diffVal, filledColor)

    if bOutput is True:
        name = "{}/subtitle-pre-{}.png".format(dataDir, counter)
        cv2.imwrite(name, charImg)

    return charImg


def GetSubtitlesImg(dataDir, frame, counter, bOutput=True):
    # For 1080p, subtitles in (440, 970, 440+1030, 1030+60)
    # We will get one subtitle image per 25 frames.
    startX = 440
    startY = 970
    charOrigImg = frame[startY:startY+60, startX:startX+1030, 0:3]

    # Output subtitle images.
    if bOutput == True:
        name = "{}/subtitle-{}.png".format(dataDir, counter)
        cv2.imwrite(name, charOrigImg)

    return name


def ProcessSubtitleOnly(imgList, subtitleList, dataDir, frame, counter, ocrReg, logger):
    # Get subtitle image
    subtitleImg = GetSubtitlesImg(dataDir, frame, counter)

    # Check if this subtitle is recognized.
    preImg = PreprocessImg(dataDir, subtitleImg, counter)
    bDuplicate = CheckDuplicatedImg(preImg, subtitleImg, imgList, counter, logger)

    # 1. New image will be recognized.
    # 2. If frame sequence can be divided exactly by 100.
    if bDuplicate is False or (counter % 100) == 0:
        # OCR
        result = ocrReg.GetTextOnly(subtitleImg)

        # If there is subtitle.
        if len(result['words_result']) > 0:
            subtitle = result['words_result'][0]['words']
            subtitleList.append(subtitle)
            logger.info("OCR({}): {}".format(subtitleImg, subtitle))

            # Add this image to imgList for checking duplicated subtitle images.
            logger.info("Add {}".format(subtitleImg))
            imgList.insert(0, (preImg, subtitleImg))


def ProcessFullScreen(subtitleList, dataDir, frame, counter, ocrReg, logger):
    # Write frame to file
    name = "{}/subtitle-{}.jpg".format(dataDir, counter)
    cv2.imwrite(name, frame)

    result = ocrReg.GetTextOnly(name)
    if len(result['words_result']) > 0:
        for text in result['words_result']:
            subtitle = text['words']
            subtitleList.append(subtitle)
            logger.info("OCR({}): {}".format(name, subtitle))


def ProcessVideo(videoFile, ocrReg, logger, bFullScreen = True):
    # create data record directory
    dataDir = './record'
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)

    (path, file) = os.path.split(videoFile)
    (shortname, extname) = os.path.splitext(file)
    dataDir = './record/{0}'.format(shortname)
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)

    subtitleList = []
    imgList = []

    # open video file
    cap=cv2.VideoCapture(videoFile)

    counter = 0
    while (True):
        ret, frame = cap.read()

        if ret == True:
            counter += 1

            # resize displayFrame to 720p
            displayFrame = frame.copy()
            w = frame.shape[1]
            h = frame.shape[0]

            wShow = 800
            hShow = int((float(wShow) / float(w)) * float(h))
            displayFrame = cv2.resize(displayFrame, (wShow, hShow))

            # show image
            cv2.imshow("video", displayFrame)
            key = cv2.waitKey(25)
            if key == ord('q'):
                # cv2.imwrite("frame-1080p.png", frame)
                break

            if (counter % 25) != 0:
                continue

            if bFullScreen is True:
                ProcessFullScreen(subtitleList, dataDir, frame, counter, ocrReg, logger)

                # if counter == 500:
                #     break
            else:
                ProcessSubtitleOnly(imgList, subtitleList, dataDir, frame, counter, ocrReg, logger)
        else:
            break

    name = "{}/{}-subtitle.txt".format(path, shortname)
    with open(name,'w', encoding='utf-8') as f:
        for sentence in subtitleList:
            f.write(sentence)
            f.write('\n')

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
    # Create logger
    logger = CreateLog(LOGGER_CFG_FILE)

    # Video recognize mode.
    bFullScreen = True

    # Parse arguments.
    argc = len(sys.argv)
    if argc < 2:
        logger.err("There is no video input")
    elif argc == 2:
        videoFile = sys.argv[1]
    elif argc == 3:
        videoFile = sys.argv[1]
        if sys.argv[2].find("True") >= 0:
            bFullScreen = True
        else:
            bFullScreen = False

    # Create OCR recognizer.
    ocrReg = Image2OCR.OCRReg(logger)

    # Treat video
    # ProcessVideo("C:/Users/hyyh6/OneDrive/Development/tmp/02.mp4", ocrReg, logger)

    logger.info("Treat video: {}, FullScreen: {}".format(videoFile, bFullScreen))
    ProcessVideo(videoFile, ocrReg, logger, bFullScreen)

if __name__ == '__main__':
    Main()
