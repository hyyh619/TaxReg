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
import AI
import FrameCap
import KeyboardMouse
import cv2
import numpy as np
import Image2OCR
import ImageReg


COMMON_CFG_FILE  = './cfg/common.json'
LOGGER_CFG_FILE  = './cfg/AILog.ini'
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

    logPath = "./log/AI"
    if not os.path.exists(logPath):
        os.mkdir(logPath)

    logging.config.fileConfig(loggerFile)
    logger = logging.getLogger('ai')
    logger.info('Create logger success')
    return logger


def main(logger):
    # Read Common file
    commonCfg1 = CommonCfg(logger, COMMON_CFG_FILE)

    if commonCfg1.IsCommonCfgOK() is False:
        logger.error('Read common file failed.')
        return

    # Create frame capture thread.
    frameGrab = FrameCap.FrameCapture(logger, commonCfg1)
    frameGrab.start()

    # Create controller
    controller = KeyboardMouse.KeyboardMouse(logger, commonCfg1)
    controller.start()

    time.sleep(1)

    # Create AI and run.
    aiAgent = AI.GameAI(commonCfg1, logger, INPUT_EVENT_FILE, controller, frameGrab)
    aiAgent.Run()
    return


def ProcessImgFiles(folder, mode):
    charReg = ImageReg.CharacterReg(logger, './data/WhiteChar')
    agentList = []

    for root, dirs, files in os.walk(folder):
        # print('root_dir:', root)  # 当前目录路径
        # print('sub_dirs:', dirs)  # 当前路径下所有子目录
        # print('files:', files)  # 当前路径下所有非目录子文件

        list = files  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(root, list[i])
            if os.path.isfile(path) and (path.find('png') >= 0 or path.find('bmp') >= 0):
                bPng = path.find('png')
                bBmp = path.find('bmp')

                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                if mode == 1:
                    if img.shape[1] > 1920:
                        if img.shape[0] == 1080:
                            newImg = img[0:1080, 0:1920]
                        elif img.shape[0] == 1201 or img.shape[0] == 1205:
                            newImg = img[0:1080, 0:1920]
                        else:
                            continue

                        (shotname, extension) = os.path.splitext(path)
                        newFilePath = shotname + '.png'
                        cv2.imencode('.png', newImg)[1].tofile(newFilePath)
                    elif path.find('bmp') >= 0:
                        (shotname, extension) = os.path.splitext(path)
                        newFilePath = shotname + '.png'
                        cv2.imencode('.png', img)[1].tofile(newFilePath)
                elif mode == 2:
                    if img.shape[1] == 82 and img.shape[0] == 14:
                        newImg = img[1:13, :]
                        cv2.imencode('.png', newImg)[1].tofile(path)
                elif mode == 3:
                    if path.find('无人') >= 0:
                        continue

                    newImg = ImageReg.CutSpaceFromImg(img)
                    cv2.imencode('.png', newImg)[1].tofile(path)
                elif mode == 4:
                    (shotname, extension) = os.path.splitext(list[i])
                    GenerateChar(charReg, img, shotname, 11, 11)
                elif mode == 5:
                    newImg = FillColor(img, "./data/WhiteChar/test.png")
                    newImg = ImageReg.CutSpaceFromImg(newImg, 20, 'gt')
                    cv2.imwrite('./data/WhiteChar/test1.png', newImg)



def FillColor(img, savedFile): 
    newImg = ImageReg.FillImgWithColor(img, (190, 190, 190), (0, 0, 0))
    cv2.imencode('.png', img)[1].tofile(savedFile)
    return newImg


def GenerateChar(charReg, img, name, minWidth = 8, minHeight = 10):
    dstFolder = './data/WhiteChar'
    nameLen = len(name)
    print (name)

    img = FillColor(img, "./data/WhiteChar/test.png")
    img = ImageReg.CutSpaceFromImg(img, 20, 'gt')

    w = img.shape[1]
    h = img.shape[0]

    src = img
    left = ImageReg.FindLeftNoVerticalSpaceLine(src, 128, 'gt')
    right = 0
    i = 0

    while (1):
        if left >= w or (w - left) < 9:
            break

        # Find character first
        if (left + 17) > w:
            charImg = src[:, left:]
        else:
            charImg = src[:, left:left+17]
        char = charReg.GetCharByRegion(charImg, (0, 0, charImg.shape[1], charImg.shape[0]), None, minWidth, minHeight, 128, 'gt')

        right = left

        charW = ImageReg.FindLeftVerticalSpaceLine(src[:,left:], 128, 'gt')
        if minWidth == 14 and char == '山':
            right += charW
        else:
            while charW < minWidth:
                charW += 1 # skip vertical line, such as '村', '小'
                newLeft = charW + left
                charW += ImageReg.FindLeftVerticalSpaceLine(src[:,newLeft:], 128, 'gt')
            right += charW

        if right > w:
            break

        charImg = src[:,left:right]
        charImg = ImageReg.CutSpaceFromImg(charImg, 128, 'gt')
        if charImg is None:
            break

        if char == '':
            charName = name[i]
            imgFile = '{0}/{1}.png'.format(dstFolder, charName)

            if not os.path.exists(imgFile):
                cv2.imencode('.png', charImg)[1].tofile(imgFile)
            else:
                for i in range(10):
                    imgFile = '{0}/{1}{2}.png'.format(dstFolder, charName, i)
                    if not os.path.exists(imgFile):
                        cv2.imencode('.png', charImg)[1].tofile(imgFile)
                        break

        left += charW
        left += ImageReg.FindLeftNoVerticalSpaceLine(src[:, left:], 128, 'gt')
        i += 1


if __name__ == '__main__':
    # noise_t = np.random.normal(loc=0, scale=0.1, size=7)
    # index = np.argmax(noise_t)
    # Create logger.
    logger = CreateLog(LOGGER_CFG_FILE)

    # main(logger)

    # 1: 把双屏的图像裁剪到1920x1080
    # 2：去掉城名的上下两根空白横线
    # 3：去掉字符串图片的左右两边的空白
    # 4：根据字符串图片生成一个个中文字符
    ProcessImgFiles('./tmp', 1)

    # text = Image2OCR.ProcessOneImgOCR('./tmp/龟山城-大地图.png', logger)

    # Image2OCR.ProcessOCR('./data/Pic', logger)