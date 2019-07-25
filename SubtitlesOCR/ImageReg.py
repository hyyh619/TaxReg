# ----------------------------------------------
# Project: SengokushiAI
# Author: huang ying
# Date: 2018.10.31
# ----------------------------------------------

import os
import cv2
import numpy as np
import math
from GameState import *


# All the 6 methods for comparison in a list
TEMPLATE_MATCH_METHODS = [
    cv2.TM_CCOEFF,
    cv2.TM_CCOEFF_NORMED,
    cv2.TM_CCORR,
    cv2.TM_CCORR_NORMED,
    cv2.TM_SQDIFF,
    cv2.TM_SQDIFF_NORMED
]

def MatchTemplate(src, temp, fThreshold, method = TEMPLATE_MATCH_METHODS[5]):
    # Using TM_SQDIFF_NORMED
    res = cv2.matchTemplate(src, temp, method)
    # find the min and max values
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    topLeft = None
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        if min_val < fThreshold:
            topLeft = min_loc
    return topLeft, min_val

def MatchMultiTemplate(src, temp, fThreshold, method = TEMPLATE_MATCH_METHODS[3]):
    locList = []
    w, h = temp.shape[1], temp.shape[0]

    # Using TM_SQDIFF_NORMED
    res = cv2.matchTemplate(src, temp, method)
    loc = np.where(res >= fThreshold)

    for pt in zip(*loc[::-1]):
        locList.append(pt)
        cv2.rectangle(src, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    # cv2.imwrite('tmpl.png', temp)
    # cv2.imwrite('res.png', src)
    return locList

# space mode: define how to check space color.
# 'lt' means reserved color is less than space color
# 'gt' means reserved color is great than space color
def CutSpaceFromImg(img, spaceColor=240, spaceMode='lt'):
    w = img.shape[1]
    h = img.shape[0]

    # Find Left reserved color
    bFound = False
    l = 0
    for l in range(w):
        for j in range(h):
            pixel = img[j][l]
            if spaceMode == 'lt':
                if pixel[0] < spaceColor:
                    bFound = True
                    break
            elif spaceMode == 'gt':
                if pixel[0] > spaceColor:
                    bFound = True
                    break

        if bFound is True:
            break

    # Find the most right reserved color.
    bFound = False
    r = 0
    for r in range(w-1, -1, -1):
        for j in range(h):
            pixel = img[j][r]
            if spaceMode == 'lt':
                if pixel[0] < spaceColor:
                    bFound = True
                    break
            elif spaceMode == 'gt':
                if pixel[0] > spaceColor:
                    bFound = True
                    break

        if bFound is True:
            break

    # Find the top reserved color
    bFound = False
    t = 0
    for t in range(h):
        for j in range(w):
            pixel = img[t][j]
            if spaceMode == 'lt':
                if pixel[0] < spaceColor:
                    bFound = True
                    break
            elif spaceMode == 'gt':
                if pixel[0] > spaceColor:
                    bFound = True
                    break

        if bFound is True:
            break

    # Find the bottom reserved color
    bFound = False
    b = 0
    for b in range(h-1, -1, -1):
        for j in range(w):
            pixel = img[b][j]
            if spaceMode == 'lt':
                if pixel[0] < spaceColor:
                    bFound = True
                    break
            elif spaceMode == 'gt':
                if pixel[0] > spaceColor:
                    bFound = True
                    break

        if bFound is True:
            break

    if t == b or (b+1) > h:
        return img

    if r == l or (r+1) > w:
        return img

    newImg = img[t:b+1, l:r+1]
    return newImg

# space mode: define how to check space color.
# 'lt' means reserved color is less than space color
# 'gt' means reserved color is great than space color
def FindLeftNoVerticalSpaceLine(img, spaceColor = 230, spaceMode = 'lt'):
    w = img.shape[1]
    h = img.shape[0]

    bFound = False
    i = 0
    for i in range(w):
        for j in range(h):
            pixel = img[j][i]
            if spaceMode == 'lt':
                if pixel[0] < spaceColor:
                    bFound = True
                    break
            elif spaceMode == 'gt':
                if pixel[0] > spaceColor:
                    bFound = True
                    break

        if bFound is True:
            break
    return i

# space mode: define how to check space color.
# 'lt' means reserved color is less than space color
# 'gt' means reserved color is great than space color
def FindLeftVerticalSpaceLine(img, spaceColor = 230, spaceMode = 'lt'):
    w = img.shape[1]
    h = img.shape[0]

    bFound = False
    for i in range(w):
        for j in range(h):
            pixel = img[j][i]
            if spaceMode == 'lt':
                if pixel[0] < spaceColor:
                    j = 0
                    break
            elif spaceMode == 'gt':
                if pixel[0] > spaceColor:
                    j = 0
                    break

        if j == (h-1):
            bFound = True
            break

    if bFound is True:
        return i
    else:
        return w

# color: if color of pixel is greater color, we enhance this pixel.
def FillImgWithColor(img, color, filledColor, enhanceColor = (255, 255, 255)):
    w = img.shape[1]
    h = img.shape[0]

    total = w * h
    enhancedNum = 0
    for i in range(w):
        for j in range(h):
            pixel = img[j][i]
            if pixel[0] > color[0] and pixel[1] > color[1] and pixel[2] > color[2]:
                img[j][i][0:3] = enhanceColor
                enhancedNum += 1
            else:
                img[j][i][0:3] = filledColor

    ratio = float(enhancedNum) / float(total)
    return img, ratio


class TemplateImage:
    def __init__(self, fileName, imgWRatio, imgHRatio, ROI = None):
        self.img = None
        self.ROI = ROI
        self._LoadImg(fileName, imgWRatio, imgHRatio)

    def _LoadImg(self, fileName, imgWRatio, imgHRatio):
        # self.img = cv2.imread(fileName)
        self.img = cv2.imdecode(np.fromfile(fileName, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        if self.img.shape[2] == 4:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGBA2RGB)

        self.img = self._ScaleImgToProcessSize(self.img, imgWRatio, imgHRatio)

    def _ScaleImgToProcessSize(self, img, imgWRatio, imgHRatio):
        procW = int(img.shape[1] * imgWRatio)  # width
        procH = int(img.shape[0] * imgHRatio)  # height
        img = cv2.resize(img, (procW, procH))
        return img


class ImgData:
    def __init__(self, fileName, ROI, state):
        self.fileName = fileName
        self.ROI      = ROI
        self.state    = state

class ImageReg:
    def __init__(self, logger, imgDataList, processWidth=1920, processHeight=1080):
        self.__logger = logger
        self.__processWidth  = processWidth
        self.__processHeight = processHeight

        # original picture is captured from 1920x1080
        self.__origWidth  = 1920
        self.__origHeight = 1080
        self.__imgWRatio  = float(self.__processWidth) / float(self.__origWidth)
        self.__imgHRatio  = float(self.__processHeight) / float(self.__origHeight)

        self._LoadResources(imgDataList)
        return

    def _ScaleImgToProcessSize(self, img):
        procW = int(img.shape[1] * self.__imgWRatio) # width
        procH = int(img.shape[0] * self.__imgHRatio) # height
        img = cv2.resize(img, (procW, procH))
        return img

    def _LoadResources(self, imgDataList):
        self.__stageTmpl = {}

        for imgData in imgDataList:
            ROI = imgData.ROI
            fileName = imgData.fileName
            state = imgData.state

            self.__stageTmpl[state] = TemplateImage(fileName, self.__imgWRatio, self.__imgHRatio, ROI)
            self.__logger.info("load tmpl file: {0}, state: {1}".format(fileName, state))

    def AddImageData(self, fileName, state, ROI = None):
        self.__stageTmpl[state] = TemplateImage(fileName, self.__imgWRatio, self.__imgHRatio, ROI)
        self.__logger.info("load tmpl file: {0}, state: {1}".format(fileName, state))

    def GetCurrentStage(self, frame, fThreshold = 0.01):
        bestTopLeft = None
        bestMinVal  = 1.0
        bestStage   = GAME_STAGE_UNKNOWN

        for key in self.__stageTmpl:
            stageTmpl = self.__stageTmpl[key]
            rectROI   = stageTmpl.ROI
            x1        = rectROI[0]
            x2        = rectROI[0] + rectROI[2]
            y1        = rectROI[1]
            y2        = rectROI[1] + rectROI[3]

            if x2 >= frame.shape[1]:
                x2 = frame.shape[1] - 1
            if y2 >= frame.shape[0]:
                y2 = frame.shape[0] - 1

            src = frame[y1:y2, x1:x2]
            #cv2.imwrite('hy.jpg', src)

            if src.shape[0] < stageTmpl.img.shape[0] or \
               src.shape[1] < stageTmpl.img.shape[1] or \
               src.shape[2] != stageTmpl.img.shape[2]:
               continue

            # savedStageImg = frame[49:73, 1741:1912]
            # cv2.imwrite('stage.png', savedStageImg)

            topLeft, minVal = MatchTemplate(src, stageTmpl.img, 0.01)

            if minVal < bestMinVal:
                if topLeft is not None:
                    bestTopLeft = topLeft
                    bestStage   = key
                    bestMinVal  = minVal

        return bestStage, bestTopLeft

    def GetCurrentStageByROI(self, frame, ROI, fThreshold = 0.01):
        bestTopLeft = None
        bestMinVal  = 1.0
        bestStage   = GAME_STAGE_UNKNOWN

        for key in self.__stageTmpl:
            stageTmpl = self.__stageTmpl[key]
            rectROI   = ROI
            x1        = rectROI[0]
            x2        = rectROI[0] + rectROI[2]
            y1        = rectROI[1]
            y2        = rectROI[1] + rectROI[3]

            if x2 >= frame.shape[1]:
                x2 = frame.shape[1] - 1
            if y2 >= frame.shape[0]:
                y2 = frame.shape[0] - 1

            src = frame[y1:y2, x1:x2]
            # cv2.imwrite('hy.jpg', src)

            if src.shape[0] < stageTmpl.img.shape[0] or \
               src.shape[1] < stageTmpl.img.shape[1] or \
               src.shape[2] != stageTmpl.img.shape[2]:
               continue

            # savedStageImg = frame[49:73, 1741:1912]
            # cv2.imwrite('stage.png', savedStageImg)

            topLeft, minVal = MatchTemplate(src, stageTmpl.img, 0.01)
            # if key == '尼子家':
            #     cv2.imwrite('src.png', src)
            #     cv2.imwrite('tmpl.png', stageTmpl.img)

            if minVal < bestMinVal:
                if topLeft is not None:
                    bestTopLeft = topLeft
                    bestStage   = key
                    bestMinVal  = minVal

        return bestStage, bestTopLeft

    def FindBestTmplByState(self, frame, state, fThreshold = 0.01):
        stageTmpl = self.__stageTmpl[state]
        rectROI   = stageTmpl.ROI
        x1        = rectROI[0]
        x2        = rectROI[0] + rectROI[2]
        y1        = rectROI[1]
        y2        = rectROI[1] + rectROI[3]

        if x2 >= frame.shape[1]:
            x2 = frame.shape[1] - 1
        if y2 >= frame.shape[0]:
            y2 = frame.shape[0] - 1

        src = frame[y1:y2, x1:x2]

        topLeft, minVal = MatchTemplate(src, stageTmpl.img, fThreshold)

        return minVal, topLeft, rectROI

    def FindAllTmplPosByState(self, frame, state, fThreshold = 0.99):
        posList = []
        stageTmpl = self.__stageTmpl[state]
        rectROI   = stageTmpl.ROI
        x1        = rectROI[0]
        x2        = rectROI[0] + rectROI[2]
        y1        = rectROI[1]
        y2        = rectROI[1] + rectROI[3]

        if x2 >= frame.shape[1]:
            x2 = frame.shape[1] - 1
        if y2 >= frame.shape[0]:
            y2 = frame.shape[0] - 1

        src = frame[y1:y2, x1:x2]

        topLeftList = MatchMultiTemplate(src, stageTmpl.img, fThreshold)
        for pos in topLeftList[::-1]:
            x = pos[0] + x1
            y = pos[1] + y1
            posList.append([x, y])

        return posList

    def FindTmplPosByStateAndROI(self, frame, state, ROI, fThreshold = 0.99):
        stageTmpl = self.__stageTmpl[state]
        rectROI   = ROI
        x1        = rectROI[0]
        x2        = rectROI[0] + rectROI[2]
        y1        = rectROI[1]
        y2        = rectROI[1] + rectROI[3]

        if x2 >= frame.shape[1]:
            x2 = frame.shape[1] - 1
        if y2 >= frame.shape[0]:
            y2 = frame.shape[0] - 1

        src = frame[y1:y2, x1:x2]

        topLeft, minVal = MatchTemplate(src, stageTmpl.img, fThreshold)

        return topLeft, minVal


class DigitalReg:
    def __init__(self, logger, digitalFileList, digitalPosList):
        self.__logger = logger
        self.__digitalPos = digitalPosList
        self._LoadDigitals(digitalFileList)

    def _LoadDigitals(self, digitalFileList):
        self.__digitalTmpl = {}

        for imgData in digitalFileList:
            fileName = imgData.fileName
            state = imgData.state

            self.__digitalTmpl[state] = TemplateImage(fileName, 1.0, 1.0)
            self.__logger.info("load digital file: {0}, state: {1}".format(fileName, state))

        self.__digitalLen = len(self.__digitalPos)

    def _GetOneDigital(self, pos, frame):
        bestTopLeft = None
        bestMinVal  = 1.0
        bestStage   = 10

        for key in self.__digitalTmpl:
            stageTmpl = self.__digitalTmpl[key]
            rectROI   = pos
            x1        = rectROI[0]
            x2        = rectROI[0] + rectROI[2]
            y1        = rectROI[1]
            y2        = rectROI[1] + rectROI[3]

            if x2 >= frame.shape[1]:
                x2 = frame.shape[1] - 1
            if y2 >= frame.shape[0]:
                y2 = frame.shape[0] - 1

            src = frame[y1:y2, x1:x2]
            topLeft, minVal = MatchTemplate(src, stageTmpl.img, 0.01)

            # cv2.imwrite('src.png', src)
            # cv2.imwrite('tmpl.png', stageTmpl.img)

            if minVal < bestMinVal:
                if topLeft is not None:
                    bestTopLeft = topLeft
                    bestStage   = key
                    bestMinVal  = minVal

        return bestStage, bestTopLeft

    def GetCurrentNum(self, frame):
        num = self.GetDigital(frame, self.__digitalLen, self.__digitalPos)
        return num

    def GetDigital(self, frame, numOfDigitals, posList):
        num = 0
        len = numOfDigitals

        for pos in posList:
            len -= 1
            (digital, _) = self._GetOneDigital(pos, frame)
            if digital == 10:
                continue

            num = digital + num * 10

        return num

    def GetDigitalByRegion(self, frame, region, digitalW, digitalH, numOfDigitals, expandSize=0):
        x = region[0]
        y = region[1]
        posList = []
        for i in range(numOfDigitals):
            posList.append((x+i*digitalW-expandSize, y-expandSize, digitalW+2*expandSize+1, digitalH+2*expandSize))

        num = self.GetDigital(frame, numOfDigitals, posList)
        return num

class CharacterReg:
    def __init__(self, logger, dataFolder):
        self.__logger = logger
        self._LoadChars(dataFolder)

    def _LoadChars(self, dataFolder):
        self.__charTmpl = {}

        if not os.path.exists(dataFolder):
            os.mkdir(dataFolder)

        for root, dirs, files in os.walk(dataFolder):
            list = files  # 列出文件夹下所有的目录与文件
            for i in range(0, len(list)):
                path = os.path.join(root, list[i])
                (shortname, extname) = os.path.splitext(list[i])
                if os.path.isfile(path) is not True:
                    continue
                if path.find('png') < 0 and path.find('bmp') < 0:
                    continue
                if path.find('CharImg') >= 0 or \
                   path.find('test') >= 0 or \
                   path.find('Non') >= 0:
                    continue

                fileName = path
                state = shortname

                self.__charTmpl[state] = TemplateImage(fileName, 1.0, 1.0)
                # self.__logger.info("load char file: {0}, state: {1}".format(fileName, state))

    def AddImageData(self, fileName, state, ROI = None):
        self.__charTmpl[state] = TemplateImage(fileName, 1.0, 1.0, ROI)
        self.__logger.info("load tmpl file: {0}, state: {1}".format(fileName, state))

    def _GetOneChar(self, charImg):
        bestTopLeft = None
        bestMinVal  = 1.0
        bestStage   = GAME_STAGE_UNKNOWN

        for key in self.__charTmpl:
            stageTmpl = self.__charTmpl[key]
            if charImg.shape[0] < stageTmpl.img.shape[0] or \
               charImg.shape[1] < stageTmpl.img.shape[1] or \
               charImg.shape[2] != stageTmpl.img.shape[2]:
               continue

            topLeft, minVal = MatchTemplate(charImg, stageTmpl.img, 0.01)

            # cv2.imwrite('src.png', src)
            # cv2.imwrite('tmpl.png', stageTmpl.img)

            if minVal < bestMinVal:
                if topLeft is not None:
                    bestTopLeft = topLeft
                    bestStage   = key
                    bestMinVal  = minVal

        return bestStage, bestTopLeft

    def GetCharByRegion(self, frame, region, textReg, minWidth = 9, minHeight = 10,
                        spaceColor = 230, spaceMode = 'lt', dir = './data/Char',
                        textFillColor = 255):
        name = ''
        if region is not None:
            x = region[0]
            y = region[1]
            w = region[2]
            h = region[3]
        else:
            x = 0
            y = 0
            w = frame.shape[1]
            h = frame.shape[0]

        src = frame[y:y+h, x:x+w]
        left = FindLeftNoVerticalSpaceLine(src, spaceColor, spaceMode)
        right = 0
        i = 0

        while (1):
            if left >= w or (w - left) < 9:
                break

            right = left

            charW = FindLeftVerticalSpaceLine(src[:,left:], spaceColor, spaceMode)
            while charW < minWidth:
                charW += 1 # skip vertical line, such as '村', '小'
                newLeft = charW + left
                charW += FindLeftVerticalSpaceLine(src[:,newLeft:], spaceColor, spaceMode)
            right += charW

            if right > w:
                break

            charImg = src[:,left:right]
            if charImg is None:
                break

            if charImg.shape[2] == 4:
                charImg = cv2.cvtColor(charImg, cv2.COLOR_RGBA2RGB)

            # cv2.imwrite("src.png", charImg)
            (char, _) = self._GetOneChar(charImg)
            if char == '川' and minWidth == 9:
                charW += 2
            elif char == '川' and minWidth == 14:
                charW += 3

            if charImg.shape[0] < minHeight or charImg.shape[1] < minWidth:
                continue

            if char == GAME_STAGE_UNKNOWN and textReg != None:
                cv2.imwrite('./data/WhiteChar/CharImg.png', charImg)
                text = textReg.GetTextOnly(charImg, textFillColor)

                # Remove unused whitespace
                charImg = CutSpaceFromImg(charImg, spaceColor, spaceMode)

                if not 'words_result' in text:
                    break

                if len(text['words_result']) > 0:
                    fileName = text['words_result'][0]['words']
                else:
                    fileName = 'Non'

                if not os.path.exists(dir):
                    os.mkdir(dir)

                imgName = '{0}/{1}.png'.format(dir, fileName)

                if not os.path.exists(imgName):
                    cv2.imencode('.png', charImg)[1].tofile(imgName)
                elif fileName != 'Non':
                    for i in range(10):
                        imgName = '{0}/{1}{2}.png'.format(dir, fileName, i)
                        newFileName = '{0}{1}'.format(fileName, i)
                        if not os.path.exists(imgName):
                            cv2.imencode('.png', charImg)[1].tofile(imgName)
                            fileName = newFileName
                            break
                elif fileName == 'Non':
                    cv2.imencode('.png', charImg)[1].tofile(imgName)

                self.AddImageData(imgName, fileName)

                (char, _) = self._GetOneChar(charImg)

            if minWidth >= 11:
                if char == '山':
                    charW = 12

                if char == '川0':
                    charW = 13

                if char == '川1':
                    charW = 14

                if char == '瀧1':
                    charW = 15

                if char == '甑1':
                    charW = 30

            left += charW
            left += FindLeftNoVerticalSpaceLine(src[:, left:], spaceColor, spaceMode)
            i += 1

            if char == GAME_STAGE_UNKNOWN or \
               char == 'Non' or \
               char.find('Wrong') >= 0:
                continue

            if len(char) > 1:
                name += char[0]
            else:
                name += char

        return name
