# ----------------------------------------------
# Project: SengokushiAI
# Author: huang ying
# Date: 2018.10.31
# ----------------------------------------------

import threading
import schedule
import time
import os
import json
from PIL import ImageGrab
import cv2
import numpy as np
import queue


class FrameCapture(threading.Thread):
    def __init__(self, logger, commonCfg):
        super(FrameCapture, self).__init__()

        self.__logger = logger
        self._ReadCommonCfg(commonCfg)

        self.__lastFrameTime = time.time()
        self.__frame = None
        self.__frameMutex = threading.Lock()

    def _ReadCommonCfg(self, commonCfg):
        self.__commonCfg = commonCfg
        self.__logger.info('Screen size: {0} x {1}, Screen wait: {2}s'.format(
            self.__commonCfg.screenWidth, self.__commonCfg.screenHeight, self.__commonCfg.timePerFrame))

    def _Sleep(self):
        timeNow = time.time()
        timePassed = timeNow - self.__lastFrameTime
        if timePassed < self.__commonCfg.timePerFrame:
            timeDelay = self.__commonCfg.timePerFrame - timePassed
            time.sleep(timeDelay)

        self.__lastFrameTime = timeNow

    def _GetNextFrame(self):
        #self.__logger.info('Get next frame')

        image = ImageGrab.grab((self.__commonCfg.screenX, self.__commonCfg.screenY, self.__commonCfg.screenWidth, self.__commonCfg.screenHeight))

        cv2Img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        cv2ShowImg = cv2.resize(cv2Img, (self.__commonCfg.showWidth, self.__commonCfg.showHeight))
        cv2.imshow(self.__commonCfg.gameName, cv2ShowImg)
        cv2.waitKey(1) # minisecond

        cv2ProcImg = cv2.resize(cv2Img, (self.__commonCfg.processWidth, self.__commonCfg.processHeight))
        self._SetFrame(cv2ProcImg)
        return

    def _SetFrame(self, frame):
        self.__frameMutex.acquire()
        self.__frame = frame
        self.__frameMutex.release()

    def GetFrame(self):
        self.__frameMutex.acquire()
        frame = self.__frame
        self.__frameMutex.release()
        return frame

    def run(self):
        schedule.every(self.__commonCfg.timePerFrame).seconds.do(self._GetNextFrame)
 
        while True:
            schedule.run_pending()
            self._Sleep()