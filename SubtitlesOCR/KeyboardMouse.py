# ----------------------------------------------
# Project: SengokushiAI
# Author: huang ying
# Date: 2018.10.31
# ----------------------------------------------

import threading
import time
import json
import queue
import os
from pynput.mouse import Button, Controller
from pynput import keyboard


EVENT_INVALID            = -1
EVENT_MOUSE              = 10000
EVENT_MOUSE_RIGHT_CLICK  = EVENT_MOUSE + 1
EVENT_MOUSE_LEFT_CLICK   = EVENT_MOUSE + 2
EVENT_MOUSE_RIGHT_DCLICK = EVENT_MOUSE + 3
EVENT_MOUSE_LEFT_DCLICK  = EVENT_MOUSE + 4
EVENT_KEYBOARD           = 20000
EVENT_KEYBOARD_CHAR      = EVENT_KEYBOARD + 1
EVENT_KEYBOARD_ALT_CHAR  = EVENT_KEYBOARD + 2
EVENT_KEYBOARD_CTRL_CHAR = EVENT_KEYBOARD + 3

class InputEvent:
    def __init__(self, eventType, bBack = False):
        self.eventType = eventType
        self.bBack     = bBack      # Check if go back to original position.

    # The default sleep time is 0.1s.
    def MouseClick(self, type, x, y, sleep = 0.1):
        self.mouseInput = type
        self.mouseX     = x
        self.mouseY     = y
        self.sleep      = sleep

    # The default sleep time is 0.1s.
    def KeyboardInput(self, type, char, sleep = 0.1):
        self.keyInput = type
        self.keyChar  = char
        self.sleep    = sleep


class KeyboardMouse(threading.Thread):
    def __init__(self, logger, commonCfg):
        super(KeyboardMouse, self).__init__()

        self.__logger = logger
        self._LoadCommonParams(commonCfg)
        self.__lastFrameTime = time.time()

        # Input event queue
        self.__inputEventQueue      = queue.Queue()
        #self.__inputEventQueueMutex = threading.Lock()

        self._CreateEventDict()

        # Create listener
        self.__keyboardListenerTH = threading.Thread(target=self._KeyBoardListener, args=())

        # short key state
        self._InitShortKeyState()

        # Create Mouse controller
        self.__mouse = Controller()

    def _InitShortKeyState(self):
        self.__keyBoardMutex      = threading.Lock()
        self.__bRunningController = False
        self.__bLCtrlPress        = False

    def _SetLCtrlState(self, bPress):
        self.__keyBoardMutex.acquire()
        self.__bLCtrlPress = bPress
        self.__keyBoardMutex.release()

    # L-Ctrl + 'r' : run controller
    # L-Ctrl + 'p' : pause controller
    def _SetRunningFlag(self, key):
        if self.__bLCtrlPress is False:
            return

        self.__keyBoardMutex.acquire()
        if key.char == 'R' or key.char == 'r':
            self.__bRunningController = True
        elif key.char == 'P' or key.char == 'p':
            self.__bRunningController = False
        self.__keyBoardMutex.release()

    def IsRunning(self):
        self.__keyBoardMutex.acquire()
        bRunning = self.__bRunningController
        self.__keyBoardMutex.release()
        return bRunning

    def _LoadCommonParams(self, commonCfg):
        self.__commonCfg = commonCfg

    def _Sleep(self):
        timeNow = time.time()
        timePassed = timeNow - self.__lastFrameTime
        if timePassed < self.__commonCfg.timePerFrame:
            timeDelay = self.__commonCfg.timePerFrame - timePassed
            time.sleep(timeDelay)

        self.__lastFrameTime = timeNow

    def PushEvent(self, inputEvent):
        #self.__inputEventQueueMutex.acquire()
        self.__inputEventQueue.put(inputEvent)
        #self.__inputEventQueueMutex.release()

    def PopEvent(self):
        #self.__inputEventQueueMutex.acquire()
        inputEvent = self.__inputEventQueue.get()
        #self.__inputEventQueueMutex.release()
        return inputEvent

    def _CreateEventDict(self):
        self.__eventDic = {}
        self.__eventDic[EVENT_MOUSE]              = 'mouse'
        self.__eventDic[EVENT_MOUSE_RIGHT_CLICK]  = 'mouse_right_click'
        self.__eventDic[EVENT_MOUSE_LEFT_CLICK]   = 'mouse_left_click'
        self.__eventDic[EVENT_MOUSE_RIGHT_DCLICK] = 'mouse_right_dclick'
        self.__eventDic[EVENT_MOUSE_LEFT_DCLICK]  = 'mouse_left_dclick'
        self.__eventDic[EVENT_KEYBOARD]           = 'key'
        self.__eventDic[EVENT_KEYBOARD_CHAR]      = 'key_char'
        self.__eventDic[EVENT_KEYBOARD_ALT_CHAR]  = 'key_alt_char'
        self.__eventDic[EVENT_KEYBOARD_CTRL_CHAR] = 'key_ctrl_char'

    def _GetEventName(self, eventType):
        return self.__eventDic[eventType]

    def OnPress(self, key):
        try:
            self.__logger.info('alphanumeric key  {0} pressed'.format(key.char))
            self._SetRunningFlag(key)
        except AttributeError:
            self.__logger.info('special key {0} pressed'.format(key))
            if key == keyboard.Key.ctrl_l:
                self._SetLCtrlState(True)

    def OnRelease(self, key):
        try:
            self.__logger.info('{0} released'.format(key))
        except AttributeError:
            if key == keyboard.Key.ctrl_l:
                self._SetLCtrlState(False)

            if key == keyboard.Key.esc:
                return False

    def _KeyBoardListener(self):
        while True:
            with keyboard.Listener(
                on_press = self.OnPress,
                on_release = self.OnRelease) as listener:
                listener.join()

    def _ExecuteInputEvent(self, inputEvent):
        if inputEvent.eventType == EVENT_MOUSE:
            self._ExecuteMouseEvent(inputEvent)
        elif inputEvent.eventType == EVENT_KEYBOARD:
            self._ExecuteKeyboardEvent(inputEvent)

    # Before executing mouse event we should record current mouse position.
    # After executing mouse event, we should move mouse pointer to original position.
    def _ExecuteMouseEvent(self, inputEvent):
        lastPos = self.__mouse.position

        self.__mouse.position = (inputEvent.mouseX, inputEvent.mouseY)
        if inputEvent.mouseInput == EVENT_MOUSE_LEFT_CLICK:
            self.__mouse.press(Button.left)
            self.__mouse.release(Button.left)

        # mouse.press(Button.left)
        # mouse.release(Button.left)

        # #Double click
        # mouse.click(Button.left, 1)

        # #scroll two  steps down
        # mouse.scroll(0, 500)

        # Sleep after executing event.
        time.sleep(inputEvent.sleep)

        # After executing mouse event, we move mouse pointer to original position.

        if inputEvent.bBack is True:
            self.__mouse.position = lastPos
            self.__mouse.press(Button.left)
            self.__mouse.release(Button.left)
        return

    def _ExecuteKeyboardEvent(self, inputEvent):
        return

    def run(self):
        self.__keyboardListenerTH.start()

        while True:
            inputEvent = self.PopEvent()
            # self.__logger.info("Event: {0}".format(self._GetEventName(inputEvent.eventType)))
            self._ExecuteInputEvent(inputEvent)
            self._Sleep()