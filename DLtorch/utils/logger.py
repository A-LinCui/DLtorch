# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import os
import sys
import logging

__all__ = ["logger", "getLogger"]

LEVEL = "info"
LEVEL = getattr(logging, LEVEL.upper())
LOG_FORMAT = "%(asctime)s %(name)-16s %(levelname)7s: %(message)s"

logging.basicConfig(stream=sys.stdout, level=LEVEL, format=LOG_FORMAT, datefmt="%m/%d %I:%M:%S %p")
logger = logging.getLogger()

def addFile(self, filename):
    handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    self.addHandler(handler)

logging.Logger.addFile = addFile

def getLogger(name):
    return logger.getChild(name)
