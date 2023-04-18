#Copied from https://github.com/AICPS/roadscene2vec
#Copyright (c) 2021 UC Irvine Advanced Integrated Cyber-Physical Systems Lab (AICPS)

import sys
from pathlib import Path
from abc import ABC

'''Abstract base class used to create CarlaPreprocessor and RealPreprocessor'''
class Preprocessor(ABC):
    def __init__(self, config):
        self.conf = config
        self.dataset = None

