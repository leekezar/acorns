import torch
import torch.utils.data as data
from PIL import Image
from spatial_transforms import *
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random
import glob
from utils import load_value_file

class ASLLVD(data.Dataset):
	def __init__(self):

		return

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (clip, target) where target is class_index of the target class
		"""

		return

	def __len__(self):
		return len(self.data)