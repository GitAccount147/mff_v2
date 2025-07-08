#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

a = tf.constant([1, 2, 3], dtype=tf.float32)
b = tf.tile(np.arange(0, 10), [4])
b = tf.reshape(b, [4, 10])
#a = tf.reshape(a, shape=[3, 2])
sin = tf.sin(a+a)
print(a)
print(sin)
print(b)
