#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from homr_dataset import HOMRDataset

#t = tf.RaggedTensor.from_row_lengths( values=[3, 1, 4, 1, 5, 9, 2, 6], row_lengths=[4, 0, 3, 1, 0])
#t = tf.RaggedTensor([[4, 3], [2], [1, 4, 6]])
#t = tf.RaggedTensor.from_row_lengths( values=[[3, 2], [1], [1, 6], [2, 3]], row_lengths=[1, 2, 1])
#t = tf.RaggedTensor.from_nested_row_lengths(flat_values=[1, 2, 3, 4, 5], nested_row_lengths=[[1, 1], [3]])
aa = tf.constant([[1, 2, 5], [3, 4, 5], [6, 4, 7]])
bb = tf.constant([[2, 3, 6, 7, 7], [5, 2, 3, 2, 1]])

#aa = tf.constant([[[1], [2], [5]], [[3], [4], [5]], [[6], [4], [7]]])
#bb = tf.constant([[[2], [3], [6], [7], [7]], [[5], [2], [3], [2], [1]]])
t = tf.ragged.stack([aa, bb])

print(t.shape)
heights = t.row_lengths(axis=1)
widths = t.row_lengths(axis=2)
widths_short = widths.to_tensor()[:, 0]
#print(widths_short)
list_comp = [tf.convert_to_tensor([heights[i] for _ in range(widths_short[i])]) for i in range(len(widths_short))]
heights_long = tf.ragged.stack(list_comp)
#print(heights_long.values)

#print(heights, widths)
dense = t.to_tensor()
#print(dense)
trans = tf.transpose(dense, perm=[0, 2, 1])
#print(trans)
#lens = [[3, 5], [3, 3, 3, 2, 2, 2, 2, 2]]

rag = tf.RaggedTensor.from_tensor(trans, lengths=[widths_short, heights_long.values])

#rag = tf.RaggedTensor.from_tensor(dense, lengths=lens)
#trans2 = tf.transpose(rag, perm=[0, 2, 1])

#print(trans2)
print(rag)
print(t)
