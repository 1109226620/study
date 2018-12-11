# -*- coding:utf-8 -*-
__author__ = "songyibin"
__time__ = "2018/12/11"

# -*- coding:utf-8 -*-
__author__ = "songyibin"
__time__ = "2018/12/9"

import tensorflow as tf
import os
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

hello = tf.constant("hello")
sess = tf.Session()
print(sess.run(hello))