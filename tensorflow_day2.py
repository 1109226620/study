# -*- coding:utf-8 -*-
__author__ = "songyibin"
__time__ = "2018/12/11"

import tensorflow as tf
from tensorflow import keras
import numpy as np

#下载数据集

imdb = keras.datasets.imdb
(train_data,train_label),(test_data,test_label) = imdb.load_data(num_words=10000)

#探索数据

print(len(train_data),len(train_label))
print(train_data[0])
print(len(train_data[0]),len(train_data[1]),len(train_data[2]))

#将整数转换回字词

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
print(word_index)
# The first indices are reserved
word_index = {k:(v+3)for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<STRAT>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])

print(decode_review(train_data[0]))

#准备数据
#由于影评的长度必须相同，我们将使用 pad_sequences 函数将长度标准化
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding="post",
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index["<PAD>"],
                                                        padding="post",
                                                        maxlen=256)
print(len(train_data[0]),len(train_data[1]),len(train_data[2]))
print(train_data[0])
12
