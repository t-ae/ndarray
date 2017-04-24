#!/usr/bin/env python

import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.utils.np_utils import to_categorical

iris = datasets.load_iris()

start = time.time()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

normalize_mu = np.mean(x_train, 0)
normalize_sigma = np.std(x_train, 0)

x_train = (x_train - normalize_mu) / normalize_sigma
x_test = (x_test - normalize_mu) / normalize_sigma
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    InputLayer([4]),
    Dense(5, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile("sgd", "categorical_crossentropy", ["accuracy"])

for i in range(3001):
    model.train_on_batch(x_train, y_train)

print(model.evaluate(x_test, y_test, batch_size=len(x_test)))
print(f"elapsed time: {time.time()-start}sec")
