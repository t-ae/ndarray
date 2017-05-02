#!/usr/bin/env python

import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


def softmax(x):
    e = np.exp(x)
    eSum = np.sum(e, 1).reshape([-1, 1])

    return e / eSum


def relu(x):
    x = x.copy()
    x[x < 0] = 0
    return x


def d_relu(x):
    x = x.copy()
    x[x < 0] = 0
    x[x > 0] = 1
    return x


def main():
    iris = datasets.load_iris()

    start = time.time()

    x_train, x_test, label_train, label_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

    normalize_mu = np.mean(x_train, 0)
    normalize_sigma = np.std(x_train, 0)

    x_train = (x_train - normalize_mu) / normalize_sigma
    x_test = (x_test - normalize_mu) / normalize_sigma
    y_train = to_categorical(label_train)
    y_test = to_categorical(label_test)

    numFeatures = x_train.shape[1]
    numTrainSamples = x_train.shape[0]
    numOutput = y_train.shape[1]
    labelsCount = np.sum(y_train, 0)

    # Two layer neural network
    # Input(4) -> Dense(5) -> ReLU -> Dense(3) -> Softmax
    numHiddenUnits1 = 5

    # init with glorot uniform
    W1_limit = np.sqrt(6 / (numFeatures + numHiddenUnits1))
    W1 = np.random.uniform(-W1_limit, W1_limit, [numFeatures, numHiddenUnits1])  # [4, 5]
    b1 = np.zeros([numHiddenUnits1])  # [5]
    W2_limit = np.sqrt(2 / (numHiddenUnits1 + numOutput))
    W2 = np.random.uniform(-W2_limit, W2_limit, [numHiddenUnits1, numOutput])  # [5, 3]
    b2 = np.zeros([numOutput])  # [5]
    alpha = 1e-3

    for i in range(30001):
        h1_1 = np.matmul(x_train, W1)  # [90, 5]
        h1_2 = h1_1 + b1  # [90, 5]
        h1 = relu(h1_2)  # [90, 5]

        h2_1 = np.matmul(h1, W2)  # [90, 3]
        h2 = h2_1 + b2  # [90, 3]

        out = softmax(h2)  # [90, 3]

        # back propagation
        d_out_h2 = out - y_train  # [90, 3]

        d_h2_b2 = np.ones(b2.shape)  # [90, 3]
        d_h2_h2_1 = np.ones(h2_1.shape)  # [90, 3]

        d_h2_1_W2 = h1  # [90, 5]
        d_h2_1_h1 = W2  # [90, 5, 3]

        d_h1_h1_2 = d_relu(h1_2)  # [90, 5]

        d_h1_2_b1 = np.ones(b1.shape)  # [90, 5]
        d_h1_2_h1_1 = np.ones(h1_1.shape)  # [90, 5]

        d_h1_1_W1 = x_train  # [90, 4]

        # chain
        d_out_b2 = d_h2_b2 * d_out_h2  # [90, 3]
        d_out_h2_1 = d_h2_h2_1 * d_out_h2  # [90, 3]
        d_out_W2 = np.matmul(np.expand_dims(d_h2_1_W2, -1),
                             np.expand_dims(d_out_h2_1, 1))  # [90, 5, 3]
        d_out_h1 = np.matmul(d_h2_1_h1,
                             np.expand_dims(d_out_h2_1, -1)) \
            .squeeze()  # [90, 5]
        d_out_h1_2 = d_h1_h1_2 * d_out_h1  # [90, 5]
        d_out_b1 = d_out_h1_2 * d_h1_2_b1  # [90, 5]
        d_out_h1_1 = d_h1_2_h1_1 * d_out_h1_2  # [90, 5]
        d_out_W1 = np.matmul(np.expand_dims(d_h1_1_W1, -1),
                             np.expand_dims(d_out_h1_1, 1))  # [90, 4, 5]

        # update
        b2 -= alpha * np.mean(d_out_b2, 0)
        W2 -= alpha * np.mean(d_out_W2, 0)

        b1 -= alpha * np.mean(d_out_b1, 0)
        W1 -= alpha * np.mean(d_out_W1, 0)

        if i % 100 == 0:
            print(f"\nstep: {i}")
            out[out < 1e-10] = 1e-10
            losses = -y_train * np.log(out)
            loss = np.mean(np.sum(losses, 1))
            featureLosses = np.sum(losses, 0) / labelsCount
            print(f"loss: {loss}, ({featureLosses})")

            answer = np.argmax(out, 1)
            trues = np.sum(answer == label_train)
            accuracy = trues / numTrainSamples

            print(f"accuracy: {accuracy}")

    # test
    labelsCount = np.sum(y_test, 0)
    h1_1 = np.matmul(x_test, W1)  # [90, 5]
    h1_2 = h1_1 + b1  # [90, 5]
    h1 = relu(h1_2)  # [90, 5]

    h2_1 = np.matmul(h1, W2)  # [90, 3]
    h2 = h2_1 + b2  # [90, 3]

    out = softmax(h2)  # [90, 3]

    print(f"\ntest result:")
    out[out < 1e-10] = 1e-10
    losses = -y_test * np.log(out)
    loss = np.mean(np.sum(losses, 1))
    featureLosses = np.sum(losses, 0) / labelsCount
    print(f"loss: {loss}, ({featureLosses})")

    answer = np.argmax(out, 1)
    trues = np.sum(answer == label_test)
    accuracy = trues / numTrainSamples

    print(f"accuracy: {accuracy}")

    print(f"elapsed time: {time.time()-start}sec")


if __name__ == '__main__':
    main()
