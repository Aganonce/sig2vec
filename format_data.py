# from keras.datasets import mnist
import numpy as np
import pickle

# Example, load in MNIST data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)

# Data found in /home/buchaf/Documents/GOD/data
data = []
with open('raw_data/enwiki_20180420_100d.txt', 'r') as ins:
    for line in ins:
        line = line.rstrip('\n')
        line = line.split(' ')
        if len(line) == 100:
            line = [float(x) for x in line]
            data.append(np.array(line))

data = np.array(data)

# Randomly select a subset
inds = np.random.choice(len(data), size=600000, replace=False)

data = data[inds,:]

pickle.dump(data, open('data/data.pkl', 'wb'))