import keras
from keras import Sequential
from keras import optimizers
from keras.layers import Conv2D, Flatten, Dense, Dropout, Reshape, MaxPooling2D
from keras.regularizers import l2

log_filepath = "./LeNet-5"


def build_model():
    model = Sequential()
    # Reshape In:[Batch_size,28,28] Out:[Batch_size,28,28,1]
    model.add(Reshape(name="Reshape", input_shape=(28, 28), target_shape=(28, 28, 1)))
    # Conv2D In:[Batch_size,28,28,1] Out:[Batch_size,28,28,32]
    model.add(Conv2D(name="Conv_1", filters=32, input_shape=(28, 28, 1), kernel_size=(5, 5), padding="same",
                     activation="relu",
                     kernel_initializer="glorot_uniform", kernel_regularizer=l2(0.0001)))
    # Max_pool2D In:[Batch_size,28,28,32] Out:[Batch_size,14,14,32]
    model.add(MaxPooling2D(name="MaxPool_1", pool_size=(2, 2), strides=(2, 2)))
    # Conv2D In:[Batch_size,14,14,32] Out:[Batch_size,14,14,64]
    model.add(Conv2D(name="Conv_2", filters=64, kernel_size=(5, 5), padding="same", activation="relu",
                     kernel_regularizer=l2(0.0001)))
    # Max_pool2D In:[Batch_size,14,14,64] Out:[Batch_size,7,7,64]
    model.add(MaxPooling2D(name="MaxPool_2", pool_size=(2, 2), strides=(2, 2)))
    # Flattern In:[Batch_size,7,7,64] Out:[Batch_size,7*7*64]
    model.add(Flatten(name="Flattern"))
    # Dense In:[Batch_size,7*7*64] Out:[Barch_size,1024]
    model.add(Dense(name="Dense_1", units=1024, activation="relu", kernel_regularizer=l2(0.0001)))
    # DropOut In:[Batch_size,108]
    model.add(Dropout(name="DropOut_1", rate=0.4))
    # Dense In:[Batch_size,1024] Out:[Batch_size,10]
    model.add(Dense(name="Soft_max_1", activation="softmax", units=10))
    optim = optimizers.Adadelta()
    model.compile(optimizer=optim, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

model = build_model()
model.load_weights("./lenet-5-model")
tests = pd.read_csv("/home/tbxsx/Code/learnMachineLearning/Kaggle/data/myxgboosttest/test.csv")
tests = np.asarray(tests,dtype=np.float32)
tests = tests.reshape(tests.shape[0], 28, 28)
tests = tests / 255
pre = np.argmax(model.predict(tests),axis=1)
np.savetxt('/home/tbxsx/Code/learnMachineLearning/Kaggle/data/myxgboosttest/xgb_submission.csv', np.c_[range(1, len(tests) + 1), pre], delimiter=',', header='ImageId,Label',
               comments='', fmt='%d')

