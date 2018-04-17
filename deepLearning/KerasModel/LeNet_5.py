import keras.datasets.mnist as mnist
from keras import Sequential
from keras import optimizers
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.layers import Conv2D, Flatten, Dense, Dropout, Reshape, MaxPooling2D
from keras.optimizers import SGD
import keras
from keras.regularizers import l2
from keras.utils import to_categorical

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


def scheduler(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <= 120:
        return 0.01
    if epoch <= 160:
        return 0.002
    return 0.0004
img_rows = 28
img_cols = 28

if __name__ == '__main__':
    model = build_model()
    model.load_weights("lenet-5-model")
    print(model.summary())
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_test = to_categorical(y_test, 10)
    y_train = to_categorical(y_train, 10)
    x_test = x_test.astype('float32')
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test /= 255

    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]

    model.fit(x_train, y_train, validation_split=0.2, verbose=1, shuffle=True,callbacks=cbks,
              batch_size=128,
              epochs=5)
    model.save("lenet-5-model")
    print(model.evaluate(x_test, y_test, batch_size=128))
