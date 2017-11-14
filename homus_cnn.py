'''
Trains a simple LeNet-5 (http://yann.lecun.com/exdb/lenet/) adapted to the HOMUS dataset using Keras Software (http://keras.io/)

LeNet-5 demo example http://eblearn.sourceforge.net/beginner_tutorial2_train.html

This example executed with 8x8 reescaled images and 50 epochs obtains an accuracy close to 70%.
'''

import numpy as np
import glob

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras import backend as K

batch_size = 64
nb_classes = 32
epochs = 50

# HOMUS contains images of 40 x 40 pixels
# input image dimensions for train
img_rows, img_cols = 40, 40

# Set image channels order
K.set_image_data_format('channels_last')

def load_data():
    #
    # Load data from data/HOMUS/train_0, data/HOMUS/train_1,...,data/HOMUS_31 folders from HOMUS images
    #
    image_list = []
    class_list = []
    for current_class_number in range(0,nb_classes):    # Number of class
        for filename in glob.glob('./data/HOMUS/train_{}/*.jpg'.format(current_class_number)):
            im = load_img(filename, grayscale=True, target_size=[img_rows, img_cols])  # this is a PIL image
            image_list.append(np.asarray(im).astype('float32')/255)
            class_list.append(current_class_number)

    n = len(image_list)    # Total examples

    X = np.asarray(image_list).reshape(n, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    Y = np_utils.to_categorical(np.asarray(class_list), nb_classes)

    msk = np.random.rand(len(X)) < 0.9 # Train 90% and Test 10%
    X_train, X_test = X[msk], X[~msk]
    Y_train, Y_test = Y[msk], Y[~msk]

    return X_train, Y_train, X_test, Y_test, input_shape


def cnn_model(input_shape):
    #
    # LeNet-5: Artificial Neural Network Structure
    #

    model = Sequential()

    model.add(Conv2D(12, (2, 2), padding='same', input_shape = input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(20, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(18, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(Conv2D(8, (1,1), padding='same'))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


##################################################################################
# Main program

# the data split between train and test sets
X_train, Y_train, X_test, Y_test, input_shape = load_data()

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(img_rows, 'x', img_cols, 'image size')
print(input_shape, 'input_shape')
print(epochs, 'epochs')

model = cnn_model(input_shape)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=3)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, Y_test), callbacks=[early_stopping])

#
# Results
#
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:{:.2f} accuracy: {:.2f}%'.format(loss, acc*100))


# file name to save model
filename='homus_cnn.h5'

# save network model
#model.save(filename)

# load neetwork model
#model = load_model(filename)