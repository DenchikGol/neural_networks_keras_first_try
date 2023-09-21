from __future__ import print_function
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K
import os

K.set_image_data_format('channels_last')

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_mnist_trained_model.h5'


seed = 7
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255
X_test = X_test / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


# base CNN model
# def base_model():
#     model = Sequential()
#     model.add(Conv2D(32, (5,5), input_shape=(28, 28, 1), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model


# model = base_model()
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2, shuffle=True)
# scores = model.evaluate(X_test, y_test, verbose=0)

# print(f'Точность CNN {round(scores[1] * 100, 2)}%')


def large_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = large_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=200, epochs=10, shuffle=True)

scores = model.evaluate(X_test, y_test, verbose=0)
print(f'CNN с несколькими слоями нейронов. Точность: {round(scores[1] * 100, 3)}')


# Сохраняем натренированную модель
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)

# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print(f'Сохранил натренированную модель в {model_path}')
