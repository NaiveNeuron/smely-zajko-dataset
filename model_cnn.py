from __future__ import print_function
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import model_utils
import numpy as np
import utils

np.random.seed(1337)  # for reproducibility

batch_size = 32
nb_epoch = 2
window = (20, 20)
stride = 5


# eigenvalues & eigenvectors for plzen dataset
einval = np.array([0.91650828, 0.0733283, 0.04866549])
vec = np.array([[-0.5448285, 0.82209843, 0.16527574],
                [-0.60238846, -0.24659869, -0.75915561],
                [-0.58334386, -0.51316981, 0.6295766]])

dataset_train = utils.augmented_dataset_from_folder('./plzen/train',
                                                    einval, vec, resize=None)
train_data = model_utils.prepare_pixelized_dataset(dataset_train, window,
                                                   stride=stride,
                                                   image_by_image=True)
X_train, y_train, _ = train_data

num_im, rows, cols = X_train.shape[:3]
X_train = np.resize(X_train, (num_im * rows * cols, window[0], window[1], 3))
X_train = np.swapaxes(np.rollaxis(X_train, 3, 1), 2, 3)
y_train = utils.bit_to_two_cls(np.resize(y_train, (num_im * rows * cols,)))

dataset_test = utils.augmented_dataset_from_folder('./plzen/test',
                                                   einval, vec, resize=None)

test_data = model_utils.prepare_pixelized_dataset(dataset_test, window,
                                                  stride=stride,
                                                  image_by_image=True)
X_test, y_test, _ = test_data
num_im, rows, cols = X_test.shape[:3]
X_test = np.resize(X_test, (num_im * rows * cols, window[0], window[1], 3))
X_test = np.swapaxes(np.rollaxis(X_test, 3, 1), 2, 3)
y_test = utils.bit_to_two_cls(np.resize(y_test, (num_im * rows * cols,)))

print("Dataset loaded X shape: {}, y shape: {}".format(X_train.shape,
                                                       y_train.shape))
utils.show_dataset_samples(X_train, y_train)

model = Sequential()

model.add(Convolution2D(10, 5, 5, border_mode='same',
                        input_shape=(3, window[0], window[1])))
model.add(Activation('relu'))
model.add(Convolution2D(10, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))

adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam,
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_split=0.1)

model_utils.plot_history(history, [['loss', 'val_loss'], ['acc', 'val_acc']],
                                  [['o', 'o'], ['-o', '-go']])

loss, acc = model.evaluate(X_test, y_test, verbose=1)
print("\nloss: {:.4}, acc: {:.4}".format(loss, acc))
