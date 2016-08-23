from __future__ import print_function
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import model_utils
import numpy as np
import utils

np.random.seed(1337)  # for reproducibility

batch_size = 32
nb_epoch = 50

# tested delims
# delims = [60, 55, 45, 45, 55, 60]
# delims = [90, 70, 70, 90]
delims = [120, 80, 120]
# delims = None

classes = 10

if delims is not None:
    classes = len(delims)

# eigenvalues & eigenvectors for plzen dataset
einval = np.array([0.91650828, 0.0733283, 0.04866549])
vec = np.array([[-0.5448285, 0.82209843, 0.16527574],
                [-0.60238846, -0.24659869, -0.75915561],
                [-0.58334386, -0.51316981, 0.6295766]])

X_train, y_train = utils.load_augmented_dataset('./plzen/train', einval, vec,
                                                classes=classes, delims=delims)
X_test, y_test = utils.load_augmented_dataset('./plzen/test', einval, vec,
                                              classes=classes, delims=delims)
print("Dataset loaded X shape: {}, y shape: {}".format(X_train.shape,
                                                       y_train.shape))
_, y = np.nonzero(y_train)
print(np.histogram(y, bins=range(classes + 1)))

_, y = np.nonzero(y_test)
print(np.histogram(y, bins=range(classes + 1)))

utils.show_dataset_samples(X_train, y_train)

model = Sequential()

model.add(Convolution2D(8, 5, 5, border_mode='valid',
                        input_shape=(3, 240, 240),
                        subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(8, 3, 3, border_mode='valid', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(16, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(384))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(classes + 1))
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
