from __future__ import print_function

import utils

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop

from matplotlib import pyplot as plt

import numpy as np
np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_epoch = 30


def prepare_dataset(dataset, window_x=5, window_y=5):
    Xs = []
    ys = []
    for datum in dataset:
        img = datum['img']
        mask = datum['mask']
        for selection, mask_selection in prepare_image(img, mask):
            Xs.append(selection)
            ys.append(mask_selection)
    return np.array(Xs), np.array(ys)


def prepare_image(img, mask=None, window_x=5, window_y=5):
    h, w, ch = img.shape
    for j in range(w//window_y - 1):
        for i in range(h//window_x - 1):
            selection = img[i*window_x:(i+1)*window_x,
                            j*window_y:(j+1)*window_y] / 255.0
            selection = selection.reshape(-1)
            mask_selection = None
            if mask is not None:
                mask_selection = mask[i*window_x:(i+1)*window_x,
                                      j*window_y:(j+1)*window_y] / 255.0
                mask_selection = int(mask_selection.mean() > 0.5)
            yield selection, mask_selection

dataset = utils.dataset_from_folder('./plzen/train')
X_train, y_train = prepare_dataset(dataset)

dataset = utils.dataset_from_folder('./plzen/test')
X_test, y_test = prepare_dataset(dataset)

model = Sequential()
model.add(Dense(8, input_shape=(75,)))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

rms = RMSprop(lr=0.01)
model.compile(loss='mse', optimizer=rms)

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch, verbose=1,
                    validation_split=0.1)

plt.plot(history.history['loss'], 'o', label='loss')
plt.plot(history.history['val_loss'], '-go', label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test MSE: {}'.format(score))
