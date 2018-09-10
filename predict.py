from __future__ import print_function
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.core import Activation, Dense, Flatten, Dropout, MaxoutDense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage.util.shape import view_as_windows
from skimage.util.shape import view_as_windows
import click
import numpy as np
import model_utils
import cv2 as cv
import utils
import h5py
import os


mean_train = 87.062
img_size = (240, 320)

@click.command()
@click.argument('model')
@click.argument('img')
def main(model, img):
    model = load_model(model)
    img = cv.imread(img)
    img_res = cv.resize(img, img_size)
    is_cnn = len(model.input_shape) > 2
    is_categorical = model.get_weights()[-1].shape[0] > 1

    if is_cnn is True:
        w_window = model.input_shape[-1]
    else:
        w_window = int(np.sqrt(model.input_shape[-1] / 3))

    img_slices = np.squeeze(view_as_windows(img_res, (w_window, w_window, 3),
                                            step=w_window))
    img_slices = (img_slices - mean_train) / 255.0
    nx, ny, w, h, d = img_slices.shape
    if is_cnn is True:
        img_slices = np.reshape(img_slices, (nx * ny, w, h, d))
        img_slices = np.rollaxis(img_slices, 3, 1)
    else:
        img_slices = np.reshape(img_slices, (nx * ny, w * h * d))

    y_pred = model.predict(img_slices)
    reshape_size = (img_size[1] / w_window, img_size[0] / w_window)

    if is_categorical is True:
        mask = y_pred[:, 1].reshape(reshape_size)
    else:
        mask = y_pred.reshape(reshape_size)

    mask = cv.resize(mask * 255, (img_size[1], img_size[0]))
    cv.imwrite('predicted_mask.png', mask)

if __name__ == '__main__':
    main()
