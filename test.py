from __future__ import print_function
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.core import Activation, Dense, Flatten, Dropout, MaxoutDense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage.util.shape import view_as_windows
from sklearn.metrics import classification_report
import click
import glob
import numpy as np
import model_utils
import cv2 as cv
import utils
import h5py
import os


mean_train = 87.062
img_size = (240, 320)


def postprocess(pred):
    thresh = np.mean(pred)
    ret = cv.threshold(pred, thresh, 255, cv.THRESH_BINARY)[1]
    ret = cv.dilate(ret, np.ones((3, 3)), iterations=5)
    ret = cv.erode(ret, np.ones((5, 5)), iterations=7)
    return (ret / 255).astype('uint8')


def predict(img, model):
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

    return cv.resize(mask, (img_size[1], img_size[0]))


def load_image_for_dataset(filename, mask_name, resize=(240, 320)):
    img = cv.imread(filename)
    mask = cv.imread(mask_name)
    mask = (mask[:, :, 2] == 255).astype('uint8')
    if resize is not None:
        img = cv.resize(img, resize)
        mask = cv.resize(mask, resize)
    return {
        "img": img,
        "mask": mask
    }


def load_data(folder, img_ext='.png', mask_ext='_mask.jpg', resize=(240, 240)):
    X = []
    y = []
    glob_selector = '{}/*{}'.format(folder, img_ext)
    for file in glob.glob(glob_selector):
        filename = os.path.splitext(file)[0]
        mask_filename = '{}{}'.format(filename, mask_ext)
        if not os.path.exists(mask_filename):
            continue
        arr = load_image_for_dataset(file, mask_filename)
        arr['img'] = arr['img']
        yield arr


@click.command()
@click.argument('model')
def main(model):
    model = load_model(model)
    mask_arr = []
    pred_arr = []
    for i, arr in enumerate(load_data('./dataset/train', resize=None)):
        img = arr['img']
        mask = arr['mask']
        pred_mask = predict(img, model) * 255
        pred_mask = postprocess(pred_mask)
        mask_arr.append(mask.flatten())
        pred_arr.append(pred_mask.flatten())
        # if i % 20 == 0:
        #     cv.imwrite('test_post/img{}.png'.format(i), img)
        #     cv.imwrite('test_post/mask{}.png'.format(i), mask)
        #     cv.imwrite('test_post/pred_mask{}.png'.format(i), pred_mask)
    mask_arr = np.asarray(mask_arr).reshape(-1)
    pred_arr = np.asarray(pred_arr).reshape(-1)
    print(classification_report(mask_arr, pred_arr))
    acc = np.mean(mask_arr == pred_arr)
    print("acc: {}".format(acc))

if __name__ == '__main__':
    main()
