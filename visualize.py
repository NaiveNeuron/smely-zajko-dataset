from __future__ import print_function

import utils
from model_utils import prepare_pixelized_dataset
import models
from keras.utils.np_utils import to_categorical

import cv2 as cv

import numpy as np
np.random.seed(1337)  # for reproducibility

dataset = utils.dataset_from_folder('./plzen/test')

X_t, y_t, Z_t = prepare_pixelized_dataset(dataset,
                                          y_applied_function=to_categorical,
                                          image_by_image=True)

model = models.mlp(n_input=75, architecture=[(12, 'sigmoid'), (1, 'sigmoid')])

# Load weights obtained by training the above model on the './plze/train' data
model.load_weights('./mlp_12_sigmoid_1_sigmoid.hdf5')

for i in range(len(X_t)):
    X, y, Z = X_t[i], y_t[i], Z_t[i]
    y_pred = model.predict(X)
    prediction_mask = (y_pred.reshape(48, 64)*255).astype('uint8')
    cv.imshow('prediction_mask', cv.resize(prediction_mask, (320, 240)))

    prediction = cv.resize((y_pred.reshape(48, 64)), (320, 240))
    show_img = (prediction[:, :, np.newaxis] * Z).astype('uint8')
    cv.imshow('prediction', show_img)

    cv.imshow('img', Z)
    cv.waitKey(0)
