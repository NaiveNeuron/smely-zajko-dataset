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

model = models.mlp(n_input=75, architecture=[(20, 'sigmoid'), (2, 'softmax')],
                   metrics=['accuracy'])
model.load_weights('mlp_20_sigmoid_2_softmax.hdf5')

for i in range(len(X_t)):
    X, y, Z = X_t[i], y_t[i], Z_t[i]
    y_pred = model.predict(X)
    prediction = (y_pred[:, 1].reshape(48, 64)*255).astype('uint8')
    cv.imshow('prediction', cv.resize(prediction, (320, 240)))
    cv.imshow('img', Z)
    cv.waitKey(0)
