import cv2 as cv

import numpy as np


def blur_batch(batch):
    blured_batch = []
    for image in batch:
        gray_im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur_amount = cv.Laplacian(gray_im, cv.CV_64F).var()/1000.
        blured_im = cv.GaussianBlur(image, (3, 3), blur_amount, blur_amount)
        blured_batch.append(blured_im)
    return np.asarray(blured_batch)


def flip_batch(batch, batch_mask):
    batch_fliped = batch[:, :, ::-1, :]
    batch_mask_fliped = batch_mask[:, :, ::-1]
    return batch_fliped, batch_mask_fliped


def add_color_noise(batch, eigenvalues, eigenvectors):
    if (batch > 1).any():
        norm_data = batch.astype('float32') / 255.0
    else:
        norm_data = batch
    alpha = np.random.randn(norm_data.shape[0], 3) * 0.1
    noise = eigenvectors.dot((eigenvalues * alpha).T)
    norm_data += noise[:, np.newaxis, np.newaxis, :].T
    return norm_data
