from __future__ import print_function

from utils import (dataset_from_folder, bit_to_two_cls,
                   augmented_dataset_from_folder)
from model_utils import prepare_pixelized_dataset, train_and_eval
import models

import numpy as np
np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_epoch = 20

verbose = 0

loss = 'mse'

# eigenvalues & eigenvectors for plzen dataset
einval = np.array([0.91650828, 0.0733283, 0.04866549])
vec = np.array([[-0.5448285, 0.82209843, 0.16527574],
                [-0.60238846, -0.24659869, -0.75915561],
                [-0.58334386, -0.51316981, 0.6295766]])

dataset = augmented_dataset_from_folder('./plzen/train', einval,
                                        vec, resize=None)
X_train, y_train = prepare_pixelized_dataset(dataset,
                                             y_applied_function=bit_to_two_cls)

dataset = augmented_dataset_from_folder('./plzen/test', einval,
                                        vec, resize=None)
X_test, y_test = prepare_pixelized_dataset(dataset,
                                           y_applied_function=bit_to_two_cls)

print("Done loading dataset X: {}, y: {}".format(X_train.shape, y_train.shape))
print("Done loading dataset X_test: {}, y_test: {}".format(X_test.shape,
                                                           y_test.shape))

print('\n')
model = models.mlp(n_input=75, architecture=[(8, 'sigmoid'), (2, 'softmax')],
                   metrics=['accuracy'], loss=loss)
train_and_eval(model, X_train, y_train, X_test, y_test,
               score_name='accuracy', score_format='{[1]}', verbose=verbose,
               batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(12, 'sigmoid'), (2, 'softmax')],
                   metrics=['accuracy'], loss=loss)
train_and_eval(model, X_train, y_train, X_test, y_test,
               score_name='accuracy', score_format='{[1]}', verbose=verbose,
               batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(20, 'sigmoid'), (2, 'softmax')],
                   metrics=['accuracy'], loss=loss)
train_and_eval(model, X_train, y_train, X_test, y_test,
               score_name='accuracy', score_format='{[1]}', verbose=verbose,
               batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(6, 'sigmoid'),
                                             (6, 'sigmoid'),
                                             (2, 'softmax')],
                   metrics=['accuracy'], loss=loss)
train_and_eval(model, X_train, y_train, X_test, y_test,
               score_name='accuracy', score_format='{[1]}', verbose=verbose,
               batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(10, 'sigmoid'),
                                             (10, 'sigmoid'),
                                             (2, 'softmax')],
                   metrics=['accuracy'], loss=loss)
train_and_eval(model, X_train, y_train, X_test, y_test,
               score_name='accuracy', score_format='{[1]}', verbose=verbose,
               batch_size=batch_size, nb_epoch=nb_epoch)
