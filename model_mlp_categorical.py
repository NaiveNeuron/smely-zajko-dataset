from __future__ import print_function

import utils
from model_utils import (prepare_pixelized_dataset, train_and_eval,
                         train_and_eval_generator)
import models
from keras.utils.np_utils import to_categorical

import numpy as np
np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_epoch = 30


# eigenvalues & eigenvectors for plzen dataset
einval = np.array([0.91650828, 0.0733283, 0.04866549])
vec = np.array([[-0.5448285, 0.82209843, 0.16527574],
                [-0.60238846, -0.24659869, -0.75915561],
                [-0.58334386, -0.51316981, 0.6295766]])
print('\n')
model = models.mlp(n_input=230400, architecture=[(10, 'sigmoid'),
                                                 (10, 'sigmoid'),
                                                 (2, 'softmax')],
                   metrics=['accuracy'])
train_and_eval_generator(model,
                         utils.dataset_generator('./plzen/train', einval, vec),
                         utils.dataset_generator('./plzen/test', einval, vec),
                         samples_per_epoch_train=812,
                         samples_per_epoch_test=200,
                         score_format='{[1]}', nb_epoch=nb_epoch)

dataset = utils.dataset_from_folder('./plzen/train')
X_train, y_train = prepare_pixelized_dataset(dataset,
                                             y_applied_function=to_categorical)

dataset = utils.dataset_from_folder('./plzen/test')
X_test, y_test = prepare_pixelized_dataset(dataset,
                                           y_applied_function=to_categorical)

print('\n')
model = models.mlp(n_input=75, architecture=[(8, 'sigmoid'), (2, 'softmax')],
                   metrics=['accuracy'])
train_and_eval(model, X_train, y_train, X_test, y_test,
               score_name='accuracy', score_format='{[1]}',
               batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(12, 'sigmoid'), (2, 'softmax')],
                   metrics=['accuracy'])
train_and_eval(model, X_train, y_train, X_test, y_test,
               score_name='accuracy', score_format='{[1]}',
               batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(20, 'sigmoid'), (2, 'softmax')],
                   metrics=['accuracy'])
train_and_eval(model, X_train, y_train, X_test, y_test,
               score_name='accuracy', score_format='{[1]}',
               batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(6, 'sigmoid'),
                                             (6, 'sigmoid'),
                                             (2, 'softmax')],
                   metrics=['accuracy'])
train_and_eval(model, X_train, y_train, X_test, y_test,
               score_name='accuracy', score_format='{[1]}',
               batch_size=batch_size, nb_epoch=nb_epoch)

print('\n')
model = models.mlp(n_input=75, architecture=[(10, 'sigmoid'),
                                             (10, 'sigmoid'),
                                             (2, 'softmax')],
                   metrics=['accuracy'])
train_and_eval(model, X_train, y_train, X_test, y_test,
               score_name='accuracy', score_format='{[1]}',
               batch_size=batch_size, nb_epoch=nb_epoch)
