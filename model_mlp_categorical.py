from __future__ import print_function

from utils import dataset_from_folder, bit_to_two_cls
from model_utils import prepare_pixelized_dataset, train_and_eval
import models

import numpy as np
np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_epoch = 50

verbose = 0

loss = 'categorical_crossentropy'

dataset = dataset_from_folder('./plzen/train')
X_train, y_train = prepare_pixelized_dataset(dataset,
                                             y_applied_function=bit_to_two_cls)

dataset = dataset_from_folder('./plzen/test')
X_test, y_test = prepare_pixelized_dataset(dataset,
                                           y_applied_function=bit_to_two_cls)

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
