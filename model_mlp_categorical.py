from __future__ import print_function
from utils import dataset_from_folder
from model_utils import prepare_pixelized_dataset, train_and_eval
import models
import numpy as np
import model_utils


np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_epoch = 30
window = (5, 5)
stride = 5

dataset = dataset_from_folder('./plzen/train')
train_data = prepare_pixelized_dataset(dataset, window,
                                       stride=stride,
                                       regression=False,
                                       image_by_image=True)
X_train, y_train = model_utils.reshape_dataset(train_data, window,
                                               regression=False)
X_train = np.reshape(X_train, (X_train.shape[0], -1))

dataset = dataset_from_folder('./plzen/test')
test_data = prepare_pixelized_dataset(dataset, window,
                                      stride=stride,
                                      regression=False,
                                      image_by_image=True)
X_test, y_test = model_utils.reshape_dataset(test_data, window,
                                             regression=False)

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
