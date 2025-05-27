import sys

import numpy as np
import pandas as pd

sys.path.insert(
    1, '/Users/haas/Documents/Masters/MachineLearning/FYS-STK4155/project3'
)

import random

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from activation_functions import *
from FFNN import FFNN

# import accuracy score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
)
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from utils import *

cmt = 1 / 2.54
sns.set_palette('pastel')

# lenet inspired model
# now we replicate LeNet-5
Xtrain, Xtest, ytrain, ytest = Ising.load_data()
# first reshape Xtrain and Xtest to 4D arrays
Xtrain = Xtrain.reshape(Xtrain.shape[0], 50, 50, 1)
Xtest = Xtest.reshape(Xtest.shape[0], 50, 50, 1)
# create model
model1 = Sequential()
# add model layers
eta = 1e-4
l2 = 0.01
l2_reg = regularizers.l2(l2)
epochs = 1000
model1.add(
    Conv2D(
        6,
        kernel_size=5,
        activation='leaky_relu',
        input_shape=(50, 50, 1),
        kernel_regularizer=l2_reg,
        padding='same',
    )
)
# model.add(AveragePooling2D(pool_size=(4, 4)))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(
    Conv2D(
        16,
        kernel_size=5,
        activation='leaky_relu',
        kernel_regularizer=l2_reg,
        padding='same',
    )
)
model1.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(AveragePooling2D(pool_size=(4, 4)))
model1.add(Flatten())
model1.add(Dense(120, activation='relu', kernel_regularizer=l2_reg))
model1.add(Dense(84, activation='relu', kernel_regularizer=l2_reg))
model1.add(Dense(1, activation='linear'))
# compile model using accuracy to measure model performance
sgd = SGD(learning_rate=eta, decay=5e-6, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=eta, beta_1=0.9, beta_2=0.999, amsgrad=False)

model1.compile(optimizer=adam, loss='mse', metrics='mse')
# train the model
history1 = model1.fit(
    Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=epochs
)   #### change this
#
# evaluate the model
pred1 = model1.predict(Xtest)
order = np.argsort(ytest.ravel())
ytest = ytest[order]
pred1 = pred1[order]

# plot predictions
plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(13 * cmt, 12 * cmt))
plt.plot(ytest, label='True Temperature')
plt.plot(pred1, label='Predicted Temperature')
plt.legend()
plt.xlabel('Test Sample')
plt.ylabel('Temperature $(J/k_B)$')
# change ticks font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


info = (
    f'_l2reg{str(l2).replace(".", "")}'
    + '_eta'
    + str(eta).replace('.', '')
    + f'_epoch{str(epochs)}'
)
plt.tight_layout()
plt.savefig('../figs/TF_CNN' + info + '.pdf')


print('RMSE1: ', mean_squared_error(ytest, pred1, squared=False))
rmse1 = np.sqrt(mean_squared_error(pred1, ytest.ravel())) / np.mean(ytest)
print('rmse over average of ytest with LeNet-5 = ', rmse1)

################################################################

# with regularization
train, Xtest, ytrain, ytest = Ising.load_data()
# first reshape Xtrain and Xtest to 4D arrays
Xtrain = Xtrain.reshape(Xtrain.shape[0], 50, 50, 1)
Xtest = Xtest.reshape(Xtest.shape[0], 50, 50, 1)
# create model
model_reg = Sequential()
model_noreg = Sequential()
# add model layers
eta = 1e-4
l2 = 0.01
l2_reg = regularizers.l2(l2)
epochs = 400
model_reg.add(
    Conv2D(
        6,
        kernel_size=5,
        activation='leaky_relu',
        input_shape=(50, 50, 1),
        kernel_regularizer=l2_reg,
        padding='same',
    )
)
model_noreg.add(
    Conv2D(
        6,
        kernel_size=5,
        activation='leaky_relu',
        input_shape=(50, 50, 1),
        padding='same',
    )
)

model_reg.add(MaxPooling2D(pool_size=(2, 2)))
model_noreg.add(MaxPooling2D(pool_size=(2, 2)))

model_reg.add(
    Conv2D(
        16,
        kernel_size=5,
        activation='leaky_relu',
        kernel_regularizer=l2_reg,
        padding='same',
    )
)
model_noreg.add(
    Conv2D(16, kernel_size=5, activation='leaky_relu', padding='same')
)

model_reg.add(MaxPooling2D(pool_size=(2, 2)))
model_noreg.add(MaxPooling2D(pool_size=(2, 2)))

model_reg.add(Flatten())
model_noreg.add(Flatten())

model_reg.add(Dense(120, activation='relu', kernel_regularizer=l2_reg))
model_noreg.add(Dense(120, activation='relu'))

model_reg.add(Dense(84, activation='relu', kernel_regularizer=l2_reg))
model_noreg.add(Dense(84, activation='relu'))

model_reg.add(Dense(1, activation='linear'))
model_noreg.add(Dense(1, activation='linear'))


# compile model using accuracy to measure model performance
adam = Adam(learning_rate=eta, beta_1=0.9, beta_2=0.999, amsgrad=False)

model_reg.compile(optimizer=adam, loss='mse', metrics='mse')
model_noreg.compile(optimizer=adam, loss='mse', metrics='mse')

# train the model
history_ref = model_reg.fit(
    Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=epochs
)
history_noreg = model_noreg.fit(
    Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=epochs
)

# plot model history
plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(13 * cmt, 12 * cmt))
plt.plot(
    np.log(np.sqrt(history_ref.history['mse'])),
    label=f'Training ($\lambda = $ {str(l2)})',
    color='C0',
)
plt.plot(
    np.log(np.sqrt(history_ref.history['val_mse'])),
    label=f'Validation ($\lambda = $ {str(l2)})',
    color='C0',
    linestyle='--',
)
plt.plot(
    np.log(np.sqrt(history_noreg.history['mse'])),
    label='Training (no regularizer)',
    color='C1',
)
plt.plot(
    np.log(np.sqrt(history_noreg.history['val_mse'])),
    label='Validation (no regularizer)',
    color='C1',
    linestyle='--',
)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Log (rmse)')
# change ticks font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

info = (
    f'_l2reg{str(l2).replace(".", "")}'
    + '_eta'
    + str(eta).replace('.', '')
    + f'_epoch{str(epochs)}'
)
# plt.tight_layout()
plt.savefig('../figs/TF_CNN_reg_vs_noreg' + info + '.pdf')


################################################################
# classification prediction of the Ising model
Xtrain, Xtest, ytrain, ytest = Ising.load_data()

y_test_copy = ytest.copy()

true_tc = 2.269185314213022
ytrain[ytrain < true_tc] = 0
ytrain[ytrain >= true_tc] = 1
ytest[ytest < true_tc] = 0
ytest[ytest >= true_tc] = 1

# shuffle y_copy in the exact same way as Xtrain and Xtest

# use the convolutional neural network to classify the data into two classes
# first reshape Xtrain and Xtest to 4D arrays
Xtrain = Xtrain.reshape(Xtrain.shape[0], 50, 50, 1)
Xtest = Xtest.reshape(Xtest.shape[0], 50, 50, 1)

# create model
model = Sequential()
# add model layers
l2_reg = regularizers.l2(0.01)
model.add(
    Conv2D(
        6,
        kernel_size=5,
        activation='leaky_relu',
        input_shape=(50, 50, 1),
        kernel_regularizer=l2_reg,
        padding='same',
    )
)
# model.add(AveragePooling2D(pool_size=(4, 4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(
    Conv2D(
        16,
        kernel_size=5,
        activation='leaky_relu',
        kernel_regularizer=l2_reg,
        padding='same',
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(AveragePooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(120, activation='relu', kernel_regularizer=l2_reg))
model.add(Dense(84, activation='relu', kernel_regularizer=l2_reg))
model.add(Dense(1, activation='sigmoid'))

# compile model using accuracy to measure model performance
sgd = SGD(learning_rate=0.001, decay=5e-6, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# train the model
history = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=400)

# evaluate the model
pred = model.predict(Xtest)
soft_pred = pred
hard_pred = np.round(pred)
acc = accuracy_score(ytest, hard_pred)
print('accuracy with CNN = ', acc)
# accuracy with CNN =  0.9920948616600791

# plot confusion matrix with seaborn
plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(13 * cmt, 12 * cmt))
cm = confusion_matrix(ytest, hard_pred)
sns.heatmap(cm, annot=True, fmt='d')
# remove colorbar
plt.gca().collections[0].colorbar.remove()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# label as true positive, true negative, false positive, false negative
plt.text(
    0.5,
    0.3,
    'True positive',
    ha='center',
    va='center',
    color='black',
    fontsize=12,
)
plt.text(
    1.5,
    0.3,
    'False positive',
    ha='center',
    va='center',
    color='white',
    fontsize=12,
)
plt.text(
    0.5,
    1.3,
    'False negative',
    ha='center',
    va='center',
    color='white',
    fontsize=12,
)
plt.text(
    1.5,
    1.3,
    'True negative',
    ha='center',
    va='center',
    color='black',
    fontsize=12,
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], ['Above', 'Below'])
plt.yticks([0.5, 1.5], ['Above', 'Below'])


plt.savefig('../figs/TF_CNN_confusion_matrix.pdf')

################################################################
# plot the probabilities as function of temperature
plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(13 * cmt, 12 * cmt))
plt.plot(y_test_copy, soft_pred, 'o')
plt.xlabel('Temperature (T) $J/k_B$')
plt.ylabel('$P(T > T_c)$')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# highlight the point where the probability is closest to 0.5
plt.plot(
    y_test_copy[np.argmin(np.abs(soft_pred - 0.5))],
    soft_pred[np.argmin(np.abs(soft_pred - 0.5))],
    'ro',
)
# print its temperature in the plot figure

plt.text(
    y_test_copy[np.argmin(np.abs(soft_pred - 0.5))],
    soft_pred[np.argmin(np.abs(soft_pred - 0.5))],
    str(y_test_copy[np.argmin(np.abs(soft_pred - 0.5))]),
)

plt.savefig('../figs/TF_CNN_probabilities.pdf')
