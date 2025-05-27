import sys

sys.path.insert(
    1, '/Users/haas/Documents/Masters/MachineLearning/FYS-STK4155/project3'
)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from activation_functions import *
from FFNN import FFNN
from GridSearch import GridSearch_FFNN_reg
from sklearn.metrics import mean_squared_error
from utils import *

cmt = 1 / 2.54
sns.set_palette('pastel')

Xtrain, Xtest, ytrain, ytest = Ising.load_data()

eta = 1e-3
n_neurons = [2, 3, 4, 5, 6, 7, 8]
n_hidden_layers = [1, 2, 3, 4, 5, 6, 7]
lmbda_values = np.logspace(-8, -2, 7)
n_epochs = 500
activation_hidden = 'sigmoid'
batch_size = 20
gamma = 0.9

print(lmbda_values)

print('sigmoid')
rmse_values, best_lambda, best_n_layers, best_n_neurons = GridSearch_FFNN_reg(
    Xtrain,
    ytrain,
    n_layers=n_hidden_layers,
    n_neurons=n_neurons,
    lambda_values=lmbda_values,
    eta=eta,
    plot_grid=True,
    gamma=0.9,
    activation_hidden=activation_hidden,
    n_epochs=n_epochs,
    batch_size=batch_size,
    k=5,
)

# get best lamba and n_layers and n_neurons and plot predictions on test data

network = FFNN(
    n_hidden_neurons=[best_n_neurons] * best_n_layers,
    task='regression',
    n_epochs=500,
    batch_size=20,
    eta=eta,
    lmbda=best_lambda,
    gamma=gamma,
    activation_hidden=activation_hidden,
)

Xtrain, Xtest, ytrain, ytest = Ising.load_data()

network.fit(Xtrain, ytrain)
pred = network.predict(Xtest)
order = np.argsort(ytest.ravel())
ytest = ytest[order]
pred = pred[order]

# plot predictions
plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(13 * cmt, 12 * cmt))
plt.plot(ytest, label='True Temperature')
plt.plot(pred, label='Predicted Temperature')
plt.legend()
plt.xlabel('Test Sample')
plt.ylabel('Temperature $(J/k_B)$')
# change ticks font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


info = (
    f'_mom{gamma}'
    + '_activ'
    + activation_hidden
    + f'_epoch{n_epochs}_batch{20}_eta{eta}'
)
# plt.tight_layout()
plt.savefig('../figs/regression_FFNN' + info + '.pdf')

# print rmse
print('RMSE: ', mean_squared_error(ytest, pred, squared=False))
# standard deviation of the preditions
print('Standard deviation of the predictions: ', np.std(pred))
