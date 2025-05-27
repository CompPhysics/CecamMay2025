import sys
import time

sys.path.insert(
    1, '/Users/haas/Documents/Masters/MachineLearning/FYS-STK4155/project3'
)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from CNN import Max_Pool2d, ReLU, my_CNN, simple_Conv2d
from FFNN import FFNN
from ResampleMethods import *

cmt = 1 / 2.54


def GridSearch_FFNN_reg(
    X,
    y,
    n_layers,
    n_neurons,
    lambda_values,
    eta=0.001,
    plot_grid=True,
    gamma=0.9,
    activation_hidden='reLU',
    n_epochs=500,
    batch_size=20,
    k=5,
):

    rmse_values = np.zeros((len(n_layers), len(n_neurons), len(lambda_values)))
    r2_values = np.zeros((len(n_layers), len(n_neurons), len(lambda_values)))

    for i, L in enumerate(n_layers):
        for j, n in enumerate(n_neurons):
            for l, lmbda in enumerate(lambda_values):
                print(
                    f'Computing with {L} layers and {n} neurons and lambda={lmbda}.'
                )
                network = FFNN(
                    n_hidden_neurons=[n] * L,
                    task='regression',
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    eta=eta,
                    lmbda=lmbda,
                    gamma=gamma,
                    activation_hidden=activation_hidden,
                )

                mse, r2 = CrossValidation_regression(network, X, y, k=k)

                rmse_values[i][j][l] = np.sqrt(mse)
                r2_values[i][j][l] = r2

    if plot_grid:
        # make a 3d scatter plot with the MSE values as color
        fig = plt.figure(figsize=(13 * cmt, 12 * cmt))
        ax = fig.add_subplot(111, projection='3d')
        cm = plt.get_cmap('magma')

        for i, L in enumerate(n_layers):
            for j, n in enumerate(n_neurons):
                ax.scatter(
                    [L] * len(lambda_values),
                    [n] * len(lambda_values),
                    np.log10(lambda_values),
                    c=rmse_values[i][j],
                    vmin=np.min(rmse_values),
                    vmax=np.max(rmse_values),
                    cmap=cm,
                    s=100,
                )
                # add colorbar
        # put the colorbar on the bottom

        cbar = fig.colorbar(
            ax.collections[0],
            ax=ax,
            orientation='vertical',
            location='left',
            pad=0,
        )
        # normalize colorbar to the range of the MSE values
        cbar.mappable.set_clim(np.min(rmse_values), np.max(rmse_values))

        cbar.set_label('RMSE')

        ax.set_ylabel('Layers')
        ax.set_xlabel('Neurons')
        # put z lable to the left
        ax.set_zlabel('$\log_{10}(\lambda$)')
        # mark the minimum as an o and display mse value as text
        min_rmse = np.min(rmse_values)
        min_rmse_index = np.where(rmse_values == min_rmse)
        ax.scatter(
            n_layers[min_rmse_index[0][0]],
            n_neurons[min_rmse_index[1][0]],
            np.log10(lambda_values[min_rmse_index[2][0]]),
            s=120,
            marker='o',
            facecolors='none',
            edgecolors='r',
        )
        ax.text(
            n_layers[min_rmse_index[0][0]],
            n_neurons[min_rmse_index[1][0]],
            np.log10(lambda_values[min_rmse_index[2][0]]),
            f'RMSE={min_rmse:.3f}',
            color='red',
            size=10,
        )

        info = (
            f'_mom{gamma}'
            + '_activ'
            + activation_hidden
            + f'_epoch{n_epochs}_batch{batch_size}_eta{eta}'
        )
        plt.tight_layout()
        # adjust tight layout to make room for colorbar
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2
        )
        plt.savefig('../figs/gridsearch_FFNN_reg_RMSE' + info + '.pdf')

        # fig, ax = plt.subplots(figsize = (13*cmt, 12*cmt))
        # sns.heatmap(
        #    r2_values,
        #    annot=True,
        #    ax=ax,
        #    cmap="viridis",
        #    cbar_kws={'label': '$R^2$'},
        #    yticklabels=n_layers,
        #    xticklabels=n_neurons,
        #    zticklabels=np.round(np.log10(lambda_values), 2)
        #    )
        ##ax.set_title("$R^2$")
        # ax.set_ylabel("Layers")
        # ax.set_xlabel("Neurons")
        # ax.set_zlabel("$\log_{10}(\lambda$)")
        # plt.tight_layout()
        # plt.savefig("../figs/gridsearch_FFNN_reg_R2" + info + ".pdf")

    return (
        rmse_values,
        lambda_values[min_rmse_index[2][0]],
        n_layers[min_rmse_index[0][0]],
        n_neurons[min_rmse_index[1][0]],
    )


def GridSearch_my_CNN_reg(
    X,
    y,
    n_num_filters,
    n_filter_sizes,
    n_pool_sizes,
    plot_grid=True,
    n_epochs=10,
    k=5,
):

    rmse_values = np.zeros(
        (len(n_num_filters), len(n_filter_sizes), len(n_pool_sizes))
    )
    r2_values = np.zeros(
        (len(n_num_filters), len(n_filter_sizes), len(n_pool_sizes))
    )

    for i, num_filter in enumerate(n_num_filters):
        # measure time
        start = time.time()
        for j, filter_size in enumerate(n_filter_sizes):
            for l, pool_size in enumerate(n_pool_sizes):
                network = my_CNN(
                    num_filters=num_filter,
                    filter_size=filter_size,
                    pool_size=pool_size,
                    num_hidden=10,
                    epochs=n_epochs,
                )
                # time for 1 iteration

                mse, r2 = CrossValidation_regression(network, X, y, k=k)
                end = time.time()
                print(
                    '===> Progress:',
                    (i + 1) * (j + 1) * (l + 1),
                    'out of',
                    len(n_num_filters)
                    * len(n_filter_sizes)
                    * len(n_pool_sizes),
                )
                print(
                    '### Estimated time left (min)',
                    (end - start)
                    * (
                        len(n_num_filters)
                        * len(n_filter_sizes)
                        * len(n_pool_sizes)
                        - ((i + 1) * (j + 1) * (l + 1))
                    )
                    / 60,
                )
                rmse_values[i][j][l] = np.sqrt(mse)
                r2_values[i][j][l] = r2

    if plot_grid:
        # make a 3d scatter plot with the MSE values as color
        fig = plt.figure(figsize=(13 * cmt, 12 * cmt))
        ax = fig.add_subplot(111, projection='3d')
        cm = plt.get_cmap('magma')

        # normalize the color to the range of the MSE values
        # normalize = matplotlib.colors.Normalize(vmin=np.min(rmse_values), vmax=np.max(rmse_values))
        for i, num_filter in enumerate(n_num_filters):
            for j, filter_size in enumerate(n_filter_sizes):
                ax.scatter(
                    [num_filter] * len(n_pool_sizes),
                    [filter_size] * len(n_pool_sizes),
                    n_pool_sizes,
                    c=rmse_values[i][j],
                    vmin=np.min(rmse_values),
                    vmax=np.max(rmse_values),
                    cmap=cm,
                    s=100,
                )

        cbar = fig.colorbar(
            ax.collections[0],
            ax=ax,
            orientation='vertical',
            location='left',
            pad=0,
        )
        # normalize colorbar to the range of the MSE values
        cbar.mappable.set_clim(np.min(rmse_values), np.max(rmse_values))

        cbar.set_label('RMSE')

        ax.set_ylabel('Filter size')
        ax.set_xlabel('Number of filters')
        ax.set_zlabel('Pool size')
        min_rmse = np.min(rmse_values)
        min_rmse_index = np.where(rmse_values == min_rmse)
        ax.scatter(
            n_num_filters[min_rmse_index[0][0]],
            n_filter_sizes[min_rmse_index[1][0]],
            n_pool_sizes[min_rmse_index[2][0]],
            s=120,
            marker='o',
            facecolors='none',
            edgecolors='r',
        )
        ax.text(
            n_num_filters[min_rmse_index[0][0]],
            n_filter_sizes[min_rmse_index[1][0]],
            n_pool_sizes[min_rmse_index[2][0]],
            f'RMSE={min_rmse:.3f}',
            color='r',
            fontsize=12,
        )

        info = f'_epochs{n_epochs}' + '_activ' + '_relu'
        plt.tight_layout()
        # adjust tight layout to make room for colorbar
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2
        )
        plt.savefig('../figs/gridsearch_myCNN_reg_MSE' + info + '.pdf')

    return (
        rmse_values,
        n_num_filters[min_rmse_index[0][0]],
        n_filter_sizes[min_rmse_index[1][0]],
        n_pool_sizes[min_rmse_index[2][0]],
    )
