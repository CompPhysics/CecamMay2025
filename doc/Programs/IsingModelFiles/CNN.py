import numpy as np

from activation_functions import (
    leaky_reLU_derivative,
    leakyReLU,
    reLU,
    reLU_derivative,
)
from utils import *


class simple_Conv2d:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(
            num_filters, filter_size, filter_size
        ) / (filter_size * filter_size)

    def create_patches(
        self, grid
    ):   # grid is more generally an image, but in out case we already come with a matrix form
        height, width = grid.shape
        self.grid = grid
        for i in range(height - self.filter_size + 1):
            for j in range(width - self.filter_size + 1):
                patch = grid[
                    i : (i + self.filter_size), j : (j + self.filter_size)
                ]
                yield patch, i, j

    def forward_prop(self, grid):
        height, width = grid.shape
        convolved = np.zeros(
            (
                height - self.filter_size + 1,
                width - self.filter_size + 1,
                self.num_filters,
            )
        )
        for patch, i, j in self.create_patches(grid):
            convolved[i, j] = np.sum(
                patch * self.filters, axis=(1, 2)
            )   # here is the convolution!!!
        return convolved

    def back_prop(
        self, dL_dout, learning_rate
    ):   # dL_dout is the gradient of the loss with respect to the output of the convolution.
        dL_dF_params = np.zeros(self.filters.shape)
        for patch, i, j in self.create_patches(self.grid):
            for f in range(self.num_filters):
                dL_dF_params[f] += dL_dout[i, j, f] * patch

        # updating the parameters
        self.filters -= learning_rate * dL_dF_params
        return dL_dF_params


class Max_Pool2d:
    def __init__(self, pool_size):
        self.pool_size = pool_size   # after pooling the image will be reduced by 1/pool_size

    def create_new_patches(self, grid):
        new_height = grid.shape[0] // self.pool_size
        new_width = grid.shape[1] // self.pool_size
        self.grid = grid
        for i in range(new_height):
            for j in range(new_width):
                new_patch = grid[
                    (i * self.pool_size) : (
                        i * self.pool_size + self.pool_size
                    ),
                    (j * self.pool_size) : (
                        j * self.pool_size + self.pool_size
                    ),
                ]
                yield new_patch, i, j

    def forward_prop(self, grid):
        height, width, num_filters = grid.shape
        pooled = np.zeros(
            (height // self.pool_size, width // self.pool_size, num_filters)
        )
        for patch, i, j in self.create_new_patches(grid):
            pooled[i, j] = np.amax(patch, axis=(0, 1))
        return pooled

    def back_prop(
        self, dL_dout
    ):   # dL_dout is the gradient of the loss from the relu in our case
        dL_dmax = np.zeros(self.grid.shape)
        for patch, i, j in self.create_new_patches(self.grid):
            height, width, num_filters = patch.shape
            patch_max = np.amax(patch, axis=(0, 1))
            for h in range(height):
                for w in range(width):
                    for f in range(num_filters):
                        if patch[h, w, f] == patch_max[f]:
                            dL_dmax[
                                i * self.pool_size + h,
                                j * self.pool_size + w,
                                f,
                            ] = dL_dout[i, j, f]
        return dL_dmax


class ReLU:
    def __init__(self, input_nodes, output_nodes):
        # initialize weights and biases
        self.weights = np.random.randn(input_nodes, output_nodes) / input_nodes
        self.biases = np.zeros(output_nodes)

    def forward_prop(self, grid):
        self.orig_grid_shape = grid.shape
        flat_grid = grid.flatten()
        self.flat_grid = flat_grid
        output_val = flat_grid @ self.weights + self.biases
        self.out = output_val
        relu_val = reLU(output_val)
        return relu_val

    def back_prop(self, dL_dout, learning_rate):

        # gradients with respect to output (z)
        dy_dz = np.array(self.out, copy=True)
        dy_dz = reLU_derivative(self.out)

        # gradients with respect to weights and biases
        dz_dw = self.flat_grid
        dz_db = 1
        dz_dflat_grid = self.weights

        # gradients with respect to input
        dL_dflat_grid = dL_dout * dy_dz
        dL_dw = dz_dw[np.newaxis].T @ dL_dflat_grid[np.newaxis]
        dL_db = dL_dflat_grid * dz_db
        dL_dgrid = dL_dflat_grid @ dz_dflat_grid.T

        # update weights and biases
        self.weights -= learning_rate * dL_dw
        self.biases -= learning_rate * dL_db

        return dL_dgrid.reshape(self.orig_grid_shape)


class Leaky_ReLu:
    def __init__(self, input_nodes, output_nodes):
        # initialize weights and biases
        self.weights = np.random.randn(input_nodes, output_nodes) / input_nodes
        self.biases = np.zeros(output_nodes)

    def forward_prop(self, grid):
        self.orig_grid_shape = grid.shape
        flat_grid = grid.flatten()
        self.flat_grid = flat_grid
        output_val = flat_grid @ self.weights + self.biases
        self.out = output_val
        relu_val = leakyReLU(output_val)
        return relu_val

    def back_prop(self, dL_dout, learning_rate):
        # gradients with respect to output (z)
        dy_dz = np.array(self.out, copy=True)
        dy_dz = leaky_reLU_derivative(self.out)

        # gradients with respect to weights and biases
        dz_dw = self.flat_grid
        dz_db = 1
        dz_dflat_grid = self.weights

        # gradients with respect to input
        dL_dflat_grid = dL_dout * dy_dz
        dL_dw = dz_dw[np.newaxis].T @ dL_dflat_grid[np.newaxis]
        dL_db = dL_dflat_grid * dz_db
        dL_dgrid = dL_dflat_grid @ dz_dflat_grid.T

        # update weights and biases
        self.weights -= learning_rate * dL_dw
        self.biases -= learning_rate * dL_db

        return dL_dgrid.reshape(self.orig_grid_shape)


class my_CNN:
    def __init__(
        self,
        num_filters=2,
        filter_size=4,
        pool_size=2,
        num_hidden=10,
        epochs=10,
    ):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.num_hidden = num_hidden
        self.epochs = epochs

        self.first_relu_dim = (
            (50 - self.filter_size + 1) // self.pool_size
        ) ** 2 * self.num_filters

        self.conv1 = simple_Conv2d(
            num_filters, filter_size
        )   # 2 filters, 4x4 filter
        self.pool1 = Max_Pool2d(pool_size)   # 2x2 pooling
        self.relu1 = ReLU(
            self.first_relu_dim, num_hidden
        )   # 23x23x2 input, 10 output # 23 = 50 - 4 + 1, 2 = 2 filters
        self.relu2 = ReLU(num_hidden, 1)   # 10 input, 1 output

    def cnn_forward_prop(self, grid, label):
        out_p = self.conv1.forward_prop(grid)
        out_p = self.pool1.forward_prop(out_p)
        out_p = self.relu1.forward_prop(out_p)
        out_p = self.relu2.forward_prop(out_p)

        # calculate the loss as mean squared error
        loss = np.mean((out_p - label) ** 2)
        return out_p, loss

    def training_cnn(self, grid, label, lr=0.001):

        # forward propagation
        out, loss = self.cnn_forward_prop(grid, label)
        # initial gradient
        grad = 2 * (out - label)

        # backpropagation
        grad = self.relu2.back_prop(grad, lr)
        grad = self.relu1.back_prop(grad, lr)
        grad = self.pool1.back_prop(grad)
        grad = self.conv1.back_prop(grad, lr)

        return loss

    # train the model
    def fit(self, Xtrain, ytrain):
        for epoch in range(self.epochs):
            cumul_loss = 0
            for i, (grid, label) in enumerate(zip(Xtrain, ytrain)):
                # here it is best we unravel the grid
                grid = grid.reshape(50, 50)
                cumul_loss += self.training_cnn(grid, label, lr=0.001)

    def predict(self, Xtest):
        # forward propagation
        pred = np.array([])
        for grid in Xtest:
            # here it is best we unravel the grid
            grid = grid.reshape(50, 50)
            out_p, _ = self.cnn_forward_prop(grid, 0)
            pred = np.append(pred, out_p)
        return pred
