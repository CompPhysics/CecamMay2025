import sys

sys.path.insert(
    1, '/Users/haas/Documents/Masters/MachineLearning/FYS-STK4155/project3'
)

from CNN import simple_Conv2d, Max_Pool2d, ReLU, my_CNN
from utils import *
import matplotlib.pyplot as plt
import random
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
)
from GridSearch import GridSearch_my_CNN_reg
import seaborn as sns

cmt = 1 / 2.54
# use pastel seaborn pallete
sns.set_palette('pastel')

random.seed(42)
Xtrain, Xtest, ytrain, ytest = Ising.load_data()

# Showing the affects of the filters parameters

# plot before and after convolution for multiple filter sizes
plt.subplot(1, 4, 1)
#
plt.rcParams.update({'font.size': 30})
#
## make plot bigger
plt.axis("off")
plt.imshow(Xtrain[0].reshape(50,50))
plt.title("Original")

for i in range(1, 4):

    conv = simple_Conv2d(1, i)
    out = conv.forward_prop(Xtrain[0].reshape(50,50))
    plt.subplot(1, 4, i + 1)


    plt.imshow(out[:,:,0])
    plt.title("Filter size = " + str(i))
    # turn off the axis
    plt.axis("off")
# add horizontal colorbar
plt.colorbar(orientation="horizontal", cax=plt.axes([0.15, 0.1, 0.7, 0.06])) # [left, bottom, width, height]
# clim -1 to 1
plt.clim(-1, 1)
#
## make the plot bigger
plt.gcf().set_size_inches(18, 6.7)
## tight layout
plt.tight_layout()
## save the plot
plt.savefig("../figs/convolution_filter_sizes.pdf")
#
## exploring the number of filters
#
plt.subplot(1, 4, 1)
plt.rcParams.update({'font.size': 30})
## make plot bigger
plt.axis("off")
plt.imshow(Xtrain[0].reshape(50,50))
plt.title("Original")
for i in range(1, 4):
   conv = simple_Conv2d(i, 3)
   out = conv.forward_prop(Xtrain[0].reshape(50,50))
   plt.subplot(1, 4, i+1)
   plt.imshow(out[:,:,0])
   plt.title("Num. filters = " + str(i))
   plt.axis("off")
#
## add horizontal colorbar
plt.colorbar(orientation="horizontal", cax=plt.axes([0.15, 0.1, 0.7, 0.06])) # [left, bottom, width, height]
## clim -1 to 1
plt.clim(-1, 1)
#
## make the plot bigger
plt.gcf().set_size_inches(18, 6.7)
## tight layout
plt.tight_layout()
plt.savefig("../figs/convolution_filter_number.pdf")
#
#
## exploring the maxpooling in action
conv = simple_Conv2d(1, 3)
out = conv.forward_prop(Xtrain[0].reshape(50,50))
#
plt.subplot(1, 4, 1)
plt.rcParams.update({'font.size': 30})
#
## make plot bigger
plt.axis("off")
#
plt.imshow(Xtrain[0].reshape(50,50))
plt.title("Convolution (3, 1)")
#
#
pool = Max_Pool2d(2)
out2 = pool.forward_prop(out)
plt.imshow(out2[:,:,0])
#
for i in range(1, 4):
   plt.subplot(1, 4, i + 1)

   pool = Max_Pool2d(i+1)
   out2 = pool.forward_prop(out)
   plt.imshow(out2[:,:,0])
   plt.title("Pool size = " + str(i+1))
   # turn off the axis
   plt.axis("off")
## add horizontal colorbar
plt.colorbar(orientation="horizontal", cax=plt.axes([0.15, 0.1, 0.7, 0.06])) # [left, bottom, width, height]
## clim -1 to 1
plt.clim(-1, 1)
#
## make the plot bigger
plt.gcf().set_size_inches(18, 6.7)
## tight layout
plt.tight_layout()
plt.savefig("../figs/convolution_maxpooling.pdf")
exit()
### Notice the edges are somewhat preserved while preserving a lot of the computational req!!!

######### Now lets build a CNN!

### define the  best model
X, y = Ising.load_data(split=False)
n_num_filters = [1, 2, 3]
n_filter_sizes = [2, 3, 4]
n_pool_sizes = [2, 3, 4]
(
    rmse_values,
    best_num_filters,
    best_filter_size,
    best_pool_size,
) = GridSearch_my_CNN_reg(
    X, y, n_num_filters, n_filter_sizes, n_pool_sizes, n_epochs=5, k=3
)
print('rmse with cross validation: ', rmse_values)
print('best number of filters: ', best_num_filters)
print('best filter size: ', best_filter_size)
print('best pool size: ', best_pool_size)


# now we test the best model
pred = np.array([])
Xtrain, Xtest, ytrain, ytest = Ising.load_data()

model = my_CNN(
    num_filters=best_num_filters,
    filter_size=best_filter_size,
    pool_size=best_pool_size,
    num_hidden=10,
    epochs = 10,
)
model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)

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
plt.savefig('../figs/my_best_CNN_predictions.pdf')

rmse = np.sqrt(mean_squared_error(pred, ytest.ravel()))
print('rmse with myConv = ', rmse)
# rmse over average of ytest
rmse = mean_squared_error(pred, ytest.ravel(), squared=False) / np.mean(ytest)
print('rmse over average of ytest with myConv= ', rmse)

"""
best number of filters:  3
best filter size:  2
best pool size:  3
rmse with myConv =  0.19774642484200544
rmse over average of ytest with myConv=  0.08961264841753816
"""