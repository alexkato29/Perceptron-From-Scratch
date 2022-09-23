import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris

dataset_iris = load_iris()

# The first two labels are samples 0 to 99
# The first two column are sepal length and sepal width
X_iris = dataset_iris['data'][:100, :2]
y_iris = dataset_iris['target'][:100]

print("Iris dataset shape:", X_iris.shape)
# N is the number of samples
# d is the number of features
N_iris, d_iris = np.shape(X_iris)

# Take a peek at every tenth data point
print(X_iris[::10])
print(y_iris[::10])


COLORS = ["green", "red"]

# A handy function to plot 2 features
def plot_data(X, y, feat_x=0, feat_y=1, xlim=None, ylim=None):
    featx = X[:, feat_x]
    featy = X[:, feat_y]

    # Plot the data points using the two features as x and y-axis
    plt.scatter(x=featx, y=featy, c=y, s=10, cmap=ListedColormap(COLORS))

    # Adjust the limits of the axis to display all points
    if xlim is None:
        plt.xlim(np.min(featx) - 0.2, np.max(featx) + 0.2)
    else:
        plt.xlim(xlim[0], xlim[1])
    if ylim is None:
        plt.ylim(np.min(featy) - 0.2, np.max(featy) + 0.2)
    else:
        plt.ylim(ylim[0], ylim[1])


def plot_data_and_model(model, X, y, feat_x=0, feat_y=1, xlim=None, ylim=None):
    # Plot the data scatter
    plot_data(X, y, feat_x, feat_y, xlim, ylim)

    # Now plot the fitted line. We need only two points to plot the line
    if xlim == None:
        plot_x = np.array([np.min(X[:, feat_x]) - 0.5, np.max(X[:, feat_x]) + 0.5])
    else:
        plot_x = np.array(xlim)
    if 0.0 in model.W[1] * (model.W[0] * plot_x + model.b):
        return
    plot_y = - 1 / model.W[1] * (
                model.W[0] * plot_x + model.b)  # comes from, w0*x + w1*y + b = 0 then y = (-1/w1) (w0*x + b)
    plt.plot(plot_x, plot_y, color='k', linewidth=2)
