import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import linear_model
from data_visualization import plot_data
from data_visualization import plot_data_and_model

dataset_iris = load_iris()

# The first two labels are samples 0 to 99
# The first two column are sepal length and sepal width
X_iris = dataset_iris['data'][:100, :2]
y_iris = dataset_iris['target'][:100]

print("Iris dataset shape:", X_iris.shape)
# N is the number of samples
# d is the number of features
N_iris, d_iris = np.shape(X_iris)

# Splits the dataset into training and testing
"""
Stratify when set to a result will make sure that the ratio of results is the same in your training and testing data.
Suppose you can categorize your results (y) as 0 or 1 and 25% of the results are 0 and 75% are 1. stratify=y will make
sure that your training data contains 25% 0s and 75% 1s (or as close as it can get) and the same for testing data.
from sklearn.model_selection import train_test_split
"""
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3,
                                                    shuffle=True,
                                                    stratify=y_iris,
                                                    random_state=0)

# Plot the data just to see
plot_data(X_iris, y_iris)
plt.show()

# Function that prints results of training and testing
def printAccuracy(predictions, y_data, name_of_set):
    wrong_counter = 0

    for i in range(len(predictions)):
        if predictions[i] != y_data[i]:  # If wrong prediction
            wrong_counter += 1

    score = "%.2f" % (100 - (wrong_counter / len(predictions)) * 100)
    print(name_of_set + ": " + str(score) + "% (" + str(len(predictions) - wrong_counter) + "/" + str(
        len(predictions)) + ")")


logreg_model = linear_model.LogisticRegression()

# Fitting the Iris model
logreg_model.fit(X_train, y_train)
predictions_iris_train = logreg_model.predict(X_train)
predictions_iris_test = logreg_model.predict(X_test)

# Uses print function above
printAccuracy(predictions_iris_train, y_train, "Iris Train Score")
printAccuracy(predictions_iris_test, y_test, "Iris Test Score")



def STEP(v):
    return np.heaviside(v, 0)
    # The above is the same as the following:
    #   return 1 if val > 0 else 0
    # The advantage is that np.heaviside can handle both scalars (numbers) and numpy arrays


class Perceptron():
    def __init__(self, num_features, learning_rate=1.0, activation=STEP,
                 max_iterint=1000,
                 tol=.01,
                 n_iter_no_change=5,
                 shuffle=True,
                 random_state=None,
                 learning_rate_schedule='constant',
                 verbose=False):
        """
        args:
            num_features: The number of input variables or features
            activation: Activation function.
        """
        # d: feature count
        self.d = num_features
        # W: array of weights for each feature
        self.W = np.zeros(num_features, np.float32)
        # b: bias value
        self.b = 0
        # activation: activation function
        self.activation = activation
        # learning_rate:
        self.learning_rate = learning_rate
        # max_epochs: max number of epochs
        self.max_epochs = max_iterint
        # tolerance: limit of the difference between best score and current score
        self.tolerance = tol
        # n_iter_no_change: number of plateauing iterations before changing learning rate
        self.n_iter_no_change = n_iter_no_change
        # shuffle: choose to shuffle data or not
        self.shuffle = shuffle
        # random_state: random_state for the shuffle
        self.random_state = random_state
        # learning_rate_schedule: the type of learning occurring (constant or adaptive)
        self.learning_rate_schedule = learning_rate_schedule
        # verbose: print the epoch #, training score, and current learning_rate after each epoch during fit.
        self.verbose = verbose

    def predict_single(self, sample):
        """
        args:
            features: np array of feature values, length d
        returns:
            returns: predicted label, 0 or 1
        """

        # Make sure input is valid using assert, then compute output
        assert len(sample) == self.d
        output = np.dot(sample, self.W) + self.b

        # Return the result of the activation function, since this is a classification problem
        return self.activation(output)

    def train_single(self, features, label):
        """
        args:
            X: 1D array of features (input)
            y: label (correct output)
            learning_rate: learning rate
        """

        prediction = self.predict_single(features)
        if prediction != label:
            error = label - prediction
            for i in range(0, len(self.W), 1):
                self.W[i] += error * features[i] * self.learning_rate
            self.b += error * self.learning_rate

        return prediction

    def predict(self, X, y):
        predictions = []

        for i in range(len(X)):
            predictions.append(self.train_single(X[i], y[i]))

        return predictions

    def fit(self, initX, inity):
        adaptive = self.learning_rate_schedule == 'adaptive'
        no_improvement_counter = 0
        best_score = 0

        for i in range(0, self.max_epochs, 1):
            wrong_counter = 0

            if self.shuffle:
                X, y = shuffle(initX, inity, random_state=self.random_state)

            predictions = model.predict(X, y)

            for j in range(len(predictions)):
                if predictions[j] != y[j]:
                    # print("Prediction %.2f Correct: %.2f" % (predictions[j], y[j]))
                    wrong_counter += 1

            score = (len(predictions) - wrong_counter) / len(predictions)

            if self.verbose:
                print("--------------------------------------------------------")
                print("Epoch: " + str(i))
                print("Learning Rate: " + str(self.learning_rate))
                print("Score: %.2f" % (score * 100) + "%")

            if score > best_score:
                best_score = score
                no_improvement_counter = 0
            elif (score <= best_score) and (score > best_score - self.tolerance):
                no_improvement_counter += 1
                if no_improvement_counter >= self.n_iter_no_change * 2:
                    break
                if adaptive and (no_improvement_counter == self.n_iter_no_change):
                    self.learning_rate /= 2

        print("Best Score: " + str(best_score * 100) + "%")
        plot_data_and_model(model, X, y)
        plt.show()


model = Perceptron(d_iris, learning_rate=.5, max_iterint=100, tol=.01, n_iter_no_change=2, shuffle=True, random_state=1,
                   learning_rate_schedule='adaptive', verbose=False)
model.fit(X_iris, y_iris)
