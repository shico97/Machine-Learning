import inline as inline
import pandas as pd
from statistics import mean
from statistics import stdev
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np

path_to_file = "D:\Pycharm Projects\ML2\house_data_complete.csv"
data = pd.read_csv(path_to_file).dropna()

# Dividing the data into Training set, Cross validation and Data set

data = data.drop(['id', 'date'], axis=1)

for i in range(3):

    Training_set, Cross_validation, Test_set = np.split(data.sample(frac=1), [int(.6 * len(data)), int(.8 * len(data))])

    # Visualizing Part of the data and plotting them

    y = Training_set["price"]
    x = Training_set["bedrooms"]

    plt.plot(x, y, 'ro', ms=8, mec='k')
    plt.xlabel('Bedrooms')
    plt.ylabel('Prices')
    plt.show()

    # Normalizing the data
    column_count_update = sum(1 for line in data)
    print(column_count_update)

    Number_of_Features = column_count_update - 1
    Normalization = []
    y = Training_set["price"]

    Training_set = Training_set.drop(["price"], axis=1)

    Training_set = Training_set.values
    Cross_validation = Cross_validation.values
    Test_set = Test_set.values
    y = y.values

    m = Training_set[:, 2].size
    i = 0

    Normalization = Training_set.copy()
    mu = np.zeros(Training_set.shape[1], dtype=int)
    sigma = np.zeros(Training_set.shape[1])

    for i in range(Number_of_Features):
        mu[i] = np.mean(Training_set[:, i])
        sigma[i] = np.std(Training_set[:, i])
        Normalization[:, i] = (Training_set[:, i] - mu[i]) / sigma[i]

    Training_set = np.concatenate([np.ones((m, 1)), Normalization], axis=1)
    theta = np.zeros(Number_of_Features + 1)


    def CostFunction(x, y, theta, lambda_, h):
        J = 1 / (2 * m) * np.sum(np.square(h - y)) + (lambda_ / (2 * m)) * np.sum(np.square(theta))
        return J


    def GradientDescent1(x, y, theta, alpha, num_iters, lambda_):
        theta = theta.copy()
        J_history = []
        for i in range(num_iters):
            h1 = np.dot(Training_set, theta)
            # theta = theta - (alpha / m) * ((np.dot(x.T, h1 - y)) + lambda_ * theta)
            theta = theta * (1 - (alpha * lambda_) / m) - ((alpha / m) * (np.dot(Training_set.T, h1 - y)))
            J_history.append(CostFunction(x, y, theta, lambda_, h1))

        return theta, J_history


    def GradientDescent2(x, y, theta, alpha, num_iters, lambda_):
        theta = theta.copy()
        J_history = []
        for i in range(num_iters):
            h2 = np.dot(np.power(Training_set, 2), theta)
            # theta = theta - (alpha / m) * ((np.dot(x.T, h2 - y)) + lambda_ * theta)
            theta = theta * (1 - (alpha * lambda_) / m) - ((alpha / m) * (np.dot(Training_set.T, h2 - y)))
            J_history.append(CostFunction(x, y, theta, lambda_, h2))

        return theta, J_history


    def GradientDescent3(x, y, theta, alpha, num_iters, lambda_):
        theta = theta.copy()
        J_history = []
        Training_set[:, 2] = np.power(Training_set[:, 2], 2)
        for i in range(num_iters):
            h3 = np.dot(Training_set, theta)
            # theta = theta - (alpha / m) * ((np.dot(x.T, h3 - y)) + lambda_ * theta)
            theta = theta * (1 - (alpha * lambda_) / m) - ((alpha / m) * (np.dot(Training_set.T, h3 - y)))
            J_history.append(CostFunction(x, y, theta, lambda_, h3))

        return theta, J_history


    alpha = 0.01
    alpha2 = 0.003
    Number_Of_Iterations = 100
    lambda_ = 2.00

    # calling the Gradient Descent to get the Theta and mean error square

    theta1, J_history1 = GradientDescent1(Training_set, y, theta, alpha, Number_Of_Iterations, lambda_)
    theta2, J_history2 = GradientDescent2(Training_set, y, theta, alpha2, Number_Of_Iterations, lambda_)
    theta3, J_history3 = GradientDescent3(Training_set, y, theta, alpha, Number_Of_Iterations, lambda_)

    # plotting the graph

    plt.plot(np.arange(len(J_history1)), J_history1, 'r', lw=2, label='h1')
    plt.plot(np.arange(len(J_history2)), J_history2, 'b', lw=2, label='h2')
    plt.plot(np.arange(len(J_history3)), J_history3, 'g', lw=2, label='h3')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

