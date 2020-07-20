from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
from sklearn import datasets
from skimage import exposure
from sklearn.decomposition import PCA
from itertools import product
from ggplot import *
import numpy as np
import sklearn
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# load the MNIST digits dataset
from keras.datasets import mnist as MN
(X_train, y_train), (X_test, y_test) = MN.load_data()

#change shape of training dataset arrays from 3 dimentional array to 2 dimensional array
newXtrain = []
for x in X_train:
    newlist = []
    for y in x:
        for z in y:
            newlist.append(z)
    newXtrain.append(newlist)

newXtrain = np.array(newXtrain)
X_train = newXtrain

#change shape of testing dataset arrays from 3 dimentional array to 2 dimensional array
newXtest = []
for x in X_test:
    newlist = []
    for y in x:
        for z in y:
            newlist.append(z)
    newXtest.append(newlist)

newXtest = np.array(newXtest)
X_test = newXtest

# Take 10% of the training data and use that for validation data
(trainData, valData, trainLabels, valLabels) = train_test_split(X_train, y_train, test_size=0.1, random_state=84)

# initialize the testing values of k
startK = 1
endK = 15
kInt = 2
kVals = range(startK, endK, kInt)
accuracies = []

#configure how many of the 60000 train data to use
#currently using a smaller subset of 2000
datasize = 100

# external code from gurus.pyimagesearch.com/lesson-sample-k-nearest-neighbor-classification
# loop over various values of k for the k-Nearest Neighbor classifier
print("From k = " + str(startK) + " to k = " + str(endK) + " with interval of " + str(kInt))
print("Using datasize of " + str(datasize) + "/60000")
for k in range(startK, endK, kInt):
          # train the k-Nearest Neighbor classifier with the current value of `k` and evaluate
          model = KNeighborsClassifier(n_neighbors=k)
          model.fit(trainData[:datasize], trainLabels[:datasize])
          #calculate accuracy
          score = model.score(valData, valLabels)
          print("k=%d, accuracy=%.2f%%" % (k, score * 100))
          #store accuracy
          accuracies.append(score)

# find the value of k that has the largest accuracy
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],accuracies[i] * 100))

# re-train our classifier using the best k value and predict the labels of the test data
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainData[:datasize], trainLabels[:datasize])
predictions = model.predict(X_test)

# show classification reports demonstrating the accuracy of the classifier for each of the digits
print(classification_report(y_test, predictions))

model = KNeighborsClassifier(n_neighbors=kVals[i])
visualizer = ClassificationReport(model, support=True)
visualizer.fit(trainData[:datasize], trainLabels[:datasize])
visualizer.score(X_test, y_test)
g = visualizer.poof()

#class prediction error plot
model = KNeighborsClassifier(n_neighbors=kVals[i])
visualizer = ClassPredictionError(model, support = True)
visualizer.fit(trainData[:datasize], trainLabels[:datasize])
visualizer.score(X_test, y_test)
g = visualizer.poof()

#external code from gist.github.com/amueller/4299381
#plot pairs pca plots
X_train, y_train = trainData[:datasize], trainLabels[:datasize]
pca = PCA(n_components=2)
fig, plots = plt.subplots(10, 10)
fig.set_size_inches(50, 50)
plt.prism()
for i, j in product(range(10), repeat=2):
    if i > j:
        continue
    X_ = X_train[(y_train == i) + (y_train == j)]
    y_ = y_train[(y_train == i) + (y_train == j)]
    X_transformed = pca.fit_transform(X_)
    plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
    plots[i, j].set_xticks(())
    plots[i, j].set_yticks(())

    plots[j, i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
    plots[j, i].set_xticks(())
    plots[j, i].set_yticks(())
    if i == 0:
        plots[i, j].set_title(j)
        plots[j, i].set_ylabel(j)

plt.tight_layout()
plt.savefig("mnist_pairs.png")
