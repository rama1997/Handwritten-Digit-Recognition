# Handwritten-Digit-Recognition

Intro Machine Learning Handwritten Digit Recognition using KNN classifer. Uses data from the MNIST dataset.

Recognizes numerical digits using images of handwritten digits from the MNIST dataset via the Nearest Neighbor Classifier.

Displayed classification reports to show accuracy, precision, and recall.

Achieved a recognition accuracy of 97%.

The dataset that I am using for this project comes from the MNIST database which contains 60,000 digits images, ranging from 0-9, that will be used to train our algorithms. The database also contains another 10,000 digits images that I will use as test data. Each image in this dataset is a digit centered in a 28 x 28 gray image, which means each image has 784 pixels of features.

The algorithm that I used to recognize handwritten digits from the MNIST dataset is via the k-Nearest Neighbors Classifier (KNN). The KNN classifier is a simple image classification algorithm that relies on the distance between feature vectors to accurately classify unknown data points, digits in this project, by finding the most common class among the k-closest examples. The most common class is determined by whichever class has the majority of the votes or most votes out of the k-closest neighbors. The KNN classifier generally uses the Euclidean distance to calculate distance between two data points. Generally, if the k-value is too small, the algorithm will be more efficient due to having to account for less number of neighbors but might also lose accuracy as the model becomes too “complex”, thus becoming more affected by noise and outliers. If the value of k is too big, then I risk overfitting and our accuracy will decrease. Less emphasis will be put on individual points and our decision boundary will become simpler and more smooth.

With an accuracy rating of 97%, the KNN classifier proves to be an effective algorithm to recognize handwritten digits. Looking at the PCAs plots for pairs of the MNIST dataset, there seems to be a clear distinction between any two digits which accounts for the high accuracy of the KNN classifier.

Full report pdf can be found in the repo.

