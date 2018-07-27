
#HANDWRITING RECOGNITION USING SVM(SUPPORT VECTOR MACHINES) AND REGRESSION

#import datasets
from sklearn import datasets
iris=datasets.load_iris()
digits=datasets.load_digits()
print(digits.data)
digits.target
digits.images[0]
#Learning and Predicting
from sklearn import svm
clf=svm.SVC(gamma=0.001, C=100)
print(clf)
clf.fit(digits.data[:-1], digits.target[:-1])  
clf.predict(digits.data[-1:])
clf.fit(digits.data[:-1], digits.target[:-1])
#Model Persistence
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)  
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])
from sklearn.externals import joblib
joblib.dump(clf,'filename.pkl')
#Refitting and updating parameters
import numpy as np
from sklearn import random_projection
rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
X.dtype
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
X_new.dtype
#Multiclass vs multilabel fitting
from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = SVC()
clf.fit(iris.data, iris.target)
list(clf.predict(iris.data[:3]))
clf.fit(iris.data,iris.target_names[iris.target])
list(clf.predict(iris.data[:3]))

import numpy as np
from sklearn.svm import SVC
rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)
clf = SVC()
clf.set_params(kernel='linear').fit(X, y)
clf.predict(X_test)
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
classif.fit(X, y).predict(X)
y = LabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)
from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()
plt.scatter(expected,predicted)
plt.xlabel("Expected")
plt.ylabel("Predicted")
plt.show()

