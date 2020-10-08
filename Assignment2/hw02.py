import numpy as np
from sklearn import svm
from skimage import feature as ft
import sklearn
import pickle
import pandas as pd

image = np.load('img.npy')
features_test = np.load('features_test.npy')
labels_test = np.load('labels_test.npy')

#extract the feature from one image
feature = ft.hog(image, orientations=18, feature_vector=True)
print('The feature vector of one image is as follow:')
print(feature)

#load our trained system
with open('clf.pickle','rb') as f:
    clf = pickle.load(f)

#run classifier
pred = clf.predict(features_test)

#generate classification report
report = sklearn.metrics.classification_report(labels_test, pred)
print('The classification report is as follow:')
print(report)

#generate confusion matrix
result = sklearn.metrics.confusion_matrix(labels_test, pred) 
cm = pd.DataFrame(result)
print('The confusion matrix is as follow:')
print(cm)
