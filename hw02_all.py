import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
from sklearn import naive_bayes as nb
import sklearn
from skimage import feature as ft #hog
import cv2
import time
import pandas as pd
import pickle

time_start = time.time() #start timing

data_set = np.load("data_set.npy")
class_set = np.load('class_set.npy')
X_train, X_test, y_train, y_test = ms.train_test_split(data_set, class_set, train_size=0.9)
cv_res = []

m_list = [4, 6, 9, 12, 18]
# Corresponding to 45°,30°,20°,15° and 10°
use_time = []
time_cv_st = time.time()
for m in m_list:
    features = ft.hog(X_train[0], orientations=m, feature_vector=True) #feature extration
    for i in range(1, X_train.shape[0]):
        features = np.vstack((features, ft.hog(X_train[i], orientations=m, feature_vector=True)))
    clf = sklearn.naive_bayes.GaussianNB()
    res = ms.cross_val_score(clf, features, y_train, cv = 10, scoring='accuracy')
    cv_res.append(np.mean(res))
best_m = m_list[cv_res.index(max(cv_res))]
print('The best m is: %i' % (best_m))
time_cv_ed = time.time()

features = []
features = ft.hog(X_train[0], orientations=best_m, feature_vector=True)
np.save('img.npy', X_train[0]) #write the RGB data of one image into npy file 
print('One of the feature vector is as follow:')
print(features)

for i in range(1, X_train.shape[0]):
    features = np.vstack((features, ft.hog(X_train[i], orientations=best_m, feature_vector=True)))
clf = sklearn.naive_bayes.GaussianNB()
clf.fit(features, y_train)
with open('clf.pickle','wb') as f: #write the trained modle into file
    pickle.dump(clf,f)
np.save('labels_test',y_test) #write the labels of test data set into npy file

features_test = ft.hog(X_test[0], orientations=best_m, feature_vector=True)
for i in range(1, X_test.shape[0]):
    features_test = np.vstack((features_test, ft.hog(X_test[i], orientations=best_m, feature_vector=True)))
np.save('features_test.npy', features_test) #write the features of test data set into npy file

lab = clf.predict(features_test) #classify the test data set
report = sklearn.metrics.classification_report(y_test, lab) #generate classification report
result = sklearn.metrics.confusion_matrix(y_test, lab) #generate confusion matrix
cm = pd.DataFrame(result)
time_end = time.time() #timing ends

print('Real label: ', y_test)
print('Predicted label: ', lab)
print('Classification report:')
print(report)
print('The confusion matrix is as follow:')
print(cm)
validation_time = time_cv_ed-time_cv_st
print('Time for cross-validation:', validation_time)
total_time = time_end - time_start
print('time for main: ', total_time - validation_time)
print('Total:', total_time)
plt.plot(m_list, cv_res)
plt.xlabel('Orientation N')
plt.ylabel('Accuracy')
plt.show()

