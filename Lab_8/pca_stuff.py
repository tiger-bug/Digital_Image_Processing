
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



data=fetch_lfw_people(min_faces_per_person=100)
#data = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
random_state = 0
X = data.data
y = data.target
np.max(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=random_state)
pca_var = PCA(svd_solver='randomized',random_state=0)
pca_var.fit(X)
var = pca_var.explained_variance_ratio_
#var[150]

component_var = var[np.where(var>0.001)]
n_components = component_var.shape[0]
#
#n_components = 3



pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
X_train_pca,X_test_pca = pca.transform(X_train),pca.transform(X_test)


param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf = clf.fit(X_train_pca, y_train)
#print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
y_pred = clf.predict(X_test_pca)
acc = accuracy_score(y_test,y_pred)

#y_pred = clf.predict(X_test_pca)
def pca_component_size(X,var_thresh = 0.001):
    '''In order to speed up the processing of the SVM algorithm, we return the pca with the appropriate number of
    components for a given variance threshold'''
    global random_state
    pca_var = PCA(svd_solver='randomized',random_state=random_state)
    pca_var.fit(X)
    var = pca_var.explained_variance_ratio_
    component_var = var[np.where(var>var_thresh)]
    n_components = component_var.shape[0]
    print(n_components)
    return PCA(n_components=n_components,svd_solver='full',random_state=random_state)

pca = pca_component_size(X)
pca.fit(X_train)


X_train_pca,X_test_pca = pca.transform(X_train),pca.transform(X_test)
print(confusion_matrix(y_test,y_pred))

#
#parameters = {'kernel':('linear', 'rbf'), 'C':[1., 10.]}
#svc = SVC(gamma=0.1)
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf = clf.fit(X_train_pca, y_train)
#print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

y_pred = clf.predict(X_test_pca)
#print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))










import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



#data=fetch_lfw_people(min_faces_per_person=50)
data = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
random_state = 0
X = data.data
y = data.target
#np.max(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state=random_state)
pca_var = PCA(svd_solver='randomized',random_state=0)
pca_var.fit(X)
var = pca_var.explained_variance_ratio_
#var[150]

component_var = var[np.where(var>0.0001)]
#n_components = component_var.shape[0]
#
n_components = 150



pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
X_train_pca = pca.fit(X_train)
X_test_pca = pca.fit(X_test)
X_train_pca,X_test_pca = pca.transform(X_train),pca.transform(X_test)


param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf = clf.fit(X_train, y_train)
#print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
y_pred = clf.predict(X_test)
target_names = data.target_names
n_classes = target_names.shape[0]
#print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred))
score = accuracy_score(y_test, y_pred)















