
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
#from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV



def svc_analysis(data):
    '''
    Function to conduct a SVM classification.  Inputs are downloaded datasets.
    The testing size range from 0.1*data_size to 0.9*data_size (incrementing by 0.1).  It prints out a confusion matrix as well as accuracy scores.  The next step is to plot the accuracy score based on the testing size and accuracy scores to determine the best training and testing split for the data.  The Principal Component analysis was done to speed up the process.  The GridSearchCV was used to find the best parameters per testing size for the model.  
    
    
The code for the principal component analysis was borrowed from 

https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py.
    '''
    
    X = data.data
    y = data.target
    print('There are {0} faces in dataset'.format(X.shape[0]))
    print('There are {0} number of features (aka number of pixels.)'.format(X.shape[1]))
    print('There are {0} people in the image (aka number of classes).'.format(max(y)+1))
    test_vals = np.arange(0.1,1.0,0.1)
    acc = np.zeros(len(test_vals))
    for i in range(len(test_vals)):
        print('Test size: {0}'.format(test_vals[i]))
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_vals[i],random_state = 0)
        
        pca_var = PCA(svd_solver='randomized',random_state=0)
        pca_var.fit(X)
        var = pca_var.explained_variance_ratio_

        component_var = var[np.where(var>0.001)]
        
        n_components = component_var.shape[0]
        
        pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
        X_train_pca,X_test_pca = pca.transform(X_train),pca.transform(X_test)

        
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
        clf = clf.fit(X_train_pca, y_train)
#print("done in %0.3fs" % (time() - t0))
        print("Best estimator found by grid search:")
#        print(clf.best_estimator_)
        y_pred = clf.predict(X_test_pca)
        acc[i] = accuracy_score(y_test,y_pred)
        print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred))

    return test_vals,acc

if __name__=='__main__':
    path = r'D:\Digital_Image_Processing\Lab_8'
    os.chdir(path)
#    data = fetch_lfw_people()
    data = data=fetch_lfw_people(min_faces_per_person=100)

#    clf = SVC(gamma=0.0005,C = 1.)
    img = img= data.images
    print('First 3 images...')
    fig,ax = plt.subplots(ncols=len(range(3)), figsize=(10, 5))
    for i in range(3):
        ax[i].imshow(img[i],interpolation='nearest', cmap=plt.cm.gray)
        ax[i].axis('off')
        ax[i].set_title('Image '+str(i+1))   
    plt.show()
    test_vals,acc = svc_analysis(data)
    plt.scatter(100*test_vals,acc,marker='o')
    plt.xlabel('Percent of data used for testing')
    plt.ylabel('Accuracy Score')
    plt.title('Test Size/Accuracy Score')
    plt.show()
    m, = np.where(acc == np.max(acc))
    print('Testing size, highest accuracy: {}%, {}'.format(100*test_vals[m][0],acc[m][0]))
        







