

#from sklearn import datasets
from sklearn.datasets import fetch_olivetti_faces
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
#
faces = fetch_olivetti_faces()
##faces.target.shape
#plt.imshow(faces.images[44],cmap = 'gray')
#plt.axis('off')
#digits = datasets.load_digits()

X = faces.data
y = faces.target
test_size = 0.3
random_state = 100

print('There are {} digits in this dataset.'.format(str(X.shape[0])))

print('There are {} pixels per image.'.format(str(X.shape[1])))

print('There are {} classes.'.format(str(max(y+1))))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_size,random_state=random_state)


param_grid = {
        'hidden_layer_sizes': [(200 ,6), (300,6), (400, 6),(500,6)],
        'activation': ['relu','logistic','tanh'],
        'learning_rate_init': [5e-1, 5e-2, 5e-3]
    }

clf = GridSearchCV(MLPClassifier(solver='sgd',verbose=True,random_state=random_state),param_grid=param_grid,n_jobs=-1)
clf = clf.fit(X_train,y_train)
#clf = GridSearchCV(MLPClassifier(solver='sgd',verbose=True,random_state=random_state,param_grid),n_jobs=-1)
y_pred = clf.predict(X_test)
#clf.fit(X_train, y_train)
#mlp.fit(X_train,y_train)
#y_pred = mlp.predict(X_test)
print('Best learning rate, hidden layer size, activation function: {},{},{}'.format(clf.best_estimator_.learning_rate_init,clf.best_estimator_.hidden_layer_sizes,clf.best_estimator_.activation))
print('Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
print(confusion_matrix(y_test,y_pred))
#print(confusion_matrix(y_test,y_pred))
# clf.best_estimator_.early_stopping



def mlp_fun():
    '''Main function that runs mlp algorithm:
    
    ########################parameters########################
        
        hidden layer sizes: list of tuples of nodes,layers.
        activation funcion: list of activation functions to run.
        learning rate: list of learning rates.  
    
    ########################returns###########################
    
        best classifier with parameters
        accuracy score








