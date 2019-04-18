

from sklearn.datasets import fetch_olivetti_faces
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
#import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def mlp_fun(X,y,hidden_layer,activation,learning_rate,test_size = 0.3,random_state = 100):
    '''Main function that runs mlp algorithm:
    
    ########################parameters########################
        
        hidden layer sizes: list of tuples of nodes,layers.
        activation funcion: list of activation functions to run.
        learning rate: list of learning rates. 
        
    
    ########################returns###########################
    
        best classifier with parameters
        predictecd values
        tested values
        
    '''

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_size,random_state=random_state)
    

    param_grid = {
            'hidden_layer_sizes': hidden_layer,
            'activation': activation,
            'learning_rate_init': learning_rate
        } 

    clf = GridSearchCV(MLPClassifier(solver='sgd',verbose=False,random_state=random_state),param_grid=param_grid,n_jobs=-1)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return clf,y_pred,y_test


if __name__ == '__main__':
    faces = fetch_olivetti_faces()
    X = faces.data
    y = faces.target
    print('There are {} digits in this dataset.'.format(str(X.shape[0])))
    print('There are {} pixels per image.'.format(str(X.shape[1])))
    print('There are {} classes.'.format(str(max(y+1))))

    hidden_layer = [(200 ,6), (300,100), (400, 200),(500,500)]
    activation = ['relu','logistic','tanh']
    learning_rate = [5e-1, 5e-2, 5e-3]
    
    clf,y_pred,y_test = mlp_fun(X,y,hidden_layer,activation,learning_rate)
    
    print('Accuracy Score: {}'.format(accuracy_score(y_test,y_pred)))
    print('Best learning rate, hidden layer size, activation function: {},{},{}'.format(clf.best_estimator_.learning_rate_init,clf.best_estimator_.hidden_layer_sizes,clf.best_estimator_.activation))
    

    






