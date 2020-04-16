import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        y_test = (np.where(y==0,-1,1))
        for i in range(0, max_iterations):
            indices = np.where(y_test*(X.dot(w)+b)<=0)
            w += step_size* y_test[indices].dot(X[indices]) /N
            b += step_size* np.sum(y_test[indices]) /N
        ############################################
    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        for i in range(0, max_iterations):
            error = y-sigmoid(X.dot(w) + b)
            w += step_size * error.dot(X) / N
            b += step_size * np.sum(error) / N

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def step_function(x):
    ans = []
    for i in range(0,len(x)):
        if x[i]>0:
            ans.append(1)
        else:
            ans.append(0)
    return np.array(ans)
def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    z = 1/(1+np.exp(-z))
    value = z
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        preds = step_function(X.dot(w) + b)
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)

        for i in range(N):
            if sigmoid(X[i].dot(w) + b)>0.5:
                preds[i]=1
            else:
                preds[i]=0

        
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """
    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        def softmax(x):
            e = np.exp(x - np.max(x))
            if e.ndim == 1:
                return e / np.sum(e, axis=0)
            else:  
                return e / np.array([np.sum(e, axis=1)]).T
    
        for i in range(max_iterations):
            indices = np.random.choice(N)
            Xr=X[indices]
            yr = y[indices]
            s = softmax((w.dot(Xr.T)).T + b)
            s[yr]-=1
            w_gradient = s.reshape((len(s),1)).dot(Xr.reshape((len(Xr),1)).T)
            b_gradient = s
            w -= step_size * w_gradient
            b -= step_size * b_gradient
        
        
        
        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #

        w = np.zeros((C, D))
        b = np.zeros(C)
        def softmax(x):
            e = np.exp(x - np.max(x))
            if e.ndim == 1:
                return e / np.sum(e, axis=0)
            else:  
                return e / np.array([np.sum(e, axis=1)]).T
    
        y = np.eye(C)[y]
        for i in range(max_iterations):
            error = softmax((w.dot(X.T)).T + b) - y
            w_gradient = error.T.dot(X) / N
            b_gradient = np.sum(error, axis=0) / N
            w -= step_size * w_gradient
            b -= step_size * b_gradient
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    def softmax(x):
            e = np.exp(x - np.max(x))
            if e.ndim == 1:
                return e / np.sum(e, axis=0)
            else:  
                return e / np.array([np.sum(e, axis=1)]).T

    v = softmax((w.dot(X.T)).T + b).tolist()
    for i in range(0,len(v)):
        preds[i]=v[i].index(max(v[i]))

    ############################################

    assert preds.shape == (N,)
    return preds