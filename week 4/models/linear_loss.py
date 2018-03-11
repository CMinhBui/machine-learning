import numpy as np

def linear_loss_naive(W, X, y, reg):
    """
    Linear loss function, naive implementation (with loops)

    Inputs have dimension D, there are N examples.

    Inputs:
    - W: A numpy array of shape (D, 1) containing weights.
    - X: A numpy array of shape (N, D) containing data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c is a real number.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the linear loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    #compute loss 
    for i, y_i in enumerate(y):
        _y = 0.0
        for j, w in enumerate(W):
            _y += (w*X[i][j])
        loss += 1/(2*len(y))*(y_i - _y)**2
    #adding regularization term
    for w in W: 
        loss += reg*w*w
    
    #compute gradient
    gradient_dimension = len(W)
    for j in range(gradient_dimension):
        dW[j] = reg*2*W[j]
        for i, y_i in enumerate(y):
            dW[j] -= 1/len(y) * X[i][j] * y[i]
            for k, w_k in enumerate(W):
                dW[j] += 1/len(y) * X[i][j] * X[i][k] * w_k
            
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def linear_loss_vectorized(W, X, y, reg):
    """
    Linear loss function, vectorized version.

    Inputs and outputs are the same as linear_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the linear loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    xw_y = np.matmul(X, W) - y
    loss = 1/(2*len(y)) * np.matmul(xw_y.transpose(), xw_y) + reg*np.matmul(W.transpose(), W)
    dW = (1/len(y)) * np.matmul(np.matmul(X.transpose(), X), W) - (1/len(y)) * np.matmul(X.transpose(), y) + 2*reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW