from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N=X.shape[0]
    D=X.shape[1]
    C=W.shape[1]
    for i in range(N):
        f_i = X[i].dot(W)
        f_i -= np.max(f_i)
        sum_j = np.sum(np.exp(f_i))
        loss += -np.log(np.exp(f_i[y[i]])/sum_j)
        for k in range(C):
            dW[:, k] += (np.exp(f_i[k])/sum_j - (k == y[i])) * X[i]
    loss /= N
    loss += 0.5 * reg * np.sum(W*W)
    dW /= N
    dW += reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N=X.shape[0]
    D=X.shape[1]
    C=W.shape[1]
    f = X.dot(W)
    f -= np.max(f,axis=1, keepdims=True)
    sum = np.sum(np.exp(f), axis=1, keepdims=True)
    y_correct = np.zeros((N,C))
    y_correct[np.arange(N),y]=1

    loss += -np.sum(np.multiply(y_correct, np.log(np.exp(f)/sum))) / N
    loss += 0.5 * reg * np.sum(W*W)

    dW += X.T.dot(np.exp(f)/sum - y_correct)
    dW /= N
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
