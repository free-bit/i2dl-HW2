import numpy as np
import re

from exercise_code.layers import *
from exercise_code.layer_utils import *

numeric_suffix = lambda text: re.search(r"\d+", text).group(0)

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    
    The architecure should be affine - relu - affine - softmax.
  
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
  
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0, loss_function='softmax'):
        """
        Initialize a new network.
    
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        if loss_function != 'softmax':
            raise Exception('Wrong loss function')
        else:
            self.loss_function = 'softmax'

        ############################################################################
        # DONE: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim)) 
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b1'] = np.zeros((hidden_dim,))
        self.params['b2'] = np.zeros((num_classes,))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
    
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
    
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
    
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # DONE: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        fc1, cache_fwd1 = affine_forward(X, W1, b1)
        active, cache_relu = relu_forward(fc1)
        scores, cache_fwd2 = affine_forward(active, W2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # DONE: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        loss, dscores = softmax_loss(scores, y)
        loss += (np.sum(W1**2) + np.sum(W2**2)) * (reg/2)   # Regularization term is added to loss, lambda is halved based on description in the task

        dh, dw2, db2 = affine_backward(dscores, cache_fwd2) # score = w2*h+b
        dw2 += reg * W2                                     # Regularization term is added to grad, lambda is halved based on description in the task

        do = relu_backward(dh, cache_relu)                  # h = ReLU(o)
        
        dx, dw1, db1 = affine_backward(do, cache_fwd1)      # o = w1*x+b
        dw1 += reg * W1                                     # Regularization term is added to grad, lambda is halved based on description in the task

        grads = {"W1": dw1, "W2": dw2, "b1": db1, "b2": db2}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be
    
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - loss function
    
    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None,
                 loss_function='softmax'):
        """
        Initialize a new FullyConnectedNet.
        
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.loss_function = loss_function
        if loss_function == 'softmax':
            self.chosen_loss_function = softmax_loss
        elif loss_function == 'l2':
            self.chosen_loss_function = l2_loss
        else:
            raise Exception('Wrong loss function')

        ############################################################################
        # DONE: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        num_hidden = len(hidden_dims) # Number of hidden layers
        num_layers = num_hidden + 1   # Number of all layers

        # Generate keys for dictionary of parameters for all layers
        weight_keys = ["W"+str(i) for i in range(1, num_layers+1)] 
        bias_keys = ["b"+str(i) for i in range(1, num_layers+1)]

        # Initialize layer parameters
        # Keep dimensions such that input and output dimensions are also included
        dims = [input_dim, *hidden_dims, num_classes]
        # Iterate over dims, two elements at a time
        for i in range(num_layers):
          self.params[weight_keys[i]] = np.random.normal(scale=weight_scale, size=(dims[i], dims[i+1])) 
          self.params[bias_keys[i]] = np.zeros((dims[i+1],))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
    
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # DONE: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        all_keys = list(self.params.keys())
        all_keys.sort(key=numeric_suffix) # Keys are probably already sorted, but doesn't have to be.
        weight_keys = list(filter(lambda p: p.startswith("W"), all_keys))
        bias_keys = list(filter(lambda p: p.startswith("b"), all_keys))
        
        num_hidden = len(weight_keys) - 1
        fwd_caches = []
        relu_caches = []
        Xi = X
        for i in range(num_hidden):
          weight_key, bias_key = weight_keys[i], bias_keys[i]
          Wi, bi = self.params[weight_key], self.params[bias_key] # Extract W & b for the ith iteration
          fc, cache_fwd = affine_forward(Xi, Wi, bi)
          active, cache_relu = relu_forward(fc)

          Xi = active                                             # Input to next iteration
          fwd_caches.append(cache_fwd)
          relu_caches.append(cache_relu)

        weight_key, bias_key = weight_keys[-1], bias_keys[-1]
        W, b = self.params[weight_key], self.params[bias_key]
        scores, cache_fwd = affine_forward(Xi, W, b)
        fwd_caches.append(cache_fwd)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        # We compute loss and gradient of last layer
        # For another notebook, we will need to switch between different loss functions
        # By default we choose the softmax loss
        loss, dscores = self.chosen_loss_function(scores, y)
        #######################################################################
        # DONE: Implement the backward pass for the fully-connected net. Store#
        # the loss in the loss variable and gradients in the grads dictionary.#
        #                                                                     #
        #1_FullyConnectedNets.ipynb                                           #
        # Compute                                                             #
        # data loss using softmax, and make sure that grads[k] holds the      #
        # gradients for self.params[k]. Don't forget to add L2 regularization!#
        #                                                                     #
        # When using batch normalization, you don't need to regularize the    #
        # scale and shift parameters.                                         #
        #                                                                     #
        # NOTE: To ensure that your implementation matches ours and you pass  #
        # the automated tests, make sure that your L2 regularization includes #
        # a factor of 0.5 to simplify the expression for the gradient.        #
        #                                                                     #
        #######################################################################
        reg_coeff = self.reg
        reg_sum = 0
        weight_key, bias_key = weight_keys[-1], bias_keys[-1]

        # Calculate backward pass for the last affine layer
        cache_fwd = fwd_caches.pop()
        dxi, dwi, dbi = affine_backward(dscores, cache_fwd)

        # Store grads for the last layer
        wi = self.params[weight_key]
        dwi += reg_coeff * wi
        reg_sum += np.sum(wi**2)
        grads.update({weight_key: dwi, bias_key: dbi})
  
        # Calculate backward pass for each hidden layer
        for i in range(num_hidden-1, -1, -1):
          weight_key, bias_key = weight_keys[i], bias_keys[i]

          # Calculate backward pass for ReLU
          cache_relu = relu_caches.pop()
          do = relu_backward(dxi, cache_relu)

          # Calculate backward pass for affine
          cache_fwd = fwd_caches.pop()
          dxi, dwi, dbi = affine_backward(do, cache_fwd)
          wi = self.params[weight_key]
          dwi += reg_coeff* wi
          reg_sum += np.sum(wi**2)

          # Store grads for ith layer
          grads.update({weight_key: dwi, bias_key: dbi})

        # Regularization term is added to loss, lambda is halved based on description in the task
        loss += reg_sum * reg_coeff / 2
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################


        return loss, grads
