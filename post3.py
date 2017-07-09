import numpy as np
from random import shuffle


##################
# Loss functions #
##################


def svm_loss(W, X, y, reg):
  '''
  A vectorized implementation of SVM loss 

  Input has class 'C' and dimension 'D'. 

  Variable description:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N,C). This is the input with minibatches of data.
    - y: A numpy array of shape (N,). This has the class labels.
    - reg: Regularization strength.
  '''
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train=X.shape[0]
  scores=X.dot(W)
  correct_class_score=scores[np.arange(num_train),y]
  margin=np.maximum(0,scores-correct_class_score[:,np.newaxis]+1)
  margin[np.arange(num_train),y]=0
  loss=np.sum(margin)
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)

  return loss

def softmax_loss(x, y):
  '''
  A vectorized implementation of softmax_loss.
  '''
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N

  return loss


################
# Update Rules #
################


def sgd(w, dw, config=None):
  '''
  Implementation for Stochastic Gradient descent
  
  Variable descreption:
   - w: Weight matrix
   - dw: Weight Gradient
   - config: Dictionary for hyperparameters as learning rate, reg etc.
  '''
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2) # If congif not provided by user, provide a default value
  w -= config['learning_rate'] * dw

  return w, config

def sgd_momentum(w, dw, config=None):
  '''
  SGD with momentum
  '''
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)               
  v = config.get('velocity', np.zeros_like(w))    
  next_w = None
  v=config['momentum']*v-config['learning_rate']*dw
  w+=v
  config['velocity'] = v
  next_w=w

  return next_w, config

def nesterov(x,dx,congig=None):
  '''
  Implementation for Nesterov update rule
  '''
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)  
  next_x=None             
  v = config.get('velocity', np.zeros_like(w))  
  v_prev = v                           # initial v will be needed for updating x
  v = mu * v - learning_rate * dx      # velocity update - same as momentum
  x += -mu * v_prev + (1 + mu) * v   
  next_x=x

  return next_x,config  

def adagrad(x,dx,config=None):
  '''
  Adagrad update rule

  Variable description:
   - epsilon: Small scalar used for smoothing to avoid dividing by zero.
   - cache: For storing moving average of second moments of gradients.
  '''

  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', np.zeros_like(x))
  next_x=None
  cache += dx**2
  x += - learning_rate * dx / (np.sqrt(cache) + eps)
  next_x=x

  return next_x

  
def rmsprop(x, dx, config=None):
  '''
  RMSProp update rule uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.
  
  Variable description:
   - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
  '''
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('decay_rate', 0.99)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('cache', np.zeros_like(x))
  next_x = None
  config['cache']=config['decay_rate']*config['cache']+(1-config['decay_rate'])*dx**2
  x+=-config['learning_rate']*dx/(np.sqrt(config['cache'])+config['epsilon'])
  next_x=x

  return next_x, config

def adam(x, dx, config=None):
  '''
  Adam update rule incorporates moving averages of both the
  gradient and its square and a bias correction term.

  Variable description:
   -beta1: Decay rate for moving average of first moment of gradient.
   -beta2: Decay rate for moving average of second moment of gradient.
   -epsilon: Small scalar used for smoothing to avoid dividing by zero.
   -m: Moving average of gradient.
   -ma: Moving average of squared gradient.
   -t: Iteration number.
   '''
   if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('ma', np.zeros_like(x))
  config.setdefault('t', 0)
  next_x = None
  config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dx
  config['ma'] = config['beta2']*config['ma'] + (1-config['beta2'])*(dx**2)
  x += -config['learning_rate'] * config['m'] / (np.sqrt(config['ma']) + config['epsilon'])
  next_x=x

  return next_x, config

  ###############
  # Activations #
  ###############

def sigmoid_actv(x):
    '''
    Sigmoid Activation function
    '''
    sigmoid = 1 / (1 + np.exp(-x))

    return sigmoid

    # To use this, call sigmoid as a function

def tan_h(x):
  '''
  Tanh activation
  '''
  out = np.tanh(x)

  return out

def relu(x):
  '''
  Relu activation
  '''
  out = np.max(0,x) # For leaky relu, just replace 0 with desired lower bound

  return out

def maxout(w1,w2,b1,b2,x1,x2):
  '''
  Maxout activation
  '''
  out = np.max(np.dot(w1,x1)+b1,np.dot(w2,x2)+b2)

  # One must be aware of the fact that maxout increases the 
  # amount of computations.

  return out


