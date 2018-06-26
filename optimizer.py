import numpy as np


# adam optimizer adapted from CS231n assi
def adam(x, dx, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.
  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-4)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)
  
  next_x = None
  #############################################################################
  # TODO: Implement the Adam update formula, storing the next value of x in   #
  # the next_x variable. Don't forget to update the m, v, and t variables     #
  # stored in config.                                                         #
  #############################################################################
  lr = config['learning_rate']
  beta1 = config['beta1']
  beta2 = config['beta2']
  eps = config['epsilon']
  m = config['m']
  v = config['v']
  t = config['t']
    
  t += 1  
    
  m = beta1*m + (1-beta1)*dx
  v = beta2*v + (1-beta2)*(dx**2)
  mb = m/(1-beta1**t)
  vb = v/(1-beta2**t)
    
  next_x = x - lr * mb / (np.sqrt(vb) + eps)

  config['m'] = m
  config['v'] = v
  config['t'] = t
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return next_x, config