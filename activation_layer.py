import numpy as np

class relu(object):
  """
  ReLU layer
  """
  def __init__(self, input_shape):
    """
  	Initialize relu layer

    """
    self.d_x = np.zeros(input_shape)
    self.output_shape = input_shape

  def forward(self, x):
    """
    Forward pass
    Input:
    - x: input batch
    Return: output batch
    """
    self.x = x
    return np.maximum(self.x, 0) # x * (x > 0)

  def backprop(self, grad):
    """
    Backward pass
    Input:
    - grad: output gradients
    Return:
    - d_x: input gradients
    """
    self.d_x = grad
    self.d_x[self.x<0]=0 #d_x * (x > 0)
    return self.d_x