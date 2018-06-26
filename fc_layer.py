import numpy as np
from functools import reduce

class fc_layer(object):
  """
  Fully Connect layer
  """
  def __init__(self, input_shape, output_dim):
    """
    Initialize fc_layer

    """
  	# input param
    self.input_shape = input_shape
    self.N = input_shape[0]
    # cal channels
    self.C = reduce(lambda x, y: x * y, input_shape[1:])
    # output param
    self.d_out = np.zeros((self.N, output_dim)) #NC'
    self.output_shape = self.d_out.shape
    # define weights
    self.weights = np.random.standard_normal((self.C, output_dim)) / 100
    self.bias = np.random.standard_normal(output_dim) / 100
    # define gradient
    self.d_weights = np.zeros(self.weights.shape)
    self.d_bias = np.zeros(self.bias.shape)

  def forward(self, x):
    """
    Forward pass
    Input:
    - x: input batch
    Return: output batch
    """
    self.x = x.reshape([self.N, -1])
    return np.dot(self.x, self.weights)+self.bias

  def backprop(self, grad):
    """
    Backward pass
    Input:
    - grad, d_out: output gradients
    Helper:
    - col_x: x by column
    - d_out_i: d_out by column
    Return:
    - d_x: input gradients
    """
    self.d_out = grad
    # weight graidents
    for i in range(self.N):
      col_x = self.x[i][:, np.newaxis]
      d_out_i = self.d_out[i][:, np.newaxis].T
      self.d_weights += np.dot(col_x, d_out_i)
      self.d_bias += d_out_i.reshape(self.bias.shape)
    # input gradients
    d_x = np.dot(self.d_out, self.weights.T)
    d_x = np.reshape(d_x, self.input_shape)

    return d_x

  def update(self, alpha=0.00001, weight_decay=0.0004):
    # weight_decay = L2 regularization
    self.weights *= (1 - weight_decay)
    self.bias *= (1 - weight_decay)
    self.weights -= alpha * self.d_weights
    self.bias -= alpha * self.d_bias
    # clear gradients
    self.d_weights = np.zeros(self.weights.shape)
    self.d_bias = np.zeros(self.bias.shape)


if __name__ == "__main__":
  img = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
  fc = fc_layer(img.shape, 2)
  out = fc.forward(img)

  fc.backprop(np.array([[1, -2],[3,4]]))

  print(fc.d_weights)
  print(fc.d_bias)

  fc.update()
  print(fc.weights)