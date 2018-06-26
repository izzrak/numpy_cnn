import numpy as np
from unfold import unfold
from optimizer import adam

class conv_layer(object):
  """
  Convolution layer 
  Assumptions:
  - convs are done with square input data and kernels
    H = W,
    k_h = k_w = k_size, 
    s_h = s_w = stride, 
    pad_h = pad_w = pad
  """
  def __init__(self,input_shape,output_dim,k_size,stride,pad):
    """
    Initialize conv_layer
    - data format: NHWC
    - weights format: KKCC'
    - d_out: placeholder for output gradient
    - d_weights: placeholder for weights gradient
    - d_bias: placeholder for bias gradient
    """
    # input param
    self.input_shape = input_shape
    self.N, self.H, self.W, self.C = input_shape
    # conv param
    self.K = k_size   
    self.S = stride
    self.P = pad 
    # output param
    self.out_N = output_dim
    self.out_H = int((self.H + 2*self.P - self.K)/self.S + 1)
    self.out_W = int((self.W + 2*self.P - self.K)/self.S + 1)
    # set output placeholder
    self.d_out = np.zeros((self.N,self.out_H,self.out_W,self.out_N)) #NHWC
    self.output_shape = self.d_out.shape
    # define weights and gradients
    self.weights = np.random.standard_normal((self.K,self.K,self.C,self.out_N)) / 1000 #KKCC'
    self.bias = np.random.standard_normal(self.out_N) / 1000
    self.d_weights = np.zeros(self.weights.shape)
    self.d_bias = np.zeros(self.bias.shape)
    self.config_w = None
    self.config_b = None

  def forward(self, x):
    """
    Forward pass
    Input:
    - x: input batch
    Helper:
    - exp_weights: expanded weights for fast convolution
    - exp_x: expanded input for fast convolution
    Return:
    - out: convolution output batch
    """
    out = np.zeros(self.output_shape)
    # expand weights 
    exp_weights = self.weights.reshape([-1, self.out_N]) #KKC*C'
    self.exp_x = []

    # conv forward
    for i in range(self.N):
      x_i = x[i][np.newaxis,:] #single image
      # unfold input image
      exp_x_i = unfold(x_i,self.K,self.S,self.P) #N*KKC
      # store expanded images
      self.exp_x.append(exp_x_i)
      # perform fast convolution
      out[i] = np.reshape(np.dot(exp_x_i,exp_weights) + self.bias, self.output_shape[1:]) #H'W'C'
    self.exp_x = np.array(self.exp_x)
    return out

  def backprop(self, grad):
    """
    Backward pass
    Input:
    - grad, d_out: output gradients
    Helper:
    - flipped_weights: flipped weight for inverse conv
    - exp_flipped_weights: expanded weights for fast convolution
    - exp_d_out: expanded d_out for fast convolution
    Return:
    - d_x: input gradients
    """
    self.d_out = grad
    col_d_out = np.reshape(self.d_out, [self.N, -1, self.out_N])
    # weight graidents
    for i in range(self.N):
      self.d_weights += np.dot(self.exp_x[i].T, col_d_out[i]).reshape(self.weights.shape)
      self.d_bias += np.sum(col_d_out, axis=(0, 1))
    # inverse conv for d_x
    # flip weights
    flipped_weights = np.flipud(np.fliplr(self.weights))
    # swap input and output dimension
    flipped_weights = flipped_weights.swapaxes(2, 3)

    # conv backward
    d_x = np.zeros(self.input_shape)
    # expand flipped weights
    exp_flipped_weights = flipped_weights.reshape([-1, self.C])

    for i in range(self.N):
      d_out_i = self.d_out[i][np.newaxis,:]
      # unfold input image
      exp_d_out = unfold(d_out_i,self.K,self.S,self.P)
      d_x[i] = np.reshape(np.dot(exp_d_out,exp_flipped_weights),self.input_shape[1:]) #HWC
    return d_x

  def update(self, lr=0.00001, weight_decay=0.0004):
    '''
    # mini-batch SGD
    self.weights *= (1 - weight_decay)
    self.bias *= (1 - weight_decay)
    self.weights -= lr * self.d_weights
    self.bias -= lr * self.d_bias
    '''
    # adam optimizer
    self.weights, self.config_w = adam(self.weights * (1 - weight_decay),
                                       self.d_weights, 
                                       config=self.config_w)
    self.bias, self.config_b = adam(self.bias * (1 - weight_decay),
                                    self.d_bias, 
                                    config=self.config_b)
    # clear gradients
    self.d_weights = np.zeros(self.weights.shape)
    self.d_bias = np.zeros(self.bias.shape)
    

if __name__ == "__main__":
  # img = np.random.standard_normal((2, 32, 32, 3))
  img = np.ones((2, 32, 32, 4))
  img *= 2
  conv = conv_layer(img.shape, 12, 3, 1, 1)
  next = conv.forward(img)
  next1 = next.copy() + 1
  conv.backprop(next1-next)
  print(conv.d_weights.shape)
  print(conv.d_bias.shape)
  conv.update()
  print(conv.weights.shape)
  print(conv.bias)


