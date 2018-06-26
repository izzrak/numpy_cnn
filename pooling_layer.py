import numpy as np
import cv2
import matplotlib.pyplot as plt
from unfold import unfold

class max_pooling(object):
  """
  Max Pooling layer 
  Assumptions:
  - kernel_size = stride = 2 by default
  - input_height % stride = 0
  - input_width % stride = 0
  """
  def __init__(self, input_shape, k_size=2, stride=2):
    """
    Initialize max_pooling
    - data format: NHWC
    - weights format: KKCC'
    - d_out: placeholder for output gradient
    - d_weights: placeholder for weights gradient
    - d_bias: placeholder for bias gradient
    """
    self.k_size = k_size
    self.stride = stride
    self.input_shape = input_shape
    self.N, self.H, self.W, self.C = input_shape
    self.indices = np.zeros(input_shape)
    self.output_shape = [input_shape[0], 
                         int(input_shape[1]/self.stride), 
                         int(input_shape[2]/self.stride), 
                         self.C] #NH'W'C'

  def forward(self, x):
    """
    Forward pass
    Input:
    - x: input batch
    Helper:
    - ufimg: expanded image for fast pooling
    - max_col: positions of max values
    Return:
    - out: output batch
    """
    out = np.zeros(self.output_shape)
    # perform max pooling by batch and channel
    for i in range(self.N):
      for c in range(self.C):
        frame = x[i,:,:,c].reshape(1,self.H,self.W,1)
        # expand image
        self.ufimg = unfold(frame,self.k_size,self.stride,0) 
        out[i,:,:,c] = np.amax(self.ufimg,axis=-1).reshape(self.output_shape[1],self.output_shape[2])
        # find max position for gradient calculation
        max_col = np.argmax(self.ufimg,axis=-1).reshape([-1,1]) # pick first max value
        # set 1 for max value
        for r in range(max_col.shape[0]):
          block_dim = self.output_shape[1]
          hh = 2 * int(r / block_dim) + int(max_col[r]/2)
          ww = 2 * int(r % block_dim) + int(max_col[r]%2)
          self.indices[i,hh,ww,c] = 1
    return out

  def backprop(self, grad):
    """
    Backward pass
    Input:
    - grad: gradients
    Return: input data with max value indices
    """
    return np.repeat(np.repeat(grad, self.stride, axis=1), self.stride, axis=2) * self.indices

if __name__ == "__main__":
  img = cv2.imread('72.png')
  img2 = np.array([img,img]).reshape([2, img.shape[0], img.shape[1], img.shape[2]])
  print(img2.shape)
  pool = max_pooling(img2.shape, 2, 2)
  img_out = pool.forward(img2)
  img_back = pool.backprop(img_out)
  print(img2[0,4:8,4:8,1])
  print(img_out[0,2:4,2:4,1])
  print(img_back[0,4:8,4:8,1])
  #plt.imshow(img2[0])
  #plt.imshow(img_out[0].astype(int))
  #plt.show()
