import numpy as np

def unfold(x,k_size,stride,pad):
  """
  expand input data for fast convolution 
  Input:
  - x: input data, 1HWC format
  Return:
  - expanded input
  """
  # unfold_height = input_dim * k_size * k_size
  # unfold_width = output_height * output_width
  ufinput = []
  # set padding array
  padding_array = ((0,0),(pad,pad),(pad,pad),(0,0)) #NHWC
  padded = np.pad(x,padding_array,'constant',constant_values=0)
  for i in range(0, padded.shape[1] - k_size + 1, stride):
    for j in range(0, padded.shape[2] - k_size + 1, stride):
      col = padded[:, i:i + k_size, j:j + k_size, :].reshape([-1]) #KKC
      ufinput.append(col)
  return np.array(ufinput)