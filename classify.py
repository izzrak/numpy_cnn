import numpy as np
import cv2
from conv_net import conv_net
import os 
 
class classify_api(object):
  """
  Classification API for MNIST images
  """
  def __init__(self, path):
    self.model = conv_net([1, 28, 28, 1])
    # load_param
    param_dict = np.load(path, encoding='bytes').item()
    conv1_param = param_dict['conv1']
    conv2_param = param_dict['conv2']
    fc_param = param_dict['fc']

    # fill weights
    self.model.conv1.weights = conv1_param['weights']
    self.model.conv1.bias = conv1_param['bias']
    self.model.conv2.weights = conv2_param['weights']
    self.model.conv2.bias = conv2_param['bias']
    self.model.fc.weights = fc_param['weights']
    self.model.fc.bias = fc_param['bias']

  def classify_by_dir(self, img_dir):
    print('wait to be done')

  def classify_by_image(self, img_path):
    """
    Input: 
    img_path: image path
    """
    img = cv2.imread(img_path,0) #grayscale
    img_np = np.array(img).reshape([1,28,28,1])
    print(img_np.shape)
    self.model.forward(img_np, None)
    print('image %s is number %d' %(img_path, np.argmax(self.model.loss.sf)))

if __name__ == "__main__":
 api = classify_api('param.npy')
 api.classify_by_image('72.png')