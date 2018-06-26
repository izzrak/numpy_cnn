import numpy as np
import struct
from glob import glob


def load_mnist(path, kind='train'):
  """Load MNIST data from `path`"""
  images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
  labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]

  with open(labels_path, 'rb') as lbpath:
    magic, n = struct.unpack('>II',
                 lbpath.read(8))
    labels = np.fromfile(lbpath,
               dtype=np.uint8)

  with open(images_path, 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack('>IIII',
                         imgpath.read(16))
    images = np.fromfile(imgpath,
               dtype=np.uint8).reshape(len(labels), 784)

  return images, labels

# implemented in solver
def load_param(path, model):
  param_dict = np.load(path, encoding='bytes').item()
  '''
  for layer in param_dict:
    print(layer)
    for data_type in param_dict[layer]:
      tmp = param_dict[layer]
      #print(tmp)
      #print(data_type)
      for data in tmp[data_type]:
        print(data_type)
        print(data)

  '''
