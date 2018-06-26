import numpy as np
from conv_net import conv_net

from data_utils import *
import time

class Solver(object):
  """
  Solver class for train and test
  methods:
  - train: train conv_net
  - test: test trained net
  """
  def __init__(self):
    # config
    self.lr = 1e-4
    self.weight_decay=0.0004
    self.batch_size = 50
    self.model = conv_net([self.batch_size, 28, 28, 1]) #MNIST input
    self.max_epoch = 1
    self.param_path = 'param.npy'

  def train(self):
    images, labels = load_mnist('./dataset', 'train')
    print(images.shape, labels.shape)

    # start trainig
    for epoch in range(self.max_epoch):
      # clear loss
      self.train_acc = 0
      self.train_loss = 0

      batch_num = int(images.shape[0] / self.batch_size)
      for i in range(batch_num):
        # load batch
        imgs = images[i * self.batch_size:(i + 1) * self.batch_size].reshape([self.batch_size, 28, 28, 1])
        lbls = labels[i * self.batch_size:(i + 1) * self.batch_size]

        # compute one batch
        self.model.forward(imgs,lbls)
        self.model.backprop()
        self.model.update(self.lr, self.weight_decay)
        # compute loss
        self.train_acc += self.model.batch_acc
        self.train_loss += self.model.batch_loss

        if i % 10 == 0:
          # compute average
          avg_batch_acc = float(self.model.batch_acc / self.batch_size)
          avg_batch_loss = self.model.batch_loss / self.batch_size
          print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + 
                              " epoch: %d, batch: %d, batch_acc: %.4f, avg_batch_loss: %.4f " % (
                              epoch, i, avg_batch_acc, avg_batch_loss))
      # compute average
      avg_train_acc = float(self.train_acc / images.shape[0])
      avg_train_loss = self.train_loss /images.shape[0]
      print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + 
                          "  epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (
                           epoch, avg_train_acc, avg_train_loss))
    
    # save weights
    conv1_param = {'weights': self.model.conv1.weights,
                   'bias': self.model.conv1.bias,}
    conv2_param = {'weights': self.model.conv2.weights,
                   'bias': self.model.conv2.bias,}
    fc_param = {'weights': self.model.fc.weights,
                   'bias': self.model.fc.bias,}
    param_dict = {'conv1': conv1_param,
                  'conv2': conv2_param,
                  'fc': fc_param,}
    np.save("param.npy",param_dict)

  def test(self, path):
    images, labels = load_mnist('./dataset', 't10k')
    print(images.shape, labels.shape)

    # clear loss
    self.test_acc = 0
    self.test_loss = 0
    
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

 
    batch_num = int(images.shape[0] / self.batch_size)
    for i in range(batch_num):
      print('testing batch number %d' %(i))
      # load batch
      imgs = images[i * self.batch_size:(i + 1) * self.batch_size].reshape([self.batch_size, 28, 28, 1])
      lbls = labels[i * self.batch_size:(i + 1) * self.batch_size]

      # forward pass only
      self.model.forward(imgs,lbls)

      # compute loss
      self.test_acc += self.model.batch_acc
      self.test_loss += self.model.batch_loss

    # compute average
    avg_test_acc = float(self.test_acc / images.shape[0])
    avg_test_loss =  self.test_loss /images.shape[0]
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + 
                          " train_acc: %.4f  avg_train_loss: %.4f" % (avg_test_acc, avg_test_loss))


if __name__ == "__main__":
  solver = Solver()
  #solver.train() # uncomment to train
  solver.test('param.npy')