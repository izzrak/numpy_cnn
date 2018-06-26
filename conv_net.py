import numpy as np
from conv_layer import conv_layer
from pooling_layer import max_pooling
from activation_layer import relu
from fc_layer import fc_layer
from loss_layer import loss_layer


class conv_net(object):
  """
  Convolution net class
  methods:
  - forward: forward pass
  - backprop: back propagation
  - update: update weight 
  """
  def __init__(self, input_shape):
    # define batch param
    self.batch_loss = 0
    self.batch_acc  = 0
    self.batch_size = input_shape[0]
    
    # define network
    print('network structure:')
    self.conv1 = conv_layer (input_shape, 8, 5, 1, 2) #[batch_size, 28, 28, 1]
    print('conv1', self.conv1.output_shape)
    self.relu1 = relu(self.conv1.output_shape)
    print('relu1',self.relu1.output_shape)
    self.pool1 = max_pooling(self.relu1.output_shape)
    print('pool1', self.pool1.output_shape)
    self.conv2 = conv_layer (self.pool1.output_shape, 16, 3, 1, 1)
    print('conv2', self.conv2.output_shape)
    self.relu2 = relu(self.conv2.output_shape)
    print('relu2', self.relu2.output_shape)
    self.pool2 = max_pooling(self.relu2.output_shape)
    print('pool2', self.pool2.output_shape)
    self.fc = fc_layer(self.pool2.output_shape, 10)
    print('fc', self.fc.output_shape)
    self.loss = loss_layer(self.fc.output_shape)

  def forward(self, x, labels=None):
  	# clear batch loss
    self.batch_loss = 0
    self.batch_acc = 0

    # forward pass
    conv1_out = self.conv1.forward(x)
    relu1_out = self.relu1.forward(conv1_out)
    pool1_out = self.pool1.forward(relu1_out)
    conv2_out = self.conv2.forward(pool1_out)
    relu2_out = self.relu2.forward(conv2_out)
    pool2_out = self.pool2.forward(relu2_out)
    fc_out = self.fc.forward(pool2_out)

    # compute loss
    if type(labels) == np.ndarray:
      self.batch_loss += self.loss.loss_foward(fc_out, np.array(labels))
      for j in range(self.batch_size):
        if np.argmax(self.loss.sf[j]) == labels[j]:
          self.batch_acc += 1
    else:
      # compute softmax only
      self.loss.sf = self.loss.softmax(fc_out)


  def backprop(self):
    # back propagation
    self.conv1.backprop(self.relu1.backprop(
                        self.pool1.backprop(
                        self.conv2.backprop(
                        self.relu2.backprop(
                        self.pool2.backprop(
                        self.fc.backprop(
                        self.loss.backprop())))))))

  def update(self, lr=1e-4, weight_decay=0.0004):
    # adam optimizer
    self.fc.update(lr, weight_decay)
    self.conv2.update(lr, weight_decay)
    self.conv1.update(lr, weight_decay)

if __name__ == "__main__":
  net_test = conv_net([2,32,32,4])
  # img = np.random.standard_normal((2, 32, 32, 3))
  imgs = np.ones((2, 32, 32, 4))
  imgs *= 2
  labels = np.array([1,0]).reshape((2,1))
  net_test.forward(imgs,labels)
  net_test.backprop()

