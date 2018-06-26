import numpy as np

class loss_layer(object):
  def __init__(self, input_shape):
    self.sf = np.zeros(input_shape)
    self.d_x = np.zeros(input_shape)
    self.N = input_shape[0]

  def loss_foward(self, fc_out, label):
    """
    Softmax loss
    Input:
    - fc_out: score from fc_layer
    - label: label vector
    Return:
    - loss: softmax loss
    """
    self.label = label
    self.score = fc_out
    self.softmax(self.score)
    self.loss = 0
    for i in range(self.N):
      self.loss += np.log(np.sum(np.exp(self.score[i]))) - self.score[i, label[i]]
    return self.loss

  def softmax(self, fc_out):
    """
    Softmax
    Input:
    - fc_out: score from fc_layer
    Return:
    - sf: softmax result
    """
    self.sf = fc_out
    #print(self.sf)
    for i in range(self.N):
      self.sf[i] -= np.max(self.sf[i])
      self.sf[i] =  np.exp(self.sf[i])
      self.sf[i] /= np.sum(self.sf[i])
    return self.sf

  def backprop(self):
    """
    Backward pass
    Input:
    - sf: softmax result
    Return:
    - d_x: input gradient
    """
    self.d_x = self.sf.copy()
    #print(self.sf)
    for i in range(self.N):
      self.d_x[i, self.label[i]] -= 1
    return self.d_x

if __name__ == "__main__":
  label = np.array([0,1,2,3])
  score = np.array([[0.1,-0.2,0.6,1.5],
                         [1.3,-1.5,0.5,-0.7],
                         [2.2,3.0,-0.1,1.4],
                         [-0.4,-0.3,-0.1,-.5]])
  loss1 = loss_layer(score.shape)
  batch_loss = loss1.loss_foward(score,label)
  loss_grad = loss1.gradient()
  print(loss_grad)