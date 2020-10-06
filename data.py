from tensorflow.examples.tutorials.mnist import input_data

def mnist():
  mnist = input_data.read_data_sets('data/', one_hot=False,validation_size=0)
  return mnist

