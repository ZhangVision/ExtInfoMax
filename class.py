import numpy as np 
import tensorflow as tf 
import data
import math
from utils import *


def train(kx=200,D=784):
	mnist_data=data.mnist()
	X=mnist_data.train.images
	X,U,S,zca_mu,zca_W=zca_whiten5(X,kx)
	S=np.sqrt(S)
	X=X.T
	lr=tf.placeholder(tf.float32)

	W=tf.Variable(normr(tf.random_normal([D,kx], stddev=1,dtype=tf.float32)))
	_y = tf.placeholder(tf.int32, [None])
	_x = tf.placeholder(tf.float32, [kx,None])

	grad1,loss1=grad_loss_logqz(W,_x,q='logistic',batch_size=60000)
	grad2,loss2=grad_loss_logdet(W)
	loss=loss1+loss2
	grad=grad1+grad2
	
	grad_norm=tf.reduce_mean(tf.reduce_sum(grad*grad,1)**0.5)
	grad=grad/(grad_norm+1e-5)
	op=[W.assign(normr(W-lr*grad)),loss1,loss2]

	with tf.variable_scope('clf') as scope:
		h0=tf.matmul(W,_x)
		z=tf.nn.relu(tf.concat([h0,-h0],0)) 

		decoder_W = tf.Variable(tf.random_normal([10,D*2], stddev=0.01)) #weight_variable([z_dim*2,10],1)
		decoder_b = tf.Variable(tf.random_normal([10,1], stddev=0.00001))
		logits=tf.matmul(decoder_W,z) + decoder_b
		logits=tf.transpose(logits)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=_y, name='xentropy')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean') + \
					0.0001*tf.nn.l2_loss(decoder_W)
	
	optimizer = tf.train.AdamOptimizer(0.001)
	grads=optimizer.compute_gradients(cross_entropy_mean,[decoder_W,decoder_b])
	train_op2=optimizer.apply_gradients(grads)


	with tf.Session() as sess:
			
		sess.run(tf.global_variables_initializer())
		n_steps=501*1
		for i in range(n_steps):
			if i<1000:
				_lr=.1
			elif i<2000:
				_lr=.1
			elif i<5000:
				_lr=.01
			logs=sess.run(op,feed_dict={lr:_lr,_x:X})
		  
			if i%10==0:
				  print(i,_lr,logs[-2],logs[-1])

			# if i%500==0:
			# 	_W=sess.run(W)
			# 	B=np.dot(U[:,:kx],np.diag(S[:kx]))
			# 	# B=U[:,:kx]
			# 	B=np.dot(B,_W.T)
			# 	vis(B.T,'or_D=%d_k=%d'%(D,kx))
				
			# 	B=U[:,:kx]
			# 	B=np.dot(B,_W.T)
			# 	vis(B.T,'wh_D=%d_k=%d'%(D,kx))

		for i in range(500*2):
			indices=np.random.permutation(X.shape[1])
			batch = [X[:,indices[:1000]],\
					mnist_data.train.labels[indices[:1000]]]
			  
			# batch=mnist_data.train.next_batch(1000)
			logs=sess.run([train_op2,cross_entropy_mean],\
				feed_dict={_y:batch[1],_x:batch[0]})
			if i%10==0:
				  print(i,logs[-1])

		# dW,db,_W=sess.run([decoder_W,decoder_b,W])
		tex=mnist_data.test.images
		test_data=np.dot(tex-zca_mu, zca_W)
		pred=sess.run(logits,feed_dict={_x:test_data.T})
		# test_data=np.dot(test_data, _W.T)

		# pred=np.dot(dW,test_data.T)+db
		# print pred.argmax(0) 
		print ("accuracy",(pred.argmax(1)==mnist_data.test.labels).sum()*1.0/mnist_data.test.labels.size)



if __name__ == '__main__':
	train()
