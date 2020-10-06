import numpy as np 
import tensorflow as tf 
import data
import math
import matplotlib.pyplot as plt


def normr(W):
	"""
	norm rows of W
	"""
	W1=tf.reduce_sum(W*W,1,keep_dims=True)**0.5
	return W/W1

def grad_loss_logqz(W,X,q='logistic',batch_size=50000):
	z=tf.matmul(W,X)
	if q=='logistic':
		sigm=tf.nn.sigmoid(z)
		qz=tf.sqrt(sigm)*(1-sigm)
		logqz=-tf.reduce_sum(tf.reduce_mean(tf.log(qz+1e-5),1)) 
		grad0=(1-2*sigm)
	elif q=='lap':
		logqz=tf.reduce_sum(tf.reduce_mean(tf.abs(z),1)) 
		grad0=-tf.sign(z)
	elif q=='t':
		logqz=tf.reduce_sum(tf.reduce_mean(tf.abs(z),1)) 
		grad0=-2*z/(1+z**2)
	elif q=='tanh':
		sigm=tf.nn.tanh(z)
		logqz=tf.reduce_sum(tf.reduce_mean(tf.abs(z),1)) 
		grad0=1-sigm**2

	logqz_grad=-tf.matmul(grad0,X,transpose_b=True)/batch_size
	return logqz_grad,logqz

def grad_loss_logdet(W):
	WW=tf.matmul(W,W,transpose_a=True)+tf.eye(W.get_shape()[1].value,dtype=tf.float32)*1e-5
	tri=tf.diag_part(tf.linalg.cholesky(WW)) 
	logdet=-tf.reduce_sum(tf.log(tri))
	pinv=tf.linalg.inv(WW)
	logdet_grad= -tf.matmul(W,pinv,transpose_b=True) 
	return logdet_grad,logdet


def zca_whiten(X,k_dim):
	EPS = 10e-5
	mu=np.mean(X,0,keepdims=True)
	X=X-mu
	cov = np.dot(X.T, X)/X.shape[0]
	E ,d, V = np.linalg.svd(cov)
	k=k_dim#(d>svt).sum()
	D = np.diag(1. / np.sqrt(d + EPS))
	W = np.dot(E[:,:k], D[:k,:k])
	X_white = np.dot(X, W)
	return X_white,E,d






	