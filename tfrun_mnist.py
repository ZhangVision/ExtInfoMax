import numpy as np 
import tensorflow as tf 
import data
import math
from utils import *
		
def train(kx=200,D=784,n_steps=501):
	mnist_data=data.mnist()
	X=mnist_data.train.images
	X,U,S=zca_whiten(X,kx)
	S=np.sqrt(S)
	X=X.T
	lr=tf.placeholder(tf.float32)

	W=tf.Variable(normr(tf.random_normal([D,kx], stddev=1,dtype=tf.float32)))
	X=tf.Variable(X,dtype=tf.float32)

	grad1,loss1=grad_loss_logqz(W,X)
	grad2,loss2=grad_loss_logdet(W)
	loss=loss1+loss2
	grad=grad1+grad2
	
	grad_norm=tf.reduce_mean(tf.reduce_sum(grad*grad,1)**0.5)
	grad=grad/(grad_norm+1e-5)
	op=[W.assign(normr(W-lr*grad)),loss1,loss2]

	with tf.Session() as sess:
			
		sess.run(tf.global_variables_initializer())
		for i in range(n_steps):
			if i<1000:
				_lr=.1
			elif i<2000:
				_lr=.1
			elif i<5000:
				_lr=.01

			logs=sess.run(op,feed_dict={lr:_lr})
		  
			if i%10==0:
				  print(i,_lr,logs[-2],logs[-1])

			if i%500==0:
				_W=sess.run(W)
				B=np.dot(U[:,:kx],np.diag(S[:kx]))
				B=np.dot(B,_W.T)
				vis(B.T,'imgs/or_D=%d_k=%d.png'%(D,kx))
				B=U[:,:kx]
				B=np.dot(B,_W.T)
				vis(B.T,'imgs/wh_D=%d_k=%d.png'%(D,kx))

def vis(filters,fn):
	filters=(filters-filters.min( ))/(filters.max( )-filters.min( ))
	# np.save(fn+".npy",filters)

	space=2
	nx = ny =20# int(math.sqrt(z_dim))
	x_values = np.linspace(0, 4, nx)
	y_values = np.linspace(0, 4, ny)
	fisz=int(math.sqrt(filters.shape[1]))
	canvas = np.zeros((fisz*nx+space*nx, space*ny+fisz*ny))
	for i, yi in enumerate(x_values):
		for j, xi in enumerate(y_values):
			k=i*nx+j
			if k>=filters.shape[0] or k>=ny*nx :break
			filter=filters[k,:]
			canvas[space*i+ (i)*fisz:space*i+(i+1)*fisz, space*j+j*fisz:space*j+(j+1)*fisz] \
					= filter.reshape(fisz, fisz)
	plt.figure(figsize=(8, 10))        
	Xi, Yi = np.meshgrid(x_values, y_values)
	plt.imshow(canvas, origin="upper", vmin=0, vmax=1,interpolation='none',cmap=plt.get_cmap('gray'))
	plt.tight_layout()  
	plt.savefig(fn,format='png', dpi=1200)


if __name__ == '__main__':
	train()
