import numpy as np 
import tensorflow as tf 
from utils import *

def train(kx=50,D=14*14,q='lap',n_steps=500*4):
	X=np.load('data/nature_data.npy').T 
	X,U,S=zca_whiten(X,kx)
	S=np.sqrt(S)
	X=X.T
	lr=tf.placeholder(tf.float32)

	W=tf.Variable(normr(tf.random_normal([D,kx], stddev=1,dtype=tf.float32)))
	X=tf.Variable(X,dtype=tf.float32)

	grad1,loss1=grad_loss_logqz(W,X,q)
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
				_lr=1
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
				vis(B.T,'imgs/na_or_D=%d_k=%d_%s.png'%(D,kx,q),14,14)
				B=U[:,:kx]
				B=np.dot(B,_W.T)
				vis(B.T,'imgs/na_wh_D=%d_k=%d_%s.png'%(D,kx,q),14,14)

def vis(filters,fn,nr=16,nc=49,space=2):
	filters=(filters-filters.min(1,keepdims=True))/(filters.max(1,keepdims=True)-filters.min(1,keepdims=True))
	fisz=int(math.sqrt(filters.shape[1]))
	canvas = np.zeros((fisz*nr+space*(nr-1), space*(nc-1)+fisz*nc))
	for i  in range(nr):
		for j in range(nc):
			k=i*nc+j
			if k>=filters.shape[0] or k>=nr*nc :break
			filter=filters[k,:]
			canvas[space*i+ (i)*fisz:space*i+(i+1)*fisz, space*j+j*fisz:space*j+(j+1)*fisz] \
					= filter.reshape(fisz, fisz)
	plt.figure(figsize=(8, 10))        
	plt.imshow(canvas, origin="upper", vmin=0, vmax=1,interpolation='none',cmap=plt.get_cmap('gray'))
	plt.tight_layout()  
	plt.savefig(fn,format='png', dpi=1200)


def generate_data():
	import scipy.io as sio
	_=sio.loadmat('data/nature.mat') # download from http://www.rctn.org/bruno/sparsenet/
	IMAGES=_['IMAGES']
	image_size=IMAGES.shape[0] 
	sz=14
	nature_data=np.zeros((sz*sz,50000));
	BUFF=4

	for t in range(50000):
		this_image=IMAGES[:,:,np.random.randint(IMAGES.shape[-1])];
		r=BUFF+int((image_size-sz-2*BUFF)*np.random.rand());
		c=BUFF+int((image_size-sz-2*BUFF)*np.random.rand());
		nature_data[:,t:t+1]=np.reshape(this_image[r:r+sz,c:c+sz],(sz*sz,1));
	X=nature_data
	np.save('data/nature_data.npy',nature_data)


if __name__ == '__main__':
	train()

