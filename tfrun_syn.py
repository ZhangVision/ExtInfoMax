import numpy as np 
import tensorflow as tf 
import data
import math
import matplotlib.pyplot as plt
from utils import *


def train(kx=10,D=9*9):

	_X0=np.load('data/syn_data0.npy') 
	X3=np.load('data/syn_data3.npy').T 
	X2=np.load('data/syn_data2.npy').T 
	X1=np.load('data/syn_data1.npy').T 
	 
	X0,U,S=zca_whiten(_X0,kx)
	S=np.sqrt(S)
	X0=X0.T
	lr=tf.placeholder(tf.float32)

	W=tf.Variable(normr(tf.random_normal([D,kx], stddev=1,dtype=tf.float32)))
	X=tf.Variable(X0,dtype=tf.float32)
	
	grad1,loss1=grad_loss_logqz(W,X,q='logistic')
	grad2,loss2=grad_loss_logdet(W)
	loss=loss1+loss2
	grad=grad1+grad2
	
	grad_norm=tf.reduce_mean(tf.reduce_sum(grad*grad,1)**0.5)
	grad=grad/(grad_norm+1e-5)
	op=[W.assign(normr(W-lr*grad)),loss1,loss2]

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		n_steps=501*3
		for i in range(n_steps):
			if i<500:
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
				
				B=U[:,:kx]
				B=np.dot(B,_W.T)

				WWnp=np.dot(_W.T,_W)+np.eye(kx)*1e-5
				pinvww=np.linalg.pinv(WWnp)
				iW=np.dot(U[:,:kx],np.diag(S[:kx]))
				iW=np.dot(iW,pinvww)
				iW=np.dot(iW,_W.T)

				Xdec=np.dot(_W,X0)
				Xdec=np.dot(iW,Xdec)
				# # print ((np.dot(iW,_W)-np.eye(81))**2).sum()
				batch_size=50000
				print (((Xdec+_X0.mean(0,keepdims=True).T-_X0.T)**2).sum()/batch_size)
				print (((Xdec+_X0.mean(0,keepdims=True).T-_X0.T)**2*X3).sum()/X3.sum())
				print (((Xdec+_X0.mean(0,keepdims=True).T-_X0.T)**2*X2).sum()/X2.sum())
				print (((Xdec+_X0.mean(0,keepdims=True).T-_X0.T)**2*X1).sum()/X1.sum())
				
				vis(B.T,'imgs/sy_wh_D=%d_k=%d.png'%(D,kx),int(D**0.5),int(D**0.5),1)
				vis(Xdec.T,'imgs/sy_dec_D=%d_k=%d.png'%(D,kx),int(D**0.5),int(D**0.5),1)
				vis(_X0,'imgs/sy_in_D=%d_k=%d.png'%(D,kx),int(D**0.5),int(D**0.5),1)

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
	from  scipy.ndimage.morphology import binary_dilation  
	from scipy.ndimage import generate_binary_structure  

	X,Y=9,9
	xs=np.arange(X).reshape(1,-1)
	ys=np.arange(Y).reshape(1,-1)
	xs=np.repeat(xs,Y,0)
	ys=np.repeat(ys,X,0)
	xc,yc=xs.T.astype(np.float32),ys.astype(np.float32)
	
	struct2 = generate_binary_structure(2, 2)

	def fill(img,szs,Xs):
		sz=szs[0]
		image_size=img.shape[0]
		r=np.random.randint(image_size-sz)
		c=np.random.randint(image_size-sz)
		img[r:r+sz,c:c+sz]=1.0
		Xs[0]=img==1

		for i,sz in enumerate(szs[1:]):
			# print sz
			validmask=binary_dilation(img,struct2,sz)

			zeroxc=xc[validmask==0].flatten()
			zeroyc=yc[validmask==0].flatten()
			validmask=np.logical_and(zeroxc<=image_size-sz,zeroyc<=image_size-sz) 
			ri=np.random.randint(validmask.sum())
			zeroxc=zeroxc[validmask]
			zeroyc=zeroyc[validmask]
			r=zeroxc[ri]
			c=zeroyc[ri]
			img[r:r+sz,c:c+sz]=1.0
			Xs[i+1][r:r+sz,c:c+sz] =True

		return img,Xs[0],Xs[1],Xs[2]

	N=50000
	X0=np.zeros((N,X,Y))

	X3=np.zeros((N,X,Y))
	X2=np.zeros((N,X,Y))
	X1=np.zeros((N,X,Y))
	for i in range(X0.shape[0]):
		X0[i,:,:],X3[i,:,:],X2[i,:,:],X1[i,:,:]=\
			fill(X0[i,:,:],[3,2,1],[X3[i,:,:],X2[i,:,:],X1[i,:,:]])
		
	print (X0.sum(-1).sum(-1)!=14).sum()
	print ((X0*X3).sum(-1).sum(-1)!=9).sum()
	print ((X0*X2).sum(-1).sum(-1)!=4).sum()
	print ((X0*X1).sum(-1).sum(-1)!=1).sum()

	vis(X0.reshape(X0.shape[0],-1),'sy',10,10,1)
	np.save('syn_data0.npy',X0.reshape(X0.shape[0],-1))
	np.save('syn_data3.npy',X3.reshape(X3.shape[0],-1))
	np.save('syn_data2.npy',X2.reshape(X2.shape[0],-1))
	np.save('syn_data1.npy',X1.reshape(X1.shape[0],-1))

if __name__ == '__main__':
	# generate_data()
	train()
	



	