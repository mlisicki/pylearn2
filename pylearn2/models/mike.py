# by Michal Lisicki
from __future__ import division
#from scipy.io import loadmat,savemat
import theano
import theano.tensor as T
from pylearn2.models import mlp
from pylearn2.train import Train
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import VectorSpace
from pylearn2.utils import wraps
import numpy as np
from datetime import date
from random import randint
import os
from theano.compat.python2x import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams

class Linear(mlp.Layer):
    def __init__(self, dim, layer_name, istdev):
	# dim - dimensionality of layer aka number of hidden nodes
	super(Linear, self).__init__()
	
	self.__dict__.update(locals()) # set all the vars to self.var (e.g. self.dim=dim)
	del self.self
    
	#seed = date.today().timetuple()[0:3]
	self.rng = np.random.RandomState()
	#self.rng = RandomStreams(seed=seed)
    
    @wraps(mlp.Layer.set_input_space)
    def set_input_space(self, space):
	self.input_space = space
	
	if isinstance(space, VectorSpace):
	    self.requires_reformat = False
	    self.input_dim = space.dim
	else:
	    self.requires_reformat = True
	    self.input_dim = space.get_total_dimension()
	    self.desired_space = VectorSpace(self.input_dim)

	self.output_space = VectorSpace(self.dim)
	
	# we cannot set this in init() as we're not sure about input dimesnions yet
	if self.istdev is not None:
	    W = self.rng.randn(self.input_dim, self.dim) * self.istdev
	    b = self.rng.randn(self.dim,) * self.istdev
	else:
	    W = np.zeros((self.input_dim, self.dim))
	    b = np.zeros((self.dim,)) * self.istdev
	    
	self.W = theano.shared(theano._asarray(W,
	                        dtype=theano.config.floatX),
	                        name=(self.layer_name+'_W'))
	
	self.b = theano.shared(theano._asarray(b,
	                        dtype=theano.config.floatX),
	                        name=(self.layer_name + '_b'))		

    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):
	output = T.dot(state_below, self.W)
	output += self.b
	output.name = self.layer_name + '_z'
	return output
    
    @wraps(mlp.Layer.get_params)
    def get_params(self):
        return [self.W]
    
    @wraps(mlp.Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):
	return OrderedDict()
    
class Sigmoid(Linear):
    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):
	output = T.dot(state_below, self.W) + self.b
	output = T.nnet.sigmoid(output)
	output.name = self.layer_name + '_z'
	return output 
    
class Tanh(Linear):
    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):
	output = T.dot(state_below, self.W) + self.b
	output = T.tanh(output)
	output.name = self.layer_name + '_z'
	return output



#class RBMGB(Model):
  
    #def __init__(self, 
		 #nvis = None,
		 #nhid = None,
		 #irange = 0.5, # weight sigma for Gaussian weights initialization
		 #learning_rate = 0.01,
		 #weight_decay = 0.0001,
		 #sparse = 0.05,
		 #minibatch = 200,
		 #epochs = 50): # sparsity constraint
	#self.nvis = nvis,
	#self.nhid = nhid,
	#self.learning_rate = learning_rate,
	#self.weight_decay = weight_decay,
	#self.sparse = sparse
	#self.W = irange*np.randn(nvis,nhid)
	#self.dW = np.zeros(nvis,nhid) # previous delta w for momentum
	#self.epochs = epochs
    
	#self.trial=1
    
    #def train_all(self, dataset):	
	#error[(trial -1)] = 0
	#mbatchnum = np.round(dataset['data'].shape[0]/self.minibatch)
	#for m in range(1, (mbatchnum +1)):
	    #v = x[:, (np.dot((m - 1), minibatch) + 1 -1):np.dot(m, minibatch)]
	    ##v=v>rand(size(v));
	    ## or we just assume that they are binary?
	    #phv = 1.0 / (1 + exp(np.dot(- w[1:w.shape[0], :], np.array([np.ones(shape=(1, v.shape[1]), dtype='float64'), v]).reshape(1, -1))))
	    #rho_hat = mean(phv, 2)
	    #h = phv > rand(phv.shape)
	    ## sample from probability
	    #vh_data = np.dot(double(np.array([np.ones(shape=(1, h.shape[1]), dtype='float64'), phv]).reshape(1, -1)), double(np.array([np.ones(shape=(1, v.shape[1]), dtype='float64'), v]).reshape(1, -1)).T) / v.shape[1]
	    ## expectation of v_i*h_j over batch
	    #for j in range(1, 2):
		#pvh = np.dot(w[:, 1:w.shape[1]].T, np.array([np.ones(shape=(1, h.shape[1]), dtype='float64'), h]).reshape(1, -1))
		#v = pvh + randn(pvh.shape)
		#phv = 1.0 / (1 + exp(np.dot(- w[1:w.shape[0], :], np.array([np.ones(shape=(1, v.shape[1]), dtype='float64'), v]).reshape(1, -1))))
		#h = phv > rand(phv.shape)
		## sample from probability
	    #vh_model = np.dot(double(np.array([np.ones(shape=(1, h.shape[1]), dtype='float64'), phv]).reshape(1, -1)), double(np.array([np.ones(shape=(1, v.shape[1]), dtype='float64'), pvh]).reshape(1, -1)).T) / v.shape[1]
	    #rho = repmat(rho[0], h.shape[0], 1)
	    #dw = np.dot(- learningRate, (vh_model - vh_data)) - np.dot(lambda_, w) + np.dot(eta, dw) + repmat(np.dot(beta, np.array([1, binaryKLdq(rho, np.min(1 - eps, rho_hat + eps))]).reshape(1, -1)), 1, v.shape[0] + 1)
	    #w = w + dw
	    #error[(trial -1)] = error[(trial -1)] + np.sum(np.sum((vh_model - vh_data) ** 2)) / mbatchnum
	#error[(trial -1)]
	#if (visualize and mod(trial, 10) == 0):
	    ##filters=showData(w(2:end,2+10:end)./repmat(sqrt(sum(w(2:end,2+10:end).^2,2)),1,size(x,1)-10),10,1,3);
	    #filters = showData(w[1:w.shape[0], 1:w.shape[1]] / repmat(sqrt(np.sum(w[1:w.shape[0], 1:w.shape[1]] ** 2, 2)), 1, x.shape[0]), 10, 1, 3)
	    #filters = filters - np.min(np.min(filters))
	    #filters = filters / np.max(np.max(filters))
	    #imshow(filters)
	    #pause(0.1)
	    
	#self.trial += 1

    #def continue_learning(self):
      #if(self.trial>=self.epochs):
	#return False
      #else:
	#return True

#class LeNetConvPool(Layer):
    #def __init__(self,
                 #filters=32, 
                 #filterSize=41, 
                 #channels=4, 
                 #poolsize=2, 
                 #nvis = None, 
		 #nhid = None,
		 #irange = 0.5, # weight sigma for Gaussian weights initialization
		 #learning_rate = 0.01,
		 #weight_decay = 0.0001):

	#rng = np.random.RandomState(23455)

	#x = T.matrix('x')
	#y = T.ivector('y')
	
	#fan_in = nvis
	#fan_out = (filters * filterSize**2 / poolsize**2)	
	
        ## initialize weights with random weights
        #W_bound = np.sqrt(6. / (fan_in + fan_out))
        #self.W = theano.shared(np.asarray(
            #rng.uniform(low=-W_bound, high=W_bound, size=[filters, channels, filterSize, filterSize]),
            #dtype=theano.config.floatX),
                               #borrow=True)

        ## the bias is a 1D tensor -- one bias per output feature map
        #b_values = np.zeros((filters,), dtype=theano.config.floatX)
        #self.b = theano.shared(value=b_values, borrow=True)
	
        #x_shuffled = x.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        #filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
	
        #conv_op = FilterActs(stride=1, partial_sum=1)
        #contiguous_input = gpu_contiguous(x_shuffled)
        #contiguous_filters = gpu_contiguous(filters_shuffled)
        #conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)

        ## downsample each feature map individually, using maxpooling
        ## pooled_out = downsample.max_pool_2d(input=conv_out,
        ##                                     ds=poolsize, ignore_border=True)
        #pool_op = MaxPool(ds=poolsize[0], stride=poolsize[0])
        #pooled_out_shuffled = pool_op(conv_out_shuffled)
        #pooled_out = pooled_out_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01

        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        #self.params = [self.W, self.b]
	#self.gw, self.gb = T.grad(self.output,[self.W,self.b])
	
	#self.train = theano.function(
	                        #inputs=[x,y], 
	                        #outputs=self.output, 
	                        #updates=[(self.W, self.W - learning_rate * self.gw),
	                                 #(self.b, self.b - learning_rate * self.gb)])	
	
	    
    #def train_all(self, dataset):
	#cost = train_model(dataset)
	#self.trail+=1
	
    #def continue_learning(self):
      #if(self.trial>=self.epochs):
	#return False
      #else:
	#return True
     
    
    