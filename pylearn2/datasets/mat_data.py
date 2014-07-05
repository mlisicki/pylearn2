# by Michal Lisicki 2014
import numpy as np
import os
import logging
_logger = logging.getLogger(__name__)

import warnings
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import PyTables")

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrixPyTables
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import ensure_tables
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess
from pylearn2.expr.preprocessing import global_contrast_normalize
from theano import config

import data2python as d2py

class MATDATA(DenseDesignMatrix):
    """
    Translates standarized Matlab datasets to Pylearn
    In Matlab we define a structure with field dimensions:
    'data' -> sample x pixel x channel
    'label' -> sample
    'fileId' -> sample (name of the original file)
    'positions' -> sample x 4 (rectangle bounding box in the original image)
    Parameters:
      colorspace='none'(all channels, no preprocessing)|'rgb'(first three channels)
      step -> how to sample the data. e.g. '2' - every second pixel in 2D grid. This helps to 
	      quickly change adapt the dimensionality if there are memory or training speed issues
    """
    def __init__(self, 
	    which_set = 'full',
            path = 'train.mat',
            one_hot = False,
	    colorspace = 'none',
	    step = 1,
	    start = None, 
	    stop = None,
	    center = False, 
	    rescale = False,
	    gcn = None,
	    toronto_prepro = False,
            axes=('b', 0, 1, 'c')):

        self.__dict__.update(locals())
        del self.self	

        #
        #self.one_hot = one_hot
	#self.colorspace = colorspace
	#self.step=step
	#self.which_set=which_set
        
        self.view_converter = None

        self.path = preprocess(self.path)
        X, y = self._load_data()

	if center:
            X -= 127.5
        #self.center = center

        if rescale:
            X /= 127.5
        #self.rescale = rescale
        
        if toronto_prepro:
            assert not center
            assert not gcn
            X = X / 255.
            if which_set == 'test':
                other = MATDATA(which_set='train')
                oX = other.X
                oX /= 255.
                X = X - oX.mean(axis=0)
            else:
                X = X - X.mean(axis=0)
        #self.toronto_prepro = toronto_prepro

        #self.gcn = gcn
        if gcn is not None:
            gcn = float(gcn)
            X = global_contrast_normalize(X, scale=gcn, min_divisor=1e-8)
	    
	view_converter = DefaultViewConverter((
	    self.windowSize,self.windowSize,self.channels), axes)
        
        super(MATDATA, self).__init__(X=X, y=y, view_converter=view_converter)

    def _load_data(self):
        assert self.path.endswith('.mat')
	
	data=d2py.mat2py(self.path)
        
	y = data['labels']
	X = data['data']
	if(self.colorspace=='rgb'):
	    X=X[:,:,0:3]

	if self.start is not None and self.stop is not None:
	    X=X[self.start:self.stop,:]
	    y=y[self.start:self.stop]
	
	# apply step
	X=X.reshape((X.shape[0],np.sqrt(X.shape[1]),np.sqrt(X.shape[1]),-1),order='F')
	X=X[:,0::self.step,0::self.step,:]
	self.windowSize=X.shape[1]
	self.channels=X.shape[3]
	X=X.reshape((X.shape[0],pow(X.shape[1],2),X.shape[3]),order='F')
	
	# reshape each sample to 1D vector
	X=X.reshape((X.shape[0],-1),order='F')
	
	if(self.which_set!='full'):
	    Xs = {
		    'train' : X[0:round(0.9*X.shape[0]),:],
		    'test'  : X[round(0.9*X.shape[0])::,:]
		}

	    Ys = {
		    'train' : y[0:round(0.9*X.shape[0])],
		    'test'  : y[round(0.9*X.shape[0])::]
		}
	
	    X = np.cast['float32'](Xs[self.which_set])
	    y = Ys[self.which_set]
	
	# get unique labels and map them to one-hot positions
	labels = np.unique(y)
	labels = dict((x, i) for (i, x) in enumerate(labels))

	if self.one_hot:
	    one_hot = np.zeros((y.shape[0], len(labels)), dtype='float32')
	    for i in xrange(y.shape[0]):
		label = y[i]
		label_position = labels[label]
		one_hot[i,label_position] = 1.
	    y = one_hot

        return X, y


class MATDATAPyTables(DenseDesignMatrixPyTables):
    """
    Translates standarized Matlab datasets to Pylearn
    In Matlab we define a structure with field dimensions:
    'data' -> sample x pixel x channel
    'label' -> sample
    'fileId' -> sample (name of the original file)
    'positions' -> sample x 4 (rectangle bounding box in the original image)
    Parameters:
      colorspace='none'(all channels, no preprocessing)|'rgb'(first three channels)
      step -> how to sample the data. e.g. '2' - every second pixel in 2D grid. This helps to 
	      quickly change adapt the dimensionality if there are memory or training speed issues
    """
    def __init__(self, 
            path = 'train.mat',
	    start = None, 
	    stop = None,
	    center = False, 
	    rescale = False,
            axes=('b', 0, 1, 'c'),
            channels=4):

        self.__dict__.update(locals())
        del self.self	
	
	self.filters = tables.Filters(complib='blosc', complevel=5)
        
        self.view_converter = None

        self.path = preprocess(self.path)
	
        X, y = self._load_data()

	self.windowSize=np.uint8(np.sqrt(X.shape[1]/4))

	if center and rescale:
	    X[:] -= 127.5
	    X[:] /= 127.5
	elif center:
	    X[:] -= 127.5
	elif rescale:
	    X[:] /= 255.
	
	view_converter = DefaultViewConverter((
	    61,61,4), axes)
	
        super(MATDATAPyTables, self).__init__(X=X, y=y, view_converter=view_converter)
	
	self.h5file.flush()

    def _load_data(self):
        assert self.path.endswith('.mat')

	#data=d2py.mat2py(self.path)
	self.h5file = tables.openFile(self.path, mode = 'r+')
	data = self.h5file.getNode('/', "Data")

	if self.start != None or self.stop != None:
	    self.h5file, data = self.resize(self.h5file, self.start, self.stop)

        return data.X, data.y
    
    def resize(self, h5file, start, stop):
        ensure_tables()
        # TODO is there any smarter and more efficient way to this?

        data = h5file.getNode('/', "Data")
        try:
            gcolumns = h5file.createGroup('/', "Data_", "Data")
        except tables.exceptions.NodeError:
            h5file.removeNode('/', "Data_", 1)
            gcolumns = h5file.createGroup('/', "Data_", "Data")

        start = 0 if start is None else start
        stop = gcolumns.X.nrows if stop is None else stop

        atom = (tables.Float32Atom() if config.floatX == 'float32'
                else tables.Float64Atom())
        x = h5file.createCArray(gcolumns,
                                'X',
                                atom=atom,
                                shape=((stop - start, data.X.shape[1])),
                                title="Data values",
                                filters=self.filters)
        y = h5file.createCArray(gcolumns,
                                'y',
                                atom=atom,
                                shape=((stop - start, data.y.shape[1])),
                                title="Data targets",
                                filters=self.filters)
        x[:] = data.X[start:stop]
        y[:] = data.y[start:stop]

        h5file.removeNode('/', "Data", 1)
        h5file.renameNode('/', "Data", "Data_")
        h5file.flush()
        return h5file, gcolumns
    
