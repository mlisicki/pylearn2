"""
This script makes a dataset of 32x32 contrast normalized, approximately
whitened CIFAR-10 images.

"""

from pylearn2.utils import serial
from pylearn2.datasets import mike_preprocessing
from pylearn2.utils import string_utils
from pylearn2.datasets.cifar10 import CIFAR10

data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/cifar10')

print 'Loading CIFAR-10 train dataset...'
train = CIFAR10(which_set = 'train', one_hot=1)

print "Preparing output directory..."
output_dir = data_dir + '/pylearn2_augmented'
serial.mkdir( output_dir )

print "Learning the preprocessor and preprocessing the unsupervised train data..."
preprocessor = mike_preprocessing.AugmentAndBalance()

train.apply_preprocessor(preprocessor = preprocessor)

print 'Saving the unsupervised data'
train.use_design_loc(output_dir+'/train.npy')
serial.save(output_dir + '/train.pkl', train)

print "Loading the test data"
test = CIFAR10(which_set = 'test')

print "Preprocessing the test data"
test.apply_preprocessor(preprocessor = preprocessor2, can_fit = False)

print "Saving the test data"
test.use_design_loc(output_dir+'/test.npy')
serial.save(output_dir+'/test.pkl', test)

serial.save(output_dir + '/preprocessor.pkl',preprocessor)

