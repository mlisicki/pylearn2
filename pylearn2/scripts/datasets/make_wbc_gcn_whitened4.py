from pylearn2.utils import serial
from pylearn2.datasets import mike_preprocessing
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
from pylearn2.datasets.mat_data import MATDATA
import datetime
import time

data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/wbc')

print "Preprocessing data..."
start = time.time()

output_dir = data_dir + '/pylearn2_augmented_zca4'
serial.mkdir( output_dir )

preprocessor = mike_preprocessing.AugmentAndBalance()
preprocessor3 = preprocessing.ZCA()

zcatrain = MATDATA(path='/dos/study/research/cbc/data/20140206/classData06052014PythonMyeloidsOnlyTrain.mat', 
#zcatrain = MATDATA(path=data_dir+'/classData06052014PythonMyeloidsOnlyTrain.mat',
                which_set = 'full', start=0, stop=10000, step=4, one_hot=1, gcn=1.)
zcatrain.apply_preprocessor(preprocessor = preprocessor)
zcatrain.apply_preprocessor(preprocessor = preprocessor3, can_fit = True)
zcatrain = None

preprocessor2 = preprocessing.ShuffleAndSplit(seed=datetime.datetime.now().microsecond, 
                                              start=0, stop = 35000)

#train = MATDATA(path='/home/mike/data/wbc/classData06052014PythonMyeloidsOnlyTrain.mat',
train = MATDATA(path='/dos/study/research/cbc/data/20140206/classData06052014PythonMyeloidsOnlyTrain.mat', 
#train = MATDATA(path=data_dir+'/classData06052014PythonMyeloidsOnlyTrain.mat',
                which_set = 'full', start=0, stop=35000, step=4, one_hot=1, gcn=1.)
train.apply_preprocessor(preprocessor = preprocessor)
train.apply_preprocessor(preprocessor = preprocessor2)
train.apply_preprocessor(preprocessor = preprocessor3, can_fit = False)
train.use_design_loc(output_dir+'/train.npy')
serial.save(output_dir + '/train.pkl', train)
train = None

#test = MATDATA(path='/home/mike/data/wbc/classData06052014PythonMyeloidsOnlyTest.mat',
test = MATDATA(path='/dos/study/research/cbc/data/20140206/classData06052014PythonMyeloidsOnlyTest.mat', 
#test = MATDATA(path=data_dir+'/classData06052014PythonMyeloidsOnlyTest.mat',
               which_set = 'full', step=4, one_hot=1, gcn=1.)
test.apply_preprocessor(preprocessor = preprocessor)
test.apply_preprocessor(preprocessor = preprocessor3, can_fit = False)
test.use_design_loc(output_dir+'/test.npy')
serial.save(output_dir+'/test.pkl', test)

serial.save(output_dir + '/preprocessor.pkl',preprocessor3)

end = time.time()

print end-start