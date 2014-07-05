from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
from pylearn2.datasets.mat_data import MATDATA

data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/wbc')

output_dir = data_dir + '/pylearn2_gcn_whitened'
serial.mkdir( output_dir )

preprocessor = preprocessing.ZCA()

train = MATDATA(path='/home/mike/study/research/cbc/data/20140206/classData06052014PythonMyeloidsOnlyTrain.mat', which_set = 'full', gcn = 55., 
		start=0, stop=1000, step=2)
train.apply_preprocessor(preprocessor = preprocessor, can_fit = True)
train.use_design_loc(output_dir+'/train.npy')
serial.save(output_dir + '/train.pkl', train)

test = MATDATA(path='/home/mike/study/research/cbc/data/20140206/classData06052014PythonMyeloidsOnlyTest.mat', which_set = 'full', gcn = 55., step=2)
test.apply_preprocessor(preprocessor = preprocessor, can_fit = False)
test.use_design_loc(output_dir+'/test.npy')
serial.save(output_dir+'/test.pkl', test)

serial.save(output_dir + '/preprocessor.pkl',preprocessor)

